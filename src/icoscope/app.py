"""Qt main window: 3D viewer + control panel + status bar."""
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QIcon, QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from . import export as _export
from . import menubar as _menubar
from .coastlines import coastline_polydata
from .controls import ControlPanel
from .graticule import graticule_polydata
from .grid import goldberg
from .lonlat import latlon_mesh
from .panes import PaneContainer
from .picker import Picker
from .playback import Playback
from .tabs import Tab
from .themes import CMAPS, THEMES
from .timeline import TimelineStrip


@dataclass
class PaneState:
    """Per-pane field/coloring state.

    For the Ico and LonLat tabs there is always exactly one pane (the
    synthetic mesh rendering). For the File tab there may be 1, 2, or
    4 panes in multi-pane layouts; each pane independently picks its
    own field (``color_by``), colormap, time/level indices, etc.
    """

    color_by: str = "None"
    cmap: str = "viridis"
    center_zero: bool = False
    colorbar_on: bool = True
    cbar_color_override: str | None = None
    time_index: int = 0
    level_index: int = 0
    # Per-pane time lock — when True, the master cursor (timeline drag
    # or other pane's slider scrub) does not update this pane's
    # time_index. Useful for "current vs baseline" comparisons where
    # one pane stays pinned to a fixed time while the others sweep.
    time_locked: bool = False


@dataclass
class _TabState:
    """Per-tab display state — overlays, widths, color overrides + per-pane list.

    Each tab in the side panel owns one of these so switching tabs is fully
    stateful: the user's choices on the Ico tab don't leak into the File tab
    and vice-versa.

    Per-pane fields (``color_by``, ``cmap``, ``time_index`` etc.) now live
    on :class:`PaneState`; this class holds the list of panes (always
    length 1 for Ico/LonLat; 1, 2, or 4 for File depending on the active
    layout). The tab-shared bits (overlays, theme overrides, file metadata)
    stay here.
    """

    coastlines_on: bool = False
    graticule_on: bool = False
    edges_on: bool = True
    spin_on: bool = False
    edge_color_override: str | None = None
    coast_color_override: str | None = None
    grat_color_override: str | None = None
    edge_width: float = 0.6
    coast_width: float = 1.2
    grat_width: float = 0.6
    # file-only fields
    file_fields: dict = field(default_factory=dict)
    # presnivs (Pa) for the loaded file, or None if no vertical dim
    file_levels: object = None
    # Master datetime cursor shared across panes. When a slider scrub
    # happens in any pane it sets this; every other pane resolves the
    # cursor to its own time axis's nearest sample so files with multiple
    # time dims (daily + monthly) stay aligned. ``None`` while no
    # time-varying field is on screen (also for the Ico/LonLat tabs).
    time_cursor: object = None
    # Playback loops by default — wraps to the start at the end of the time
    # axis. Unticking the Loop checkbox makes playback stop at the last
    # frame instead. Tab-shared (same semantics as autorotate).
    loop_playback: bool = True
    # Per-pane state. Default to a single pane; the File tab may grow this
    # list to 2 or 4 entries when the user picks a multi-pane layout.
    panes: list = field(default_factory=lambda: [PaneState()])

    # ── back-compat aliases for the per-pane fields ──────────────────────
    # Existing `state.color_by = ...` style accesses transparently read/write
    # the first pane. The multi-pane scaffold (later in this PR series) will
    # migrate explicit call sites to ``state.panes[i].X`` where the selected
    # pane index matters; these aliases keep single-pane behaviour intact
    # during the staged refactor.
    @property
    def color_by(self) -> str: return self.panes[0].color_by
    @color_by.setter
    def color_by(self, v: str) -> None: self.panes[0].color_by = v

    @property
    def cmap(self) -> str: return self.panes[0].cmap
    @cmap.setter
    def cmap(self, v: str) -> None: self.panes[0].cmap = v

    @property
    def center_zero(self) -> bool: return self.panes[0].center_zero
    @center_zero.setter
    def center_zero(self, v: bool) -> None: self.panes[0].center_zero = v

    @property
    def colorbar_on(self) -> bool: return self.panes[0].colorbar_on
    @colorbar_on.setter
    def colorbar_on(self, v: bool) -> None: self.panes[0].colorbar_on = v

    @property
    def time_index(self) -> int: return self.panes[0].time_index
    @time_index.setter
    def time_index(self, v: int) -> None: self.panes[0].time_index = v

    @property
    def level_index(self) -> int: return self.panes[0].level_index
    @level_index.setter
    def level_index(self, v: int) -> None: self.panes[0].level_index = v


class MainWindow(QMainWindow):
    """Top-level IcoScope window: 3D sphere view + right-side control panel."""

    def __init__(
        self,
        verts: np.ndarray,
        cells: list[list[int]],
        centers: np.ndarray,
        initial_n: int = 8,
        relax: bool = True,
        zoom_factor: float = 1.0,
        zoom_lon: float = 0.0,
        zoom_lat: float = 45.0,
        iim: int = 96,
        jjm: int = 95,
        lmdz_clon: float = 0.0,
        lmdz_clat: float = 0.0,
        lmdz_grossismx: float = 1.0,
        lmdz_grossismy: float = 1.0,
        lmdz_dzoomx: float = 0.0,
        lmdz_dzoomy: float = 0.0,
        lmdz_taux: float = 3.0,
        lmdz_tauy: float = 3.0,
    ):
        super().__init__()
        self.setWindowTitle("IcoScope")
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.resize(1320, 840)

        # geometry state (window-level)
        self.verts = verts
        self.cells = cells
        self.centers = np.asarray(centers)
        # Per-pane scalar arrays. Index i holds the rendered field for the
        # i-th pane (or None when that pane has no field selected). The
        # back-compat property ``self.scalars`` (below) aliases the
        # currently-selected pane's array — falls back to pane 0 when no
        # selection has been made yet (single-pane behaviour).
        self._pane_scalars: list = [None] * PaneContainer.MAX_PANES
        self._selected_pane: int | None = None
        # Remember the File tab's last pane layout so we can restore it when
        # the user comes back from Ico / LonLat (we collapse to Single on
        # leave so non-File tabs don't render their synthetic mesh into
        # multiple panes; without this the File-tab layout would be lost).
        self._file_layout: int = 1
        # Camera-sync state. When True (default), rotating / panning /
        # zooming any pane mirrors the transform onto all other visible
        # panes via VTK ModifiedEvent observers installed below.
        # ``_syncing_cameras`` is the re-entrancy guard: while we're
        # propagating a camera change to the other panes, the observers
        # see _their_ cameras change and would loop without it.
        self._camera_sync_on: bool = True
        self._syncing_cameras: bool = False
        # Last successful pick: (cell_idx, lon, lat) or None. Used to refresh
        # the status-bar value display after time/level slider scrubs without
        # forcing the user to re-pick.
        self._last_pick: tuple[int, float, float] | None = None
        self.file_path = None
        self.n = initial_n
        self.max_relax_iters = 200 if relax else 0
        self.zoom_factor = float(zoom_factor)
        self.zoom_lon = float(zoom_lon)
        self.zoom_lat = float(zoom_lat)
        # LonLat-tab synthetic mesh size (LMDZ low-res defaults).
        self.iim = int(iim)
        self.jjm = int(jjm)
        # LonLat-tab LMDZ tanh-zoom parameters (identity by default).
        self.lmdz_clon = float(lmdz_clon)
        self.lmdz_clat = float(lmdz_clat)
        self.lmdz_grossismx = float(lmdz_grossismx)
        self.lmdz_grossismy = float(lmdz_grossismy)
        self.lmdz_dzoomx = float(lmdz_dzoomx)
        self.lmdz_dzoomy = float(lmdz_dzoomy)
        self.lmdz_taux = float(lmdz_taux)
        self.lmdz_tauy = float(lmdz_tauy)
        self._export_defaults = _export.ExportDefaults()

        # Theme is window-level (background colour + default overlay tints).
        self.theme_name = "Dark"
        default_cmap = THEMES[self.theme_name]["cmap"]

        # Per-tab display state — each tab keeps its own coloring, overlays,
        # animation, and color-by selection. Switching tabs swaps which
        # state is read by the rendering code (via ``self.state``). cmap
        # is set after construction because it now lives on PaneState
        # (via the panes[0] back-compat property).
        self._ico_state = _TabState()
        self._lonlat_state = _TabState()
        self._file_state = _TabState()
        for state in (self._ico_state, self._lonlat_state, self._file_state):
            state.cmap = default_cmap

        # central layout: a horizontal splitter so the user can drag the
        # divider between the 3-D view(s) and the control panel. Index 0
        # (the PaneContainer) carries the stretch on window resize; index 1
        # (the panel) has a minimum width to stay readable.
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(4)
        # Multi-pane scaffold: the central viewport is now a PaneContainer
        # that holds up to 4 panes. Stage 2 always shows exactly one pane
        # (pane 0) so single-pane behaviour is unchanged. Stage 3 adds the
        # View menu to switch between 1 / 1×2 / 2×2 layouts.
        # PaneContainer + TimelineStrip sit in a vertical container so the
        # strip docks under the viewports. The container goes into the main
        # horizontal splitter (where PaneContainer used to live directly).
        viewport_area = QWidget(self.splitter)
        va_layout = QVBoxLayout(viewport_area)
        va_layout.setContentsMargins(0, 0, 0, 0)
        va_layout.setSpacing(0)
        self._pane_container = PaneContainer(viewport_area)
        self._pane_container.pane_clicked.connect(self._on_pane_clicked)
        va_layout.addWidget(self._pane_container, stretch=1)
        self._timeline_strip = TimelineStrip(viewport_area)
        self._timeline_strip.cursor_changed.connect(self._on_timeline_cursor_changed)
        self._timeline_strip.pane_selected.connect(self._on_pane_clicked)
        self._timeline_strip.lock_toggle_requested.connect(
            self._on_timeline_lock_toggled)
        va_layout.addWidget(self._timeline_strip)
        self.splitter.addWidget(viewport_area)
        self.plotter = self._pane_container.pane(0).plotter
        # Install VTK ModifiedEvent observers on every pane's camera so we
        # can mirror movements across visible panes in sync mode. Done once
        # here (rather than every redraw) since PaneContainer eagerly
        # creates all MAX_PANES Pane widgets at construction; the cameras
        # live for the window's lifetime.
        for i in range(PaneContainer.MAX_PANES):
            cam = self._pane_container.pane(i).plotter.renderer.GetActiveCamera()
            cam.AddObserver(
                "ModifiedEvent",
                lambda c, e, src=i: self._on_camera_modified(src),
            )
        self.panel = ControlPanel(CMAPS)
        self.splitter.addWidget(self.panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        # Use a wide initial plotter allocation; the panel takes its preferred
        # width and the splitter clamps to its minimum on shrink.
        self.splitter.setSizes([1000, ControlPanel.DEFAULT_WIDTH])
        self.setCentralWidget(self.splitter)

        self._theme_actions, self._layout_actions = _menubar.build_menubar(
            self, self.theme_name, self._on_theme,
            on_pane_layout=self._on_pane_layout,
            on_reset_cameras=self._on_reset_cameras,
        )
        self._build_lonlat_widget()
        self.statusBar().showMessage("ready")

        # initial sync of panel widgets
        ico = self.panel.ico_tab
        ico.n_box.blockSignals(True)
        ico.n_box.setValue(initial_n)
        ico.n_box.blockSignals(False)
        ico.relax_iters_box.blockSignals(True)
        ico.relax_iters_box.setValue(self.max_relax_iters)
        ico.relax_iters_box.blockSignals(False)
        ico.set_zoom(self.zoom_factor, self.zoom_lon, self.zoom_lat)
        lonlat = self.panel.lonlat_tab
        lonlat.iim_box.blockSignals(True)
        lonlat.iim_box.setValue(self.iim)
        lonlat.iim_box.blockSignals(False)
        lonlat.jjm_box.blockSignals(True)
        lonlat.jjm_box.setValue(self.jjm)
        lonlat.jjm_box.blockSignals(False)
        lonlat.set_lmdz_zoom(
            self.lmdz_clon, self.lmdz_clat,
            self.lmdz_grossismx, self.lmdz_grossismy,
            self.lmdz_dzoomx, self.lmdz_dzoomy,
            self.lmdz_taux, self.lmdz_tauy,
        )
        display_tabs = (self.panel.ico_tab, self.panel.lonlat_tab, self.panel.file_tab)
        for tab in display_tabs:
            tab.set_cmap(default_cmap)
        self._sync_color_buttons()

        # wire signals — display controls live on each tab independently
        for tab in display_tabs:
            tab.cmap_changed.connect(self._on_cmap)
            tab.coastlines_toggled.connect(self._on_coast)
            tab.graticule_toggled.connect(self._on_grat)
            tab.edges_toggled.connect(self._on_edges)
            tab.colorbar_toggled.connect(self._on_colorbar)
            tab.color_by_changed.connect(self._on_color_by)
            tab.center_zero_toggled.connect(self._on_center_zero)
            tab.edge_color_changed.connect(self._on_edge_color)
            tab.coast_color_changed.connect(self._on_coast_color)
            tab.grat_color_changed.connect(self._on_grat_color)
            tab.cbar_color_changed.connect(self._on_cbar_color)
            tab.edge_width_changed.connect(self._on_edge_width)
            tab.coast_width_changed.connect(self._on_coast_width)
            tab.grat_width_changed.connect(self._on_grat_width)
            tab.autorotate_toggled.connect(self._on_spin)
            tab.sync_cameras_toggled.connect(self._on_camera_sync)
            tab.export_clicked.connect(self._on_export)

        # Tab-specific signals
        self.panel.ico_tab.n_changed.connect(self._on_n)
        self.panel.ico_tab.relax_iters_changed.connect(self._on_relax_iters)
        self.panel.ico_tab.zoom_changed.connect(self._on_zoom)
        self.panel.lonlat_tab.iim_changed.connect(self._on_iim)
        self.panel.lonlat_tab.jjm_changed.connect(self._on_jjm)
        self.panel.lonlat_tab.lmdz_zoom_changed.connect(self._on_lmdz_zoom)
        self.panel.file_tab.open_file_clicked.connect(self._on_open_file)
        self.panel.file_tab.close_file_clicked.connect(self._on_close_file)
        self.panel.file_tab.time_changed.connect(self._on_time_changed)
        self.panel.file_tab.level_changed.connect(self._on_level_changed)
        self.panel.file_tab.play_toggled.connect(self._on_play_toggled)
        self.panel.file_tab.play_speed_changed.connect(self._on_play_speed_changed)
        self.panel.file_tab.loop_toggled.connect(self._on_loop_toggled)
        self.panel.tabs.currentChanged.connect(self._on_tab_changed)

        # Cached meshes so tab-switching doesn't trigger expensive recomputes.
        self._file_cache: dict | None = None
        self._ico_cache: dict | None = None
        self._lonlat_cache: dict | None = None

        # build scene + interactions
        self._refresh_scalars()
        self._mesh = self._to_polydata()

        # Helpers — instantiate after _mesh exists; the picker reads it lazily.
        # One Picker per pane: all four are created upfront (matching
        # PaneContainer's eager pane creation), so layout switches don't have
        # to reattach pickers.
        self._pickers = [Picker(self, self._pane_container.pane(i).plotter, i)
                         for i in range(PaneContainer.MAX_PANES)]
        self.playback = Playback(self, self.plotter)

        self._build_scene()
        self.plotter.reset_camera()
        for p in self._pickers:
            p.attach()
        self._apply_stylesheet()
        QShortcut(QKeySequence("Escape"), self, activated=self._on_escape)
        self._update_status()

    # ── per-tab state plumbing ─────────────────────
    @property
    def state(self) -> _TabState:
        """Return the ``_TabState`` for the currently-active tab."""
        idx = self.panel.tabs.currentIndex()
        if idx == Tab.ICO:
            return self._ico_state
        if idx == Tab.FILE:
            return self._file_state
        return self._lonlat_state

    @property
    def _active_pane_idx(self) -> int:
        """Index of the pane the picker / single-scalar code path acts on.

        For the File tab in multi-pane mode this is the user-selected pane;
        elsewhere (or before any selection), pane 0.
        """
        idx = self._selected_pane
        if idx is None:
            return 0
        if idx >= len(self.state.panes):
            return 0
        return idx

    @property
    def pane_state(self) -> PaneState:
        """Active pane's :class:`PaneState` (selected, or pane 0 fallback)."""
        return self.state.panes[self._active_pane_idx]

    @property
    def scalars(self):
        """Back-compat alias for the active pane's scalar array.

        Single-pane code paths (picker value display, the global ``_clim``
        helper, etc.) read this; in multi-pane these will move to per-pane
        helpers in stages 5–6.
        """
        return self._pane_scalars[self._active_pane_idx]

    @scalars.setter
    def scalars(self, value) -> None:
        self._pane_scalars[self._active_pane_idx] = value

    @property
    def active_tab(self):
        """Return the currently-active tab widget (Ico, LonLat, or File)."""
        idx = self.panel.tabs.currentIndex()
        if idx == Tab.ICO:
            return self.panel.ico_tab
        if idx == Tab.FILE:
            return self.panel.file_tab
        return self.panel.lonlat_tab

    # ── helpers ───────────────────────────────────
    def _times_for(self, meta):
        """Return the cached datetime array for the field's time axis, or None.

        Looks up the active FileContext (held-open Dataset) and asks it for
        ``meta["time_dim_name"]``'s parsed datetimes. Falls back to ``None``
        when there is no context (Ico/LonLat tabs, file not loaded) or when
        the axis has no coord variable.
        """
        if not meta or not meta.get("time_dim_name"):
            return None
        ctx = self._file_cache.get("context") if self._file_cache else None
        if ctx is None:
            return None
        return ctx.get_times(meta["time_dim_name"])

    # ── colors ─────────────────────────────────────
    def _edge_color(self):
        return self.state.edge_color_override or THEMES[self.theme_name]["edge"]

    def _coast_color(self):
        return self.state.coast_color_override or THEMES[self.theme_name]["coast"]

    def _grat_color(self):
        return self.state.grat_color_override or THEMES[self.theme_name].get(
            "grat", self._coast_color())

    def _cbar_color(self, pane_idx: int | None = None):
        """Per-pane colorbar text colour (override or theme default)."""
        idx = self._active_pane_idx if pane_idx is None else pane_idx
        pane = self.state.panes[idx]
        return pane.cbar_color_override or THEMES[self.theme_name].get(
            "cbar", "white")

    def _sync_color_buttons(self):
        hex_edge = self._color_to_hex(self._edge_color())
        hex_coast = self._color_to_hex(self._coast_color())
        hex_grat = self._color_to_hex(self._grat_color())
        hex_cbar = self._color_to_hex(self._cbar_color())
        for tab in (self.panel.ico_tab, self.panel.lonlat_tab, self.panel.file_tab):
            tab.set_edge_color(hex_edge)
            tab.set_coast_color(hex_coast)
            tab.set_grat_color(hex_grat)
            tab.set_cbar_color(hex_cbar)

    @staticmethod
    def _color_to_hex(c):
        # accept "#rrggbb" or named CSS colors
        from qtpy.QtGui import QColor
        return QColor(c).name()

    # ── geometry ──────────────────────────────────
    @staticmethod
    def _pane_scalar_key(idx: int) -> str:
        """Name of the per-pane scalar array stored on the shared mesh."""
        return f"pane{idx}_scalars"

    def _to_polydata(self):
        """Build a fresh PolyData from ``self.verts/cells`` + per-pane scalars.

        Expensive on a 64 k-cell mesh because of the Python loop assembling
        the ``faces_flat`` array. Only call when geometry actually changed
        (verts or cells differ); for scalar-only swaps (slider scrubs) use
        :meth:`_update_scalars_only`, which mutates the existing PolyData
        in place and avoids the loop.

        Each pane's scalar array is bound to its own named key on the
        polydata so panes can independently colour the shared geometry.
        Stale arrays whose length doesn't match the new cell count are
        skipped (and the slot cleared) — happens on tab switch when the
        previous tab's pane scalars carry the wrong shape.
        """
        faces_flat = []
        for c in self.cells:
            faces_flat.append(len(c))
            faces_flat.extend(c)
        mesh = pv.PolyData(self.verts, faces=np.array(faces_flat, dtype=np.int64))
        n_cells = len(self.cells)
        for i, arr in enumerate(self._pane_scalars):
            if arr is None:
                continue
            if len(arr) != n_cells:
                # Stale entry from a previous mesh — drop it instead of
                # crashing in vtk's array-length check.
                self._pane_scalars[i] = None
                continue
            mesh.cell_data[self._pane_scalar_key(i)] = np.asarray(arr)
        return mesh

    def _update_scalars_only(self, pane_idx: int | None = None) -> None:
        """Swap one pane's scalars on the cached PolyData in place.

        Used in the slider-scrub hot path where geometry hasn't changed
        between frames. ``pane_idx=None`` updates the active pane only;
        pass an explicit index to update a different pane (used during
        layout-change re-renders).
        """
        if self._mesh is None:
            self._mesh = self._to_polydata()
            return
        idx = self._active_pane_idx if pane_idx is None else pane_idx
        key = self._pane_scalar_key(idx)
        arr = self._pane_scalars[idx]
        if arr is None:
            if key in self._mesh.cell_data:
                del self._mesh.cell_data[key]
        else:
            self._mesh.cell_data[key] = np.asarray(arr)

    def _clim(self, pane_idx: int | None = None):
        """Colour-limit range for a pane's scalar array (or None for auto)."""
        idx = self._active_pane_idx if pane_idx is None else pane_idx
        arr = self._pane_scalars[idx]
        pane = self.state.panes[idx]
        if arr is None or not pane.center_zero:
            return None
        s = np.asarray(arr)
        a = float(np.nanmax(np.abs(s)))
        return [-a, a] if a > 0 else None

    # ── rendering ─────────────────────────────────
    def _build_scene(self):
        """Render the current scene on every visible pane.

        Each pane gets its own actor stack on its own ``QtInteractor``
        (the shared mesh is rendered N times with N different scalar
        bindings, plus a copy of the overlay actors per pane). Defers to
        the empty-sphere fallback when there's no mesh yet.
        """
        if self._mesh is None:
            self._render_empty_sphere()
            return
        for idx in range(self._pane_container.n_visible):
            self._build_pane_scene(idx)

    def _build_pane_scene(self, pane_idx: int) -> None:
        """Render the scene on pane ``pane_idx``'s plotter.

        Each pane reads its own :class:`PaneState` (``cmap``, ``center_zero``,
        ``colorbar_on``) and binds to its own per-pane scalar key on the
        shared mesh. Overlays (coastlines, graticule, edges, theme) are
        tab-shared so every visible pane gets the same overlay actors.
        """
        plotter = self._pane_container.pane(pane_idx).plotter
        plotter.remove_actor("empty", reset_camera=False, render=False)
        # PyVista keeps the existing scalar-bar actor across add_mesh calls
        # and only honours scalar_bar_args on first creation. Drop it so
        # the new title / label / colour / format actually takes effect.
        # Suppress everything: PyVista's remove_scalar_bar raises various
        # exceptions when there's no bar (KeyError, IndexError, or
        # StopIteration depending on version) and the "nothing to remove"
        # case is benign — first call on a fresh plotter, mostly.
        import contextlib
        with contextlib.suppress(KeyError, IndexError, StopIteration):
            plotter.remove_scalar_bar()
        theme = THEMES[self.theme_name]
        plotter.set_background(theme["bg"])
        st = self.state                          # tab-shared (overlays, theme)
        pane = self.state.panes[pane_idx]        # per-pane (cmap, etc.)
        has_scalars = self._pane_scalars[pane_idx] is not None
        scalar_key = self._pane_scalar_key(pane_idx) if has_scalars else None

        plotter.add_mesh(
            self._mesh, name="grid",
            scalars=scalar_key,
            cmap=pane.cmap,
            clim=self._clim(pane_idx),
            show_edges=st.edges_on,
            edge_color=self._edge_color(),
            line_width=st.edge_width,
            smooth_shading=False,
            show_scalar_bar=pane.colorbar_on and has_scalars,
            scalar_bar_args=({"color": self._cbar_color(pane_idx),
                              "title_font_size": 12,
                              "label_font_size": 10,
                              "fmt": "%.3g",
                              # Match the side-panel "Pane N settings" header
                              # which is also 1-indexed (was previously
                              # 0-indexed via the raw scalar-key default).
                              "title": f"Pane {pane_idx + 1}"}
                             if pane.colorbar_on and has_scalars else None),
            reset_camera=False,
        )

        if st.coastlines_on:
            try:
                cl = coastline_polydata(radius=1.001)
                plotter.add_mesh(cl, name="coast",
                                 color=self._coast_color(),
                                 line_width=st.coast_width,
                                 pickable=False, reset_camera=False)
            except Exception as e:
                self.statusBar().showMessage(f"coastlines failed: {e}")
        else:
            plotter.remove_actor("coast", reset_camera=False, render=False)

        if st.graticule_on:
            try:
                g = graticule_polydata(radius=1.002, spacing=30)
                plotter.add_mesh(g, name="grat",
                                 color=self._grat_color(),
                                 line_width=st.grat_width,
                                 opacity=0.6, pickable=False, reset_camera=False)
            except Exception as e:
                self.statusBar().showMessage(f"graticule failed: {e}")
        else:
            plotter.remove_actor("grat", reset_camera=False, render=False)

        plotter.render()

    # ── ESC: clear current selection + stop spin ──
    def _on_escape(self):
        self._clear_pick_state(render=True)
        # Deselect any active multi-pane selection so the side panel can
        # swap back to Global mode (stage 6 wires that up).
        self._select_pane(None)
        if self.state.spin_on:
            self.state.spin_on = False
            self.playback.stop_spin()
            # Uncheck on whichever tab's spin checkbox is currently checked.
            # The File tab's spin_cb lives on display_global (Auto-rotate is a
            # tab-shared setting and only the global block in mode='pane' or
            # 'global' has the checkbox); Ico / LonLat use the combined block
            # where the checkbox is on `display`.
            for tab in (self.panel.ico_tab, self.panel.lonlat_tab, self.panel.file_tab):
                source = getattr(tab, "display_global", None) or tab.display
                cb = getattr(source, "spin_cb", None)
                if cb is not None and cb.isChecked():
                    cb.blockSignals(True)
                    cb.setChecked(False)
                    cb.blockSignals(False)
        self.plotter.render()

    # ── Qt stylesheet for the panel ───────────────
    def _apply_stylesheet(self):
        self.setStyleSheet("""
        QGroupBox {
            font-weight: 600;
            border: 1px solid #888;
            border-radius: 5px;
            margin-top: 12px;
            padding: 6px 4px 4px 4px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QPushButton {
            padding: 4px 10px;
        }
        QLabel, QCheckBox, QComboBox, QSpinBox {
            padding: 1px 2px;
        }
        QStatusBar {
            border-top: 1px solid #888;
        }
        """)

    # ── status bar ────────────────────────────────
    def _update_status(self):
        idx = self.panel.tabs.currentIndex()
        n_cells = len(self.cells) if self.cells else 0
        if idx == Tab.ICO:
            msg = f"Ico mesh: n={self.n}, {n_cells:,} cells"
        elif idx == Tab.LONLAT:
            msg = f"LonLat mesh: iim={self.iim} jjm={self.jjm}, {n_cells:,} cells"
        elif idx == Tab.FILE:
            if self.file_path:
                msg = f"file: {os.path.basename(self.file_path)} ({n_cells:,} cells)"
            else:
                msg = "no file loaded"
        else:
            msg = ""
        self.statusBar().showMessage(msg)

    # ── lon/lat editable widget on the status bar ─
    LON_SENTINEL = -200.0
    LAT_SENTINEL = -100.0

    def _build_lonlat_widget(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 0, 4, 0)
        h.setSpacing(4)
        h.addWidget(QLabel("Lon"))
        self.lon_box = QDoubleSpinBox()
        self.lon_box.setRange(self.LON_SENTINEL, 180)
        self.lon_box.setDecimals(2)
        self.lon_box.setSuffix("°")
        self.lon_box.setSpecialValueText("—")
        self.lon_box.setFixedWidth(80)
        self.lon_box.setKeyboardTracking(False)
        self.lon_box.setValue(self.LON_SENTINEL)
        h.addWidget(self.lon_box)
        h.addWidget(QLabel("Lat"))
        self.lat_box = QDoubleSpinBox()
        self.lat_box.setRange(self.LAT_SENTINEL, 90)
        self.lat_box.setDecimals(2)
        self.lat_box.setSuffix("°")
        self.lat_box.setSpecialValueText("—")
        self.lat_box.setFixedWidth(80)
        self.lat_box.setKeyboardTracking(False)
        self.lat_box.setValue(self.LAT_SENTINEL)
        h.addWidget(self.lat_box)
        # Value label — populated by _set_cell_value when the picker hits a
        # cell and the active field has a meaningful scalar. Hidden until then.
        self.value_label = QLabel("")
        self.value_label.setVisible(False)
        h.addWidget(self.value_label)
        self.statusBar().addPermanentWidget(w)
        # editing either field flies the camera
        self.lon_box.editingFinished.connect(self._fly_to_lonlat)
        self.lat_box.editingFinished.connect(self._fly_to_lonlat)

    def _set_lonlat(self, lon=None, lat=None):
        for box, val, sentinel in (
            (self.lon_box, lon, self.LON_SENTINEL),
            (self.lat_box, lat, self.LAT_SENTINEL),
        ):
            box.blockSignals(True)
            box.setValue(sentinel if val is None else float(val))
            box.blockSignals(False)

    def _clear_lonlat(self):
        self._set_lonlat(None, None)

    def _current_color_by_units(self) -> str:
        """Units string for the active field — '' if unknown."""
        from .display_block import SYNTHETIC_UNITS
        name = self.state.color_by
        if name == "None":
            return ""
        # File-tab fields have units in their FieldMeta; synthetic schemes use
        # the hardcoded SYNTHETIC_UNITS table.
        meta = self._file_state.file_fields.get(name) if self.file_path else None
        if meta is not None:
            return str(meta.get("units", "") or "")
        return SYNTHETIC_UNITS.get(name, "")

    @staticmethod
    def _format_cell_value(value, units: str) -> tuple[str, str]:
        """Forward to :func:`icoscope.formatters.format_cell_value`.

        Kept as a method so existing call sites in this module stay short.
        """
        from .formatters import format_cell_value
        return format_cell_value(value, units)

    def _set_cell_value(self, cell_idx, lon=None, lat=None):
        """Update the status-bar value label for the picked cell.

        Pass ``cell_idx=None`` to clear. ``lon``/``lat`` are the exact
        click-resolved coordinates (in degrees); when provided they go
        into the tooltip alongside the full-precision value, and the
        ``(cell_idx, lon, lat)`` tuple is cached so ``_refresh_picked_value``
        can re-display after time / level scrubs.
        """
        if (cell_idx is None or self.scalars is None
                or self.state.color_by == "None"):
            self.value_label.setText("")
            self.value_label.setToolTip("")
            self.value_label.setVisible(False)
            self._last_pick = None
            return
        if cell_idx >= len(self.scalars):
            # Mesh changed underneath us — the saved cell index doesn't
            # apply any more. Drop the pick rather than silently showing
            # a wrong value.
            self._last_pick = None
            self.value_label.setText("")
            self.value_label.setToolTip("")
            self.value_label.setVisible(False)
            return
        val = self.scalars[cell_idx]
        units = self._current_color_by_units()
        short, full = self._format_cell_value(val, units)
        self.value_label.setText(short)
        tip = full
        if lon is not None and lat is not None:
            tip = f"{full}\ncell {cell_idx}, lon {lon:.6f}°, lat {lat:.6f}°"
            self._last_pick = (cell_idx, lon, lat)
        self.value_label.setToolTip(tip)
        self.value_label.setVisible(True)

    def _clear_cell_value(self):
        self._set_cell_value(None)

    def _refresh_picked_value(self):
        """Re-display the value for the currently-picked cell after scalars change.

        Called from the time / level slider handlers — the highlight ring and
        lon/lat stay put, but the value at that cell is now from a different
        slice, so the displayed number needs to refresh.
        """
        if self._last_pick is None:
            return
        idx, lon, lat = self._last_pick
        self._set_cell_value(idx, lon=lon, lat=lat)

    def _fly_to_lonlat(self):
        lon_v = self.lon_box.value()
        lat_v = self.lat_box.value()
        if lon_v == self.LON_SENTINEL or lat_v == self.LAT_SENTINEL:
            return
        lon = np.radians(lon_v)
        lat = np.radians(lat_v)
        direction = np.array([np.cos(lat) * np.cos(lon),
                              np.cos(lat) * np.sin(lon),
                              np.sin(lat)])
        cam = self.plotter.camera
        pos = np.asarray(cam.position, dtype=float)
        dist = np.linalg.norm(pos) or 3.0
        cam.position = (direction * dist).tolist()
        cam.focal_point = (0.0, 0.0, 0.0)
        cam.up = (0.0, 0.0, 1.0)
        self.plotter.render()

    # ── slots ─────────────────────────────────────
    def _on_camera_modified(self, src_pane_idx: int) -> None:
        """Mirror pane ``src_pane_idx``'s camera onto every other visible pane.

        Fired by VTK's ``ModifiedEvent`` on the source camera whenever it
        changes (mouse rotate, pan, zoom, programmatic moves). Skips when:
        - sync is OFF (each pane stays independent)
        - source pane is hidden (we only mirror across *visible* panes)
        - propagation is already in progress (re-entrancy guard — without
          this every mirrored camera fires its own ModifiedEvent and loops)
        """
        if not self._camera_sync_on or self._syncing_cameras:
            return
        if src_pane_idx >= self._pane_container.n_visible:
            return
        src_cam = (
            self._pane_container.pane(src_pane_idx).plotter.renderer
            .GetActiveCamera()
        )
        pos = src_cam.GetPosition()
        fp = src_cam.GetFocalPoint()
        up = src_cam.GetViewUp()
        view_angle = src_cam.GetViewAngle()
        self._syncing_cameras = True
        try:
            for i in range(self._pane_container.n_visible):
                if i == src_pane_idx:
                    continue
                pane = self._pane_container.pane(i)
                cam = pane.plotter.renderer.GetActiveCamera()
                cam.SetPosition(*pos)
                cam.SetFocalPoint(*fp)
                cam.SetViewUp(*up)
                cam.SetViewAngle(view_angle)
                pane.plotter.renderer.ResetCameraClippingRange()
                pane.plotter.render()
        finally:
            self._syncing_cameras = False

    def _on_camera_sync(self, on: bool) -> None:
        """Toggle camera sync across visible panes.

        When turning sync ON, snap every visible pane's camera to the
        currently-selected pane's view (or pane 0 if no selection) so the
        user sees an immediate "I see the same thing" effect. When turning
        OFF, leave cameras where they are — each pane keeps the view it
        had at the moment of toggle.
        """
        self._camera_sync_on = on
        if on:
            anchor = self._active_pane_idx
            # Use the propagation path with sync already True; the source-
            # pane camera is the one that drives every other pane.
            self._on_camera_modified(anchor)

    def _on_reset_cameras(self) -> None:
        """Reset the active pane's camera to the default isometric view.

        Behaviour depends on the camera-sync mode:
        - **Sync ON**: resets the active pane, then the observer chain
          mirrors the reset onto every other visible pane (so visually
          "all reset", but driven from one source — matches what the
          rest of the sync code does).
        - **Sync OFF**: resets only the active pane. Other panes keep
          whatever vantage the user carefully positioned them at —
          desync mode exists exactly so different panes can show the
          same data from different angles, and Cmd+R wiping that would
          defeat the point.

        ``reset_camera()`` alone leaves the camera's pan offset in place;
        ``view_isometric()`` aggressively repoints the focal point at the
        origin, then ``reset_camera()`` fits the sphere to the viewport.
        """
        plotter = self._pane_container.pane(self._active_pane_idx).plotter
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.render()

    def _on_pane_clicked(self, idx: int) -> None:
        """User left-clicked pane ``idx``. Promote it to the selected pane.

        Updates the visible selection ring, swaps the File-tab side panel
        to per-pane mode for that pane, and triggers a refresh so the
        per-pane widgets reflect the new pane's :class:`PaneState`.
        """
        if idx == self._selected_pane:
            # Re-selecting the same pane is a no-op (avoids toggling off
            # during normal click-to-rotate; explicit deselect is via Esc).
            return
        self._select_pane(idx)

    def _on_pane_pick(self, pane_idx: int, cell_idx: int | None,
                      *, lon: float | None = None,
                      lat: float | None = None) -> None:
        """Route a pane-picker hit (or miss) into highlight + status updates.

        Called from :class:`~icoscope.picker.Picker` instead of having the
        picker mutate window state directly. ``cell_idx=None`` is the miss /
        empty-click signal: treat as a deselect intent and clear every
        visible pane's highlight plus the status-bar widgets, regardless of
        camera-sync mode.

        On a hit, this method:

        - promotes the clicked pane to the selected pane (so single-pane
          codepaths reading ``self.scalars`` see the right field);
        - paints the cell-outline highlight on **all visible panes** when
          camera sync is on (cell index is geometry-based, so it lines up
          spatially); on only the clicked pane otherwise (different cameras
          would put the cell at different screen positions, so highlighting
          elsewhere would mislead);
        - updates the status-bar lon/lat + the active pane's cell value.
          Per-pane value display is deferred to the future timeline-strip PR.
        """
        if cell_idx is None:
            # Miss / empty-click is a full deselect — same effect as Escape,
            # including dropping the pane selection. Matches the design doc
            # ("empty-click deselects everything").
            self._clear_pick_state(render=True)
            return

        if pane_idx != self._selected_pane:
            self._select_pane(pane_idx)

        visible = self._pane_container.visible_panes()
        if self._camera_sync_on:
            targets = [self._pickers[p.idx] for p in visible]
        else:
            targets = [self._pickers[pane_idx]]
        for p in targets:
            p.highlight_cell(cell_idx)
            p.render()

        if lon is not None and lat is not None:
            self._set_lonlat(lon, lat)
        if hasattr(self, "value_label"):
            self._set_cell_value(cell_idx, lon=lon, lat=lat)
        # Surface the same cell's value on every track of the timeline
        # strip (each pane has its own scalar field at this cell).
        self._refresh_timeline_pane_values()

    def _select_pane(self, idx: int | None) -> None:
        """Programmatic selection helper (used by clicks, Escape, tab switches).

        Updates the selection ring, syncs the File-tab side panel between
        Global / per-pane mode, and pushes the new pane's state into the
        relevant widgets when entering pane mode.
        """
        self._selected_pane = idx
        self._pane_container.set_selected(idx)
        # Side-panel mode swap (File tab only — Ico/LonLat are always
        # single-pane and don't have the mode header).
        if idx is None:
            self.panel.file_tab.set_mode("global")
        else:
            self.panel.file_tab.set_mode("pane", pane_idx=idx)
            self._sync_pane_widgets(idx)

    def _sync_pane_widgets(self, pane_idx: int) -> None:
        """Push pane[idx]'s state into the File-tab per-pane widgets.

        Called on selection change so the user sees the right colormap,
        Color-by choice, slider positions, etc. — without the widgets
        firing their `*_changed` signals back at us (each is blocked).
        """
        if not self._file_cache:
            # No file loaded — nothing meaningful to sync onto the per-pane
            # widgets. set_mode() has already updated the header.
            return
        pane = self.state.panes[pane_idx]
        ft = self.panel.file_tab
        ft.set_color_by(pane.color_by)
        ft.set_cmap(pane.cmap)
        block = ft.display_pane
        block.center_cb.blockSignals(True)
        block.center_cb.setChecked(pane.center_zero)
        block.center_cb.blockSignals(False)
        block.bar_cb.blockSignals(True)
        block.bar_cb.setChecked(pane.colorbar_on)
        block.bar_cb.blockSignals(False)
        # Sync the per-pane colorbar text colour swatch too (override or
        # current theme default).
        ft.set_cbar_color(self._color_to_hex(self._cbar_color(pane_idx)))
        # Re-sync the enable state so a colour-by pane that's switched in
        # gets its swatch + cmap / checkboxes interactive immediately.
        enable = pane.color_by != "None"
        block.center_cb.setEnabled(enable)
        block.bar_cb.setEnabled(enable)
        block.cbar_btn.setEnabled(enable)
        block.cmap_box.setEnabled(enable)
        meta = self._file_state.file_fields.get(pane.color_by)
        if meta and meta.get("time_varying"):
            n_t = meta["shape"][0]
            ft.set_time_axis(n_t, times=self._times_for(meta))
            slider = block.time_slider
            slider.blockSignals(True)
            slider.setValue(min(max(pane.time_index, 0), n_t - 1))
            slider.blockSignals(False)
            ft.set_time_label(slider.value())
        else:
            # Field has no time axis — hide the slider row + play button so
            # the user doesn't see leftover widgets from a previously-selected
            # pane that did have one. Without this the play button stays
            # clickable but _play_step instantly stops because the new field
            # isn't time-varying.
            ft.set_time_axis(0)
        if meta and meta.get("n_levels", 0) > 1 and self._file_state.file_levels is not None:
            n_l = meta["n_levels"]
            ft.set_levels(self._file_state.file_levels)
            slider = block.level_slider
            slider.blockSignals(True)
            slider.setValue(min(max(pane.level_index, 0), n_l - 1))
            slider.blockSignals(False)
            ft.set_level_label(slider.value())
        else:
            # No vertical dim — same logic as the time row above.
            ft.set_levels(None)

    def _on_pane_layout(self, n_panes: int) -> None:
        """User picked View → Pane layout → Single / 1×2 / 2×2.

        Reshuffles the :class:`PaneContainer`, grows :attr:`state.panes`
        to match (newly-visible panes inherit pane 0's settings as
        defaults), computes the new panes' scalar arrays, renders, and
        resets the camera on freshly-revealed panes so the sphere fits
        their viewport instead of starting at PyVista's default tight
        zoom.
        """
        prev_n = self._pane_container.n_visible
        # Layout change retires the prior pick: highlight outlines on now-
        # hidden panes would silently reappear out-of-date when the user
        # expands the layout again. Drop the pick so the user starts the new
        # layout with a clean slate. _build_scene() renders below, so no
        # explicit render needed here.
        if n_panes != prev_n:
            self._clear_pick_state(render=False, deselect_pane=False)
        self._pane_container.set_layout(n_panes)
        _menubar.sync_layout_checkmarks(
            getattr(self, "_layout_actions", {}), n_panes)
        # Remember the user's File-tab layout so it survives tab switches
        # (we collapse to Single when leaving File so the synthetic Ico /
        # LonLat meshes don't render into all 4 viewports).
        if self.panel.tabs.currentIndex() == Tab.FILE:
            self._file_layout = n_panes
        panes = self.state.panes
        # Newly-visible panes inherit pane 0's settings so the user has
        # a coherent starting point (see _design/multi-pane-comparison.md
        # "Default on file open"). Use dataclasses.replace so each new
        # pane is an independent copy.
        from dataclasses import replace
        while len(panes) < n_panes:
            panes.append(replace(panes[0]))
        for idx in range(n_panes):
            self._refresh_scalars(idx)
            self._update_scalars_only(idx)
            self._update_pane_banner(idx)
        # Rebuild the timeline strip — track count follows visible-pane count.
        self._refresh_timeline_strip()
        # Hide banners on panes that just became hidden so their stale
        # state doesn't pop back up next time the user widens the layout.
        for idx in range(n_panes, self._pane_container.MAX_PANES):
            self._pane_container.pane(idx).set_banner(None)
        self._build_scene()
        # Reset the camera on panes that just became visible — without
        # this they use PyVista's pre-add-mesh default which can leave
        # the sphere clipped or off-centre. Suppress the camera-sync
        # observer while doing so: every reset fires ModifiedEvent and
        # would otherwise try to mirror the freshly-defaulted view back
        # onto the existing panes (overwriting the user's view).
        self._syncing_cameras = True
        try:
            for idx in range(prev_n, n_panes):
                self._pane_container.pane(idx).plotter.reset_camera()
        finally:
            self._syncing_cameras = False
        # In sync mode, snap the new panes to the active pane's view so
        # they share the comparison vantage immediately.
        if self._camera_sync_on and n_panes > prev_n:
            self._on_camera_modified(self._active_pane_idx)

    def _on_theme(self, name):
        self.theme_name = name
        # Keep the menu's checkmark in sync (mutually-exclusive).
        _menubar.sync_theme_checkmarks(
            getattr(self, "_theme_actions", {}), name)
        suggested = THEMES[name]["cmap"]
        # Push the suggested cmap onto every tab whose cmap matched a known
        # theme default (otherwise the user has explicitly chosen, leave it).
        for tab_state, tab_widget in (
            (self._ico_state, self.panel.ico_tab),
            (self._file_state, self.panel.file_tab),
            (self._lonlat_state, self.panel.lonlat_tab),
        ):
            tab_state.cmap = suggested
            if tab_widget is not None:
                tab_widget.set_cmap(suggested)
        self._sync_color_buttons()
        self._build_scene()
        self._update_status()

    def _on_cmap(self, name):
        self.pane_state.cmap = name
        self._build_scene()

    def _on_coast(self, on):
        self.state.coastlines_on = on
        self._build_scene()

    def _on_grat(self, on):
        self.state.graticule_on = on
        self._build_scene()

    def _on_edges(self, on):
        self.state.edges_on = on
        self._build_scene()

    def _refresh_scalars(self, pane_idx: int | None = None) -> None:
        """Compute a pane's scalar array from its :class:`PaneState`.

        Defaults to the active pane (single-pane code paths); pass an
        explicit ``pane_idx`` to recompute a different pane (used when
        a layout change makes a previously-hidden pane visible).
        """
        idx = self._active_pane_idx if pane_idx is None else pane_idx
        pane = self.state.panes[idx]
        color_by = pane.color_by
        file_fields = self._file_state.file_fields
        if color_by == "Latitude":
            arr = np.degrees(np.arcsin(self.centers[:, 2]))
        elif color_by == "Cell kind":
            arr = np.array([0 if len(c) == 5 else 1 for c in self.cells],
                           dtype=float)
        elif color_by == "Mock temperature":
            # Clean synthetic field: latitude gradient + a faint zonal wave.
            # Same formula as the `tas` field in tools/make_test_nc.py.
            c = self.centers / np.linalg.norm(self.centers, axis=1, keepdims=True)
            lat = np.arcsin(np.clip(c[:, 2], -1, 1))
            lon = np.arctan2(c[:, 1], c[:, 0])
            arr = 250.0 + 50.0 * np.cos(lat) + 5.0 * np.cos(2 * lon)
        elif color_by == "Realistic temperature":
            # Earth-like surface-temperature mock: base latitudinal gradient,
            # plus broad Gaussian "hot spots" over major land masses (Sahara,
            # Arabia, Australia, North-American interior) and cold spots
            # (Greenland, Antarctic interior). Visually convincing without
            # any real climate data.
            c = self.centers / np.linalg.norm(self.centers, axis=1, keepdims=True)
            lat = np.arcsin(np.clip(c[:, 2], -1, 1))
            lon = np.arctan2(c[:, 1], c[:, 0])

            def gauss(lat_c, lon_c, amp, sigma_deg):
                dlat = lat - np.radians(lat_c)
                # angular wrap-around in longitude
                dlon = (lon - np.radians(lon_c) + np.pi) % (2 * np.pi) - np.pi
                # great-circle-ish distance (cheap approximation)
                d2 = dlat ** 2 + (dlon * np.cos(lat)) ** 2
                return amp * np.exp(-d2 / (2 * np.radians(sigma_deg) ** 2))

            # base latitudinal profile, ~245 K at poles to ~300 K near equator
            T = 275.0 + 30.0 * np.cos(lat)

            # land hot spots (degrees: lat, lon, amplitude K, sigma deg)
            T += gauss( 23,  10, +14, 14)   # Sahara
            T += gauss( 25,  48, +10, 12)   # Arabia
            T += gauss(-25, 135, + 9, 14)   # Australian outback
            T += gauss( 35, -105, +6, 15)   # SW USA / Mexico
            T += gauss(-20, -60, + 6, 15)   # interior South America

            # cold spots
            T += gauss( 75, -40, -22, 14)   # Greenland
            T += gauss(-82,   0, -32, 22)   # Antarctic interior
            T += gauss( 65,  100, -8, 18)   # Siberian winter (mild proxy)

            arr = T
        elif color_by in file_fields and self.file_path:
            from .loader import read_field
            ctx = self._file_cache.get("context") if self._file_cache else None
            arr = read_field(
                self.file_path, color_by,
                time_index=pane.time_index,
                level_index=pane.level_index,
                context=ctx,
            )
        else:
            arr = None
        self._pane_scalars[idx] = arr

    def _on_color_by(self, name):
        # The change targets the selected pane (or pane 0 on Ico/LonLat
        # where there's only ever one). Writing to self.pane_state (not
        # self.state) routes through the active pane — without this, multi-
        # pane changes would silently update pane 0 regardless of selection.
        self.pane_state.color_by = name
        # Enable/disable cmap-related widgets on the tab that emitted the change.
        tab = self.active_tab
        if hasattr(tab, "display"):
            tab.display.center_cb.setEnabled(name != "None")
            tab.display.bar_cb.setEnabled(name != "None")
            tab.display.cbar_btn.setEnabled(name != "None")
            tab.display.cmap_box.setEnabled(name != "None")
        # configure the time + level sliders for the active field (File tab only).
        meta = self._file_state.file_fields.get(name) if tab is self.panel.file_tab else None
        if tab is self.panel.file_tab:
            if meta and meta.get("time_varying"):
                self.panel.file_tab.set_time_axis(
                    meta["shape"][0],
                    times=self._times_for(meta),
                )
            else:
                self.panel.file_tab.set_time_axis(0)
            # Resolve the master cursor against the new field's axis so a
            # pane swap to a different time dim (e.g. monthly → daily) lands
            # on the closest equivalent sample instead of resetting to 0.
            self.pane_state.time_index = self._resolve_pane_to_cursor(
                self._active_pane_idx)
            self._update_pane_banner(self._active_pane_idx)
            n_levels = meta.get("n_levels", 0) if meta else 0
            if n_levels > 1 and self._file_state.file_levels is not None:
                self.panel.file_tab.set_levels(self._file_state.file_levels)
            else:
                self.panel.file_tab.set_levels(None)
            self.pane_state.level_index = 0
        self._refresh_scalars()
        # Color-by changes only the scalar field — geometry (and the picker
        # locator built from it) stay valid, so just swap scalars in place.
        self._update_scalars_only()
        # The displayed picker value is for the *previous* field's units; clear
        # it (the lon/lat + highlight stay — the geometric pick is still
        # meaningful, the user just needs to re-pick to update the value).
        if hasattr(self, "value_label"):
            self._clear_cell_value()
        # The active pane's field changed → its track on the strip needs to
        # show the new sample dots (and the strip may need to appear/hide
        # if the new field is/isn't time-varying).
        self._refresh_timeline_strip()
        self._build_scene()

    def _on_colorbar(self, on):
        self.pane_state.colorbar_on = on
        self._build_scene()

    def _on_center_zero(self, on):
        self.pane_state.center_zero = on
        self._build_scene()

    def _on_edge_color(self, hex_str):
        self.state.edge_color_override = hex_str
        self._build_scene()

    def _on_coast_color(self, hex_str):
        self.state.coast_color_override = hex_str
        self._build_scene()

    def _on_grat_color(self, hex_str):
        self.state.grat_color_override = hex_str
        self._build_scene()

    def _on_cbar_color(self, hex_str):
        self.pane_state.cbar_color_override = hex_str
        self._build_scene()

    def _on_edge_width(self, w):
        self.state.edge_width = float(w)
        self._build_scene()

    def _on_coast_width(self, w):
        self.state.coast_width = float(w)
        self._build_scene()

    def _on_grat_width(self, w):
        self.state.grat_width = float(w)
        self._build_scene()

    def _apply_mesh_change(self) -> None:
        """Refresh derived state after ``self.verts/cells/centers`` change."""
        self._refresh_scalars()
        self._mesh = self._to_polydata()
        self._invalidate_all_locators()
        # Mesh swap invalidates the picked cell: the index points at a
        # different polygon now. Drop the entire pick state (lon/lat + value
        # would otherwise stay stale). Don't reset the selected pane — the
        # user's pane selection is independent of which cell they picked.
        self._clear_pick_state(render=False, deselect_pane=False)
        self._build_scene()
        self._update_status()

    def _invalidate_all_locators(self) -> None:
        """Drop every pane-picker's cached vtkCellLocator after a mesh swap."""
        for p in self._pickers:
            p.invalidate_locator()

    def _clear_pick_state(self, *, render: bool = False,
                          deselect_pane: bool = True) -> None:
        """Drop every visible pick artifact in one place.

        Single source of truth for "deselect": clears the highlight outline
        on every pane, the status-bar lon/lat, the status-bar cell-value
        label, and (by default) the active pane selection. Used by Escape,
        the picker miss branch, mesh swaps, file unload, and layout
        changes — so the four paths can't drift apart on what "clear"
        means.

        Parameters
        ----------
        render
            Force a render on each pane after removing the highlight actor.
            Pass ``False`` when the caller will render shortly anyway
            (e.g. ``_build_scene``).
        deselect_pane
            Also drop the active pane selection. ``False`` is for system-
            triggered clears (mesh swap, file unload) where the user's
            pane selection is independent of the pick.
        """
        for p in self._pickers:
            p.clear_highlight()
            if render:
                p.render()
        self._clear_lonlat()
        self._clear_cell_value()
        # Clear per-track value columns on the timeline strip too — there's
        # no pick anymore, no value to surface per pane.
        if hasattr(self, "_timeline_strip"):
            self._refresh_timeline_pane_values()
        if deselect_pane:
            self._select_pane(None)

    def _regen_mesh(self, cache: dict | None, key: tuple, build) -> dict:
        """Reuse ``cache`` if its ``params`` match ``key``, else call ``build()``.

        ``build()`` must return ``(verts, cells, centers)``. The returned
        dict is the (possibly-new) cache the caller should store back.
        Always applies :meth:`_apply_mesh_change` at the end.
        """
        if cache is not None and cache["params"] == key:
            self.verts = cache["verts"]
            self.cells = cache["cells"]
            self.centers = cache["centers"]
        else:
            v, c, ctr = build()
            self.verts, self.cells, self.centers = v, c, np.asarray(ctr)
            cache = {
                "params": key,
                "verts": self.verts,
                "cells": self.cells,
                "centers": self.centers,
            }
        self._apply_mesh_change()
        return cache

    def _ico_params_key(self) -> tuple:
        """Cache key identifying the current Ico-tab mesh parameters."""
        return (self.n, self.max_relax_iters,
                self.zoom_factor, self.zoom_lon, self.zoom_lat)

    def _regen_synthetic(self) -> None:
        def build():
            v, c, ctr, _ = goldberg(
                n=self.n, relax=self.max_relax_iters > 0,
                max_iterations=self.max_relax_iters,
                zoom_factor=self.zoom_factor,
                zoom_lon=self.zoom_lon, zoom_lat=self.zoom_lat,
            )
            return v, c, ctr
        self._ico_cache = self._regen_mesh(
            self._ico_cache, self._ico_params_key(), build)

    def _on_n(self, n):
        self.n = n
        if self._is_ico_tab_active():
            self._regen_synthetic()

    def _on_relax_iters(self, n):
        self.max_relax_iters = n
        if self._is_ico_tab_active():
            self._regen_synthetic()

    def _on_zoom(self, factor, lon, lat):
        self.zoom_factor = float(factor)
        self.zoom_lon = float(lon)
        self.zoom_lat = float(lat)
        if self._is_ico_tab_active():
            self._regen_synthetic()

    def _is_ico_tab_active(self) -> bool:
        return self.panel.tabs.currentIndex() == Tab.ICO

    def _is_lonlat_tab_active(self) -> bool:
        return self.panel.tabs.currentIndex() == Tab.LONLAT

    def _lonlat_params_key(self) -> tuple:
        """Cache key identifying the current LonLat-tab mesh parameters."""
        return (self.iim, self.jjm,
                self.lmdz_clon, self.lmdz_clat,
                self.lmdz_grossismx, self.lmdz_grossismy,
                self.lmdz_dzoomx, self.lmdz_dzoomy,
                self.lmdz_taux, self.lmdz_tauy)

    def _regen_lonlat(self) -> None:
        """Build (or reuse) the synthetic LonLat mesh and render it.

        ``latlon_mesh`` raises ``ValueError`` for invalid LMDZ-zoom combinations
        (the 2β-G>0 check). Let it propagate — ``_on_lmdz_zoom`` catches it
        and triggers the snap-back + red error message.
        """
        def build():
            return latlon_mesh(
                iim=self.iim, jjm=self.jjm,
                clon=self.lmdz_clon, clat=self.lmdz_clat,
                grossismx=self.lmdz_grossismx,
                grossismy=self.lmdz_grossismy,
                dzoomx=self.lmdz_dzoomx, dzoomy=self.lmdz_dzoomy,
                taux=self.lmdz_taux, tauy=self.lmdz_tauy,
            )
        self._lonlat_cache = self._regen_mesh(
            self._lonlat_cache, self._lonlat_params_key(), build)

    def _on_iim(self, val):
        self.iim = int(val)
        if self._is_lonlat_tab_active():
            self._regen_lonlat()

    def _on_jjm(self, val):
        self.jjm = int(val)
        if self._is_lonlat_tab_active():
            self._regen_lonlat()

    def _on_lmdz_zoom(self, clon, clat, gx, gy, dx, dy, tx, ty):
        # Snapshot the last known good params so we can roll back if the new
        # ones fail the 2·β - G > 0 validity check inside latlon_mesh.
        snapshot = (self.lmdz_clon, self.lmdz_clat,
                    self.lmdz_grossismx, self.lmdz_grossismy,
                    self.lmdz_dzoomx, self.lmdz_dzoomy,
                    self.lmdz_taux, self.lmdz_tauy)
        self.lmdz_clon = float(clon)
        self.lmdz_clat = float(clat)
        self.lmdz_grossismx = float(gx)
        self.lmdz_grossismy = float(gy)
        self.lmdz_dzoomx = float(dx)
        self.lmdz_dzoomy = float(dy)
        self.lmdz_taux = float(tx)
        self.lmdz_tauy = float(ty)

        # Default: a valid combination keeps the toggle enabled. Either
        # branch below may disable it.
        self.panel.lonlat_tab.set_lmdz_zoom_toggle_enabled(True)
        active = self.panel.lonlat_tab.lmdz_zoom_active
        if active and self._is_lonlat_tab_active():
            # Full regen with revert-on-error. Revert restores a known-good
            # combination, so the toggle stays enabled either way.
            try:
                self._regen_lonlat()
            except ValueError as e:
                snap, err = snapshot, str(e)
                # Defer the revert past the spinbox's own valueChanged handler
                # so setValue actually repaints the line editor on macOS.
                QTimer.singleShot(0, lambda: self._revert_lmdz_zoom(snap, err))
        else:
            # Zoom off — still validate so the user knows the combination is
            # bad. We don't snap-back (lets them keep editing settings) but
            # we grey out the Activate toggle until the combination becomes
            # valid again, so they can't activate a broken zoom.
            try:
                from .lonlat import latlon_mesh
                latlon_mesh(
                    iim=4, jjm=4,                # cheap validation grid
                    clon=self.lmdz_clon, clat=self.lmdz_clat,
                    grossismx=self.lmdz_grossismx,
                    grossismy=self.lmdz_grossismy,
                    dzoomx=self.lmdz_dzoomx, dzoomy=self.lmdz_dzoomy,
                    taux=self.lmdz_taux, tauy=self.lmdz_tauy,
                )
            except ValueError as e:
                self.panel.lonlat_tab.set_lmdz_zoom_toggle_enabled(False)
                err = str(e)
                QTimer.singleShot(0, lambda: self._flash_error(err))

    def _revert_lmdz_zoom(self, snapshot, err_text: str):
        """Restore the 8 LMDZ-zoom fields + spinboxes from ``snapshot``."""
        (self.lmdz_clon, self.lmdz_clat,
         self.lmdz_grossismx, self.lmdz_grossismy,
         self.lmdz_dzoomx, self.lmdz_dzoomy,
         self.lmdz_taux, self.lmdz_tauy) = snapshot
        self.panel.lonlat_tab.set_lmdz_zoom(*snapshot)
        self._flash_error(err_text)

    def _flash_error(self, msg: str, duration_ms: int = 5000):
        """Show ``msg`` in red in the status bar for ``duration_ms`` ms.

        Uses a dedicated permanent QLabel widget with rich-text HTML so the
        red colour bypasses macOS's aggressive style overrides on the
        QStatusBar's built-in message label.
        """
        if not hasattr(self, "_error_label"):
            self._error_label = QLabel("")
            self._error_label.setTextFormat(Qt.RichText)
            # addWidget anchors to the left (where status messages normally sit);
            # addPermanentWidget would put it on the right.
            self.statusBar().addWidget(self._error_label, 1)
        self._error_label.setText(
            f'<span style="color:#d33; font-weight:bold;">{msg}</span>'
        )
        QTimer.singleShot(duration_ms, lambda: self._error_label.setText(""))

    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open NetCDF", "", "NetCDF (*.nc *.nc4 *.cdf);;All files (*)"
        )
        if not path:
            return
        try:
            from .loader import FileContext, load_grid, read_levels
            verts, cells, centers, fields = load_grid(path)
            levels = read_levels(path)
            context = FileContext(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        # Close any previous file's Dataset handle so we don't leak.
        if self._file_cache is not None:
            old_ctx = self._file_cache.get("context")
            if old_ctx is not None:
                old_ctx.close()
        self.file_path = path
        self._file_state.file_fields = fields
        self._file_state.file_levels = levels
        # Fresh file: reset selections so _activate_file_view's "preserve
        # prior choice" path doesn't carry over a field/index from the
        # previous file that may not exist or may be out-of-range here.
        self._file_state.color_by = "None"
        self._file_state.time_index = 0
        self._file_state.level_index = 0
        self._file_cache = {
            "path": path,
            "verts": verts,
            "cells": cells,
            "centers": np.asarray(centers),
            "fields": fields,
            "levels": levels,
            "context": context,
        }
        self.panel.file_tab.set_file_loaded(True)
        self._sync_file_info(path)
        self._activate_file_view()
        # Auto-switch to the File tab so the user sees the loaded data.
        self.panel.tabs.setCurrentIndex(Tab.FILE)

    def _sync_file_info(self, path: str):
        """Populate the File tab summary from the currently-loaded file."""
        from .loader import read_global_attrs
        fields = self._file_state.file_fields
        n_time = max(
            (meta["shape"][0] for meta in fields.values()
             if meta.get("time_varying")),
            default=0,
        )
        try:
            attrs = read_global_attrs(path)
        except Exception:
            attrs = {}
        self.panel.file_tab.set_file_info(
            path=path,
            n_cells=len(self.cells),
            n_fields=len(fields),
            n_time_steps=n_time,
            attrs=attrs,
        )

    def _on_close_file(self):
        """Drop the loaded file. Stays on the File tab — shows the empty sphere."""
        if not self.file_path:
            return
        # Close the held-open Dataset before dropping the cache.
        if self._file_cache is not None:
            ctx = self._file_cache.get("context")
            if ctx is not None:
                ctx.close()
        self.file_path = None
        self._file_state.file_fields = {}
        self._file_state.file_levels = None
        self._file_state.time_cursor = None
        self._file_cache = None
        # Hide any stale banners — the file they referenced is gone.
        for i in range(self._pane_container.MAX_PANES):
            self._pane_container.pane(i).set_banner(None)
        self._refresh_timeline_strip()
        # File tab's Color by only ever lists file fields, never synthetic
        # options — on unload, just "None" remains.
        self.panel.file_tab.set_color_by_items(["None"])
        self._file_state.color_by = "None"
        self.panel.file_tab.set_color_by("None")
        # color_by = "None" → grey out cmap/colorbar/center-zero, matching
        # the initial state. set_color_by blocks signals so _on_color_by
        # wouldn't fire otherwise.
        self.panel.file_tab.display.center_cb.setEnabled(False)
        self.panel.file_tab.display.bar_cb.setEnabled(False)
        self.panel.file_tab.display.cbar_btn.setEnabled(False)
        self.panel.file_tab.display.cmap_box.setEnabled(False)
        self.panel.file_tab.set_time_steps(0)
        self.panel.file_tab.set_levels(None)
        self.panel.file_tab.set_file_loaded(False)
        self.panel.file_tab.set_file_info()
        # File tab stays active; render the empty sphere. The spin timer
        # is unaffected — _spin_tick rotates the camera, not the mesh.
        self._render_empty_sphere()

    def _activate_file_view(self):
        """Render the cached file mesh (must be called only when _file_cache is set)."""
        c = self._file_cache
        assert c is not None
        self.verts, self.cells, self.centers = c["verts"], c["cells"], c["centers"]
        self._file_state.file_levels = c.get("levels")
        items = ["None"] + list(c["fields"].keys())
        self.panel.file_tab.set_color_by_items(items)
        # Preserve the previously selected field on re-entry (tab switch back).
        # Only fall back to the first field if there is no prior selection or
        # it isn't in the current file's fields (e.g. after Open NetCDF on a
        # different file).
        prior = self._file_state.color_by
        if prior in c["fields"]:
            desired = prior
        elif c["fields"]:
            desired = next(iter(c["fields"].keys()))
        else:
            desired = "None"
        self.panel.file_tab.set_color_by(desired)
        self._file_state.color_by = desired
        # Re-configure the time + level sliders for the chosen field and
        # restore their saved positions (set_time_steps / set_levels reset
        # the slider value to 0 — block signals so the restore doesn't fire
        # _on_time_changed / _on_level_changed; the subsequent
        # _apply_mesh_change call below rebuilds the scene once).
        meta = c["fields"].get(desired)
        # Guard against a degenerate empty-time dim (unlimited dim with zero
        # records written): meta says time_varying=True but shape[0]==0 would
        # produce slider.setValue(-1) and a nonsense label.
        n_t = meta["shape"][0] if (meta and meta.get("time_varying")) else 0
        if n_t > 1:
            self.panel.file_tab.set_time_axis(n_t, times=self._times_for(meta))
            # Clamp the saved index in case the caller is recycling state
            # across a different field/file (e.g. via _on_open_file).
            t_idx = min(max(self._file_state.time_index, 0), n_t - 1)
            self._file_state.time_index = t_idx
            slider = self.panel.file_tab.display.time_slider
            slider.blockSignals(True)
            slider.setValue(t_idx)
            slider.blockSignals(False)
            self.panel.file_tab.set_time_label(t_idx)
        else:
            self.panel.file_tab.set_time_steps(0)
            self._file_state.time_index = 0
        if meta and meta.get("n_levels", 0) > 1 and self._file_state.file_levels is not None:
            n_l = meta["n_levels"]
            self.panel.file_tab.set_levels(self._file_state.file_levels)
            l_idx = min(max(self._file_state.level_index, 0), n_l - 1)
            self._file_state.level_index = l_idx
            slider = self.panel.file_tab.display.level_slider
            slider.blockSignals(True)
            slider.setValue(l_idx)
            slider.blockSignals(False)
            self.panel.file_tab.set_level_label(l_idx)
        else:
            self.panel.file_tab.set_levels(None)
            self._file_state.level_index = 0
        # _on_color_by normally toggles these on/off, but set_color_by above
        # blocks signals to avoid recursion. Sync them by hand so the
        # cmap/colorbar/center-zero widgets are usable as soon as a file
        # loads (via --file at startup, via Open NetCDF, or via tab-switch
        # back to File after a previous load).
        enable = self._file_state.color_by != "None"
        self.panel.file_tab.display.center_cb.setEnabled(enable)
        self.panel.file_tab.display.bar_cb.setEnabled(enable)
        self.panel.file_tab.display.cbar_btn.setEnabled(enable)
        self.panel.file_tab.display.cmap_box.setEnabled(enable)
        self._apply_mesh_change()
        self._refresh_timeline_strip()

    def _on_tab_changed(self, idx: int):
        """Tab is the active mesh source — swap the rendered scene accordingly."""
        # Tab switching has two ordering constraints that conflict:
        # (a) leaving a multi-pane File layout for Ico/LonLat: collapse to
        #     Single BEFORE the regen, otherwise _build_scene iterates over
        #     panes that don't exist in the new tab's state.panes.
        # (b) entering File from a single-pane Ico/LonLat: activate the
        #     File mesh BEFORE restoring the multi-pane layout, otherwise
        #     _on_pane_layout's _update_scalars_only writes File-sized
        #     scalars against the previous tab's smaller mesh and crashes.
        # Handle them as two separate branches.
        is_file_tab = idx == Tab.FILE
        for action in getattr(self, "_layout_actions", {}).values():
            action.setEnabled(is_file_tab)

        if not is_file_tab:
            # Leaving File: collapse layout + clear selection BEFORE regen.
            if self._pane_container.n_visible != 1:
                self._pane_container.set_layout(1)
                _menubar.sync_layout_checkmarks(
                    getattr(self, "_layout_actions", {}), 1)
            self._select_pane(None)
            if idx == Tab.ICO:
                self._regen_synthetic()
            elif idx == Tab.LONLAT:
                self._regen_lonlat()
        else:
            # Entering File: activate the mesh FIRST so subsequent layout
            # restore writes scalars against the correct geometry.
            if self._file_cache is not None:
                self._activate_file_view()
            else:
                # No file loaded — show a plain empty sphere instead of
                # whatever was last rendered.
                self._render_empty_sphere()
            if self._pane_container.n_visible != self._file_layout:
                self._on_pane_layout(self._file_layout)
            if self._selected_pane is None:
                self._select_pane(0)
        # Per-tab colour overrides may differ → refresh swatches.
        self._sync_color_buttons()
        # Auto-rotate is per-tab state but the timer is window-level — sync
        # the timer to the new active tab's spin_on flag.
        if self.state.spin_on:
            self.playback.start_spin()
        else:
            self.playback.stop_spin()
        # Strip is only meaningful on the File tab — hide on Ico/LonLat,
        # rebuild when re-entering File.
        self._refresh_timeline_strip()
        self._update_status()

    def _render_empty_sphere(self):
        """Render a plain blank sphere on every visible pane.

        Used when a tab is active but has no mesh to show yet — the
        File tab before any NetCDF is loaded, and the LonLat placeholder
        tab. Conveys "this is the canvas, populate it" without leaking
        stale geometry from another tab into the view.
        """
        bg = THEMES[self.theme_name]["bg"]
        # pv.Sphere uses theta/phi tessellation, which leaves visible latitude
        # rings even with smooth_shading on. Icosphere has no pole singularity
        # and no axis-aligned strips, so the surface reads as a clean sphere.
        sphere = pv.Icosphere(radius=1.0, nsub=5)
        for idx in range(self._pane_container.n_visible):
            plotter = self._pane_container.pane(idx).plotter
            plotter.clear()
            plotter.set_background(bg)
            plotter.add_mesh(sphere, name="empty",
                             color="#777777", show_edges=False,
                             smooth_shading=True, reset_camera=False)
            plotter.render()
        self._mesh = None
        # Clear every pane's scalar array so a subsequent re-render isn't
        # confused by stale data left over from a prior file.
        self._pane_scalars = [None] * PaneContainer.MAX_PANES
        self._invalidate_all_locators()
        # File unload: any prior pick is meaningless on the empty sphere.
        # The render-window loop above already issued a render per plotter,
        # so we don't need another one here.
        self._clear_pick_state(render=False, deselect_pane=False)
        self._update_status()

    def _on_time_changed(self, idx):
        # Targets the selected pane (or pane 0 when no selection). The
        # scrub also sets the file-tab's master ``time_cursor`` so every
        # other visible pane resolves the cursor to its own axis — files
        # with multiple time dims (daily + monthly) stay aligned without
        # the user having to scrub each pane's slider.
        if idx == self.pane_state.time_index:
            return
        self.pane_state.time_index = idx
        meta = self._file_state.file_fields.get(self.pane_state.color_by)
        if meta:
            self.panel.file_tab.set_time_label(idx)
        self._sync_cursor_from_pane(self._active_pane_idx)
        self._refresh_picked_value()
        self._build_scene()

    def _update_pane_banner(self, pane_idx: int) -> None:
        """Show / hide the out-of-range banner on pane ``pane_idx``.

        The banner appears when the master cursor falls outside the
        pane's field's time-axis range; the pane is then clamped to its
        first/last sample, and the banner makes that explicit instead of
        silently showing stale-looking data. Cleared when in-range or
        when no cursor / no time axis applies.
        """
        from .time_axis import is_in_range
        pane_widget = self._pane_container.pane(pane_idx)
        pane = self.state.panes[pane_idx]
        cursor = self._file_state.time_cursor
        meta = self._file_state.file_fields.get(pane.color_by)
        times = self._times_for(meta) if meta else None
        if cursor is None or times is None or len(times) == 0:
            pane_widget.set_banner(None)
            return
        if is_in_range(cursor, times):
            pane_widget.set_banner(None)
            return
        from .formatters import short_datetime
        nearest = times[pane.time_index]
        pane_widget.set_banner(
            f"Showing {short_datetime(nearest)} "
            f"(cursor at {short_datetime(cursor)})"
        )

    def _resolve_pane_to_cursor(self, pane_idx: int) -> int:
        """Return the time index that surfaces ``pane_idx``'s data for the cursor.

        Uses :func:`last_previous_time_index` — the physically-correct
        resolution for climate data (a sample stamped ``t`` represents
        the period preceding ``t``, so a cursor on April 15 should show
        March's monthly mean, not April's). See ``time_axis.py`` for the
        full reasoning; future settings work may expose a "nearest"
        alternative for visualisation use cases.

        Falls back to 0 when there's no cursor yet, the pane's field has
        no time axis, or the axis can't be parsed. Caller is responsible
        for writing the result back to ``pane.time_index``.
        """
        from .time_axis import last_previous_time_index
        cursor = self._file_state.time_cursor
        if cursor is None:
            return 0
        pane = self.state.panes[pane_idx]
        meta = self._file_state.file_fields.get(pane.color_by)
        times = self._times_for(meta) if meta else None
        if times is None or len(times) == 0:
            return 0
        return last_previous_time_index(cursor, times)

    def _sync_cursor_from_pane(self, anchor_idx: int) -> None:
        """Take the anchor pane's time_index, store as master cursor, propagate.

        The anchor pane is whichever pane just had its slider scrubbed —
        its ``(color_by, time_index)`` defines the absolute datetime. Every
        other visible pane resolves the cursor against its own field's
        axis via :func:`nearest_time_index` and refreshes if its index
        changed. Anchor pane always refreshes (its index already changed).
        """
        anchor_pane = self.state.panes[anchor_idx]
        anchor_meta = self._file_state.file_fields.get(anchor_pane.color_by)
        anchor_times = self._times_for(anchor_meta) if anchor_meta else None
        cursor = (anchor_times[anchor_pane.time_index]
                  if anchor_times is not None
                  and anchor_pane.time_index < len(anchor_times)
                  else None)
        self._set_master_cursor(cursor)

    def _set_master_cursor(self, cursor) -> None:
        """Propagate a cursor datetime to every visible pane.

        Shared entry point used both by side-panel slider scrubs (via
        :meth:`_sync_cursor_from_pane`) and by direct drags on the bottom
        timeline strip. Each visible pane's ``time_index`` shifts to the
        latest sample at or before the cursor on its own axis (the
        physically-correct resolution for climate data — see
        :func:`last_previous_time_index`), **unless that pane has
        ``time_locked=True``** (the user pinned it to a fixed time via
        the timeline-strip lock). The active pane's side-panel slider
        follows along (signals blocked so we don't re-enter), and the
        timeline strip's cursor marker + per-pane value column are
        updated.
        """
        from .time_axis import last_previous_time_index
        self._file_state.time_cursor = cursor
        for i in range(self._pane_container.n_visible):
            pane = self.state.panes[i]
            if cursor is not None and not pane.time_locked:
                meta = self._file_state.file_fields.get(pane.color_by)
                times = self._times_for(meta) if meta else None
                if times is not None and len(times) > 0:
                    new_idx = last_previous_time_index(cursor, times)
                    if new_idx != pane.time_index:
                        pane.time_index = new_idx
            self._refresh_scalars(i)
            self._update_scalars_only(i)
            self._update_pane_banner(i)
        # Sync the side-panel slider for the active pane (without firing
        # its valueChanged → _on_time_changed → re-entry into this method).
        active = self._active_pane_idx
        if active < self._pane_container.n_visible:
            slider = self.panel.file_tab.display_pane.time_slider
            slider.blockSignals(True)
            slider.setValue(self.state.panes[active].time_index)
            slider.blockSignals(False)
            self.panel.file_tab.set_time_label(slider.value())
        self._refresh_timeline_cursors()
        # Each pane's scalar field now corresponds to a different time
        # slice — re-render the per-track value column from _last_pick.
        self._refresh_timeline_pane_values()

    def _on_timeline_lock_toggled(self, pane_idx: int) -> None:
        """User clicked the lock icon on track ``pane_idx`` — invert state."""
        if pane_idx >= len(self.state.panes):
            return
        pane = self.state.panes[pane_idx]
        was_locked = pane.time_locked
        pane.time_locked = not was_locked
        self._timeline_strip.set_pane_locked(pane_idx, pane.time_locked)
        if was_locked:
            # Just unlocked — the pane is now following the master cursor
            # again. Re-propagate so its time_index + scalars + render
            # catch up to wherever the cursor moved while it was locked.
            self._set_master_cursor(self._file_state.time_cursor)
            self._build_scene()
        else:
            # Just locked — pane stops following; only the per-track
            # cursor visual needs updating (data stays where it was).
            self._refresh_timeline_cursors()

    def _refresh_timeline_pane_values(self) -> None:
        """Populate each track's value column from ``_last_pick``.

        Walks every visible pane and formats its current scalar at the
        picked cell (cells are shared geometry, so the index is universal).
        Tracks whose pane has no scalars yet (e.g. color_by is "None")
        get an empty string and hide their value column.
        """
        if not hasattr(self, "_timeline_strip"):
            return
        if self._last_pick is None:
            self._timeline_strip.set_pane_values(
                [""] * self._pane_container.n_visible)
            return
        cell_idx, _lon, _lat = self._last_pick
        values: list[str] = []
        from .formatters import format_cell_value
        for i in range(self._pane_container.n_visible):
            arr = self._pane_scalars[i]
            pane = self.state.panes[i]
            if (arr is None or cell_idx >= len(arr)
                    or pane.color_by == "None"):
                values.append("")
                continue
            v = float(arr[cell_idx])
            meta = self._file_state.file_fields.get(pane.color_by)
            units = meta.get("units", "") if meta else ""
            short, _ = format_cell_value(v, units)
            # format_cell_value returns "Value: X K"; strip the "Value: "
            # prefix — the track label already names the field.
            if short.startswith("Value: "):
                short = short[len("Value: "):]
            values.append(short)
        self._timeline_strip.set_pane_values(values)

    def _on_timeline_cursor_changed(self, cursor) -> None:
        """User dragged the timeline strip — set the master cursor + render."""
        self._set_master_cursor(cursor)
        self._build_scene()

    def _refresh_timeline_strip(self) -> None:
        """Rebuild the bottom timeline from the current visible-pane state.

        Show the strip only on the File tab, only when at least one visible
        pane displays a time-varying field. One track per visible pane,
        labelled with the pane's color_by; samples are the parsed datetimes
        from that field's time axis.
        """
        on_file_tab = self.panel.tabs.currentIndex() == Tab.FILE
        if not on_file_tab or self._mesh is None:
            self._timeline_strip.set_panes([])
            return
        tracks = []
        any_time_varying = False
        for i in range(self._pane_container.n_visible):
            pane = self.state.panes[i]
            meta = self._file_state.file_fields.get(pane.color_by)
            times = self._times_for(meta) if meta else None
            # Prefix the track label with the 1-indexed pane number so the
            # strip is self-describing. Just the number (not "Pane N") to
            # keep the prefix short when the field name is long — track
            # ordering matches viewport ordering, so the surrounding
            # context makes "1:" unambiguous.
            label = (f"{i + 1}: {pane.color_by}"
                     if pane.color_by != "None" else f"Pane {i + 1}")
            if times is not None and len(times) > 0:
                any_time_varying = True
                tracks.append((label, list(times)))
            else:
                tracks.append((label, []))
        if not any_time_varying:
            self._timeline_strip.set_panes([])
            return
        self._timeline_strip.set_panes(tracks)
        # Rebuilding tracks resets their lock visuals — push the persisted
        # per-pane state back so a layout change doesn't silently unlock.
        for i in range(self._pane_container.n_visible):
            self._timeline_strip.set_pane_locked(
                i, self.state.panes[i].time_locked)
        self._refresh_timeline_cursors()
        self._refresh_timeline_pane_values()

    def _refresh_timeline_cursors(self) -> None:
        """Push per-track cursor positions to the strip.

        Locked panes keep their cursor pinned at the datetime
        corresponding to their frozen ``time_index``; unlocked panes
        follow the master cursor. Computing per-track lets a locked
        track's bar stay where the data actually is, instead of
        drifting with the master cursor that the pane is ignoring.
        """
        master = self._file_state.time_cursor
        cursors: list = []
        for i in range(self._pane_container.n_visible):
            pane = self.state.panes[i]
            if pane.time_locked:
                meta = self._file_state.file_fields.get(pane.color_by)
                times = self._times_for(meta) if meta else None
                if times is not None and pane.time_index < len(times):
                    cursors.append(times[pane.time_index])
                else:
                    cursors.append(None)
            else:
                cursors.append(master)
        self._timeline_strip.set_cursors(cursors)

    def _on_level_changed(self, idx):
        if idx == self.pane_state.level_index:
            return
        self.pane_state.level_index = idx
        self._refresh_scalars()
        self._refresh_picked_value()
        self._update_scalars_only()
        self._build_scene()

    def _on_play_toggled(self, on):
        self.playback.toggle_play(on)

    def _on_play_speed_changed(self, ms):
        self.playback.set_speed(ms)

    def _on_loop_toggled(self, on: bool) -> None:
        """User toggled the playback Loop checkbox."""
        self._file_state.loop_playback = on

    def _on_spin(self, on):
        if on:
            self.playback.start_spin()
        else:
            self.playback.stop_spin()
        self.state.spin_on = on

    def _on_export(self):
        _export.save_export(
            self, self._pane_container, defaults=self._export_defaults,
        )


def run(
    verts: np.ndarray,
    cells: list[list[int]],
    centers: np.ndarray,
    initial_n: int = 8,
    relax: bool = True,
    file_path: str | None = None,
    zoom_factor: float = 1.0,
    zoom_lon: float = 0.0,
    zoom_lat: float = 45.0,
    initial_grid: str = "ico",
    iim: int = 96,
    jjm: int = 95,
    lmdz_clon: float = 0.0,
    lmdz_clat: float = 0.0,
    lmdz_grossismx: float = 1.0,
    lmdz_grossismy: float = 1.0,
    lmdz_dzoomx: float = 0.0,
    lmdz_dzoomy: float = 0.0,
    lmdz_taux: float = 3.0,
    lmdz_tauy: float = 3.0,
) -> None:
    """Create the QApplication, show the main window, and start the Qt event loop."""
    app = QApplication.instance() or QApplication(sys.argv)
    # Set the icon on the QApplication BEFORE any window appears — that's the
    # only way macOS picks it up for the dock instead of showing the Python
    # interpreter's icon.
    icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    app.setApplicationName("IcoScope")
    app.setApplicationDisplayName("IcoScope")
    w = MainWindow(verts, cells, centers, initial_n=initial_n, relax=relax,
                   zoom_factor=zoom_factor, zoom_lon=zoom_lon, zoom_lat=zoom_lat,
                   iim=iim, jjm=jjm,
                   lmdz_clon=lmdz_clon, lmdz_clat=lmdz_clat,
                   lmdz_grossismx=lmdz_grossismx, lmdz_grossismy=lmdz_grossismy,
                   lmdz_dzoomx=lmdz_dzoomx, lmdz_dzoomy=lmdz_dzoomy,
                   lmdz_taux=lmdz_taux, lmdz_tauy=lmdz_tauy)
    if initial_grid == "lonlat" and not file_path:
        # Seed the lonlat cache with the mesh the CLI already built, then
        # switch tabs (which triggers _regen_lonlat → cache hit → render).
        w._lonlat_cache = {
            "params": w._lonlat_params_key(),
            "verts": verts,
            "cells": cells,
            "centers": np.asarray(centers),
        }
        w.panel.tabs.setCurrentIndex(Tab.LONLAT)
    elif initial_grid == "ico" and not file_path:
        # Seed the ico cache with the mesh the CLI already built so the
        # first File/LonLat → Ico tab switch hits the cache and doesn't
        # incur a fresh goldberg() regen.
        w._ico_cache = {
            "params": w._ico_params_key(),
            "verts": verts,
            "cells": cells,
            "centers": np.asarray(centers),
        }
    if file_path:
        # Load the file's mesh + fields as if the user had clicked Open in
        # the File tab. _on_open_file's logic is reused via the cache path.
        from .loader import FileContext, load_grid, read_levels
        f_verts, f_cells, f_centers, fields = load_grid(file_path)
        levels = read_levels(file_path)
        context = FileContext(file_path)
        w.file_path = file_path
        w._file_state.file_fields = fields
        w._file_state.file_levels = levels
        w._file_cache = {
            "path": file_path,
            "verts": f_verts,
            "cells": f_cells,
            "centers": np.asarray(f_centers),
            "fields": fields,
            "levels": levels,
            "context": context,
        }
        w.panel.file_tab.set_file_loaded(True)
        w._sync_file_info(file_path)
        w._activate_file_view()
        w.panel.tabs.setCurrentIndex(Tab.FILE)
    w.show()
    sys.exit(app.exec())
