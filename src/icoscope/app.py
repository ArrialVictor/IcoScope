"""Qt main window: 3D viewer + control panel + status bar."""
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
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
    QWidget,
)

from . import export as _export
from . import menubar as _menubar
from .coastlines import coastline_polydata
from .controls import ControlPanel
from .graticule import graticule_polydata
from .grid import goldberg
from .lonlat import latlon_mesh
from .picker import Picker
from .playback import Playback
from .tabs import Tab
from .themes import CMAPS, THEMES


@dataclass
class _TabState:
    """Per-tab display state — colormap, overlays, widths, color overrides.

    Each tab in the side panel owns one of these so switching tabs is fully
    stateful: the user's choices on the Ico tab don't leak into the File tab
    and vice-versa.
    """

    color_by: str = "None"
    cmap: str = "viridis"
    coastlines_on: bool = False
    graticule_on: bool = False
    edges_on: bool = True
    colorbar_on: bool = True
    center_zero: bool = False
    spin_on: bool = False
    edge_color_override: str | None = None
    coast_color_override: str | None = None
    grat_color_override: str | None = None
    edge_width: float = 0.6
    coast_width: float = 1.2
    grat_width: float = 0.6
    time_index: int = 0
    level_index: int = 0
    # file-only fields
    file_fields: dict = field(default_factory=dict)
    # presnivs (Pa) for the loaded file, or None if no vertical dim
    file_levels: object = None


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
        self.scalars = None           # what we actually render
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
        self.transparent_export = False

        # Theme is window-level (background colour + default overlay tints).
        self.theme_name = "Dark"
        default_cmap = THEMES[self.theme_name]["cmap"]

        # Per-tab display state — each tab keeps its own coloring, overlays,
        # animation, and color-by selection. Switching tabs swaps which
        # state is read by the rendering code (via ``self.state``).
        self._ico_state = _TabState(cmap=default_cmap)
        self._lonlat_state = _TabState(cmap=default_cmap)
        self._file_state = _TabState(cmap=default_cmap)

        # central layout: a horizontal splitter so the user can drag the
        # divider between the 3-D view and the control panel. Index 0 (the
        # plotter) carries the stretch on window resize; index 1 (the panel)
        # has a minimum width to stay readable.
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(4)
        self.plotter = QtInteractor(self.splitter)
        self.splitter.addWidget(self.plotter.interactor)
        self.panel = ControlPanel(CMAPS)
        self.splitter.addWidget(self.panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        # Use a wide initial plotter allocation; the panel takes its preferred
        # width and the splitter clamps to its minimum on shrink.
        self.splitter.setSizes([1000, ControlPanel.DEFAULT_WIDTH])
        self.setCentralWidget(self.splitter)

        self._theme_actions = _menubar.build_menubar(
            self, self.theme_name, self._on_theme)
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
            tab.edge_width_changed.connect(self._on_edge_width)
            tab.coast_width_changed.connect(self._on_coast_width)
            tab.grat_width_changed.connect(self._on_grat_width)
            tab.autorotate_toggled.connect(self._on_spin)
            tab.screenshot_clicked.connect(self._on_screenshot)
            tab.vector_export_clicked.connect(self._on_vector_export)

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
        self.panel.tabs.currentChanged.connect(self._on_tab_changed)

        # Cached meshes so tab-switching doesn't trigger expensive recomputes.
        self._file_cache: dict | None = None
        self._ico_cache: dict | None = None
        self._lonlat_cache: dict | None = None

        # build scene + interactions
        self._refresh_scalars()
        self._mesh = self._to_polydata()

        # Helpers — instantiate after _mesh exists; the picker reads it lazily.
        self.picker = Picker(self, self.plotter)
        self.playback = Playback(self, self.plotter)

        self._build_scene()
        self.plotter.reset_camera()
        self.picker.attach()
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
    def active_tab(self):
        """Return the currently-active tab widget (Ico, LonLat, or File)."""
        idx = self.panel.tabs.currentIndex()
        if idx == Tab.ICO:
            return self.panel.ico_tab
        if idx == Tab.FILE:
            return self.panel.file_tab
        return self.panel.lonlat_tab

    # ── colors ─────────────────────────────────────
    def _edge_color(self):
        return self.state.edge_color_override or THEMES[self.theme_name]["edge"]

    def _coast_color(self):
        return self.state.coast_color_override or THEMES[self.theme_name]["coast"]

    def _grat_color(self):
        return self.state.grat_color_override or THEMES[self.theme_name].get(
            "grat", self._coast_color())

    def _sync_color_buttons(self):
        hex_edge = self._color_to_hex(self._edge_color())
        hex_coast = self._color_to_hex(self._coast_color())
        hex_grat = self._color_to_hex(self._grat_color())
        for tab in (self.panel.ico_tab, self.panel.lonlat_tab, self.panel.file_tab):
            tab.set_edge_color(hex_edge)
            tab.set_coast_color(hex_coast)
            tab.set_grat_color(hex_grat)

    @staticmethod
    def _color_to_hex(c):
        # accept "#rrggbb" or named CSS colors
        from qtpy.QtGui import QColor
        return QColor(c).name()

    # ── geometry ──────────────────────────────────
    def _to_polydata(self):
        faces_flat = []
        for c in self.cells:
            faces_flat.append(len(c))
            faces_flat.extend(c)
        mesh = pv.PolyData(self.verts, faces=np.array(faces_flat, dtype=np.int64))
        if self.scalars is not None:
            mesh.cell_data["scalars"] = np.asarray(self.scalars)
        return mesh

    def _clim(self):
        if self.scalars is None or not self.state.center_zero:
            return None
        s = np.asarray(self.scalars)
        a = float(np.nanmax(np.abs(s)))
        return [-a, a] if a > 0 else None

    # ── rendering ─────────────────────────────────
    def _build_scene(self):
        # Defer to the empty-sphere path when there's no mesh to render
        # (File tab pre-load, LonLat placeholder). This makes overlay
        # toggles a no-op in that state instead of crashing.
        if self._mesh is None:
            self._render_empty_sphere()
            return
        # Coming back from the empty-sphere state, the "empty" actor stays
        # in the plotter unless we explicitly drop it — it would render
        # underneath the real mesh.
        self.plotter.remove_actor("empty", reset_camera=False, render=False)
        theme = THEMES[self.theme_name]
        self.plotter.set_background(theme["bg"])
        st = self.state

        self.plotter.add_mesh(
            self._mesh, name="grid",
            scalars="scalars" if self.scalars is not None else None,
            cmap=st.cmap,
            clim=self._clim(),
            show_edges=st.edges_on,
            edge_color=self._edge_color(),
            line_width=st.edge_width,
            smooth_shading=False,
            show_scalar_bar=st.colorbar_on and self.scalars is not None,
            reset_camera=False,
        )

        if st.coastlines_on:
            try:
                cl = coastline_polydata(radius=1.001)
                self.plotter.add_mesh(cl, name="coast",
                                      color=self._coast_color(),
                                      line_width=st.coast_width,
                                      pickable=False, reset_camera=False)
            except Exception as e:
                self.statusBar().showMessage(f"coastlines failed: {e}")
        else:
            self.plotter.remove_actor("coast", reset_camera=False, render=False)

        if st.graticule_on:
            try:
                g = graticule_polydata(radius=1.002, spacing=30)
                self.plotter.add_mesh(g, name="grat",
                                      color=self._grat_color(),
                                      line_width=st.grat_width,
                                      opacity=0.6, pickable=False, reset_camera=False)
            except Exception as e:
                self.statusBar().showMessage(f"graticule failed: {e}")
        else:
            self.plotter.remove_actor("grat", reset_camera=False, render=False)

        self.plotter.render()

    # ── ESC: clear current selection + stop spin ──
    def _on_escape(self):
        self.picker.clear_highlight()
        self._clear_lonlat()
        if self.state.spin_on:
            self.state.spin_on = False
            self.playback.stop_spin()
            # Uncheck on whichever tab's spin checkbox is currently checked.
            for tab in (self.panel.ico_tab, self.panel.lonlat_tab, self.panel.file_tab):
                cb = tab.display.spin_cb
                if cb.isChecked():
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
        h.addWidget(QLabel("lon"))
        self.lon_box = QDoubleSpinBox()
        self.lon_box.setRange(self.LON_SENTINEL, 180)
        self.lon_box.setDecimals(2)
        self.lon_box.setSuffix("°")
        self.lon_box.setSpecialValueText("—")
        self.lon_box.setFixedWidth(80)
        self.lon_box.setKeyboardTracking(False)
        self.lon_box.setValue(self.LON_SENTINEL)
        h.addWidget(self.lon_box)
        h.addWidget(QLabel("lat"))
        self.lat_box = QDoubleSpinBox()
        self.lat_box.setRange(self.LAT_SENTINEL, 90)
        self.lat_box.setDecimals(2)
        self.lat_box.setSuffix("°")
        self.lat_box.setSpecialValueText("—")
        self.lat_box.setFixedWidth(80)
        self.lat_box.setKeyboardTracking(False)
        self.lat_box.setValue(self.LAT_SENTINEL)
        h.addWidget(self.lat_box)
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
        self.state.cmap = name
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

    def _refresh_scalars(self):
        """Compute self.scalars from the current `color_by` choice."""
        color_by = self.state.color_by
        file_fields = self._file_state.file_fields
        if color_by == "Latitude":
            self.scalars = np.degrees(np.arcsin(self.centers[:, 2]))
        elif color_by == "Cell kind":
            self.scalars = np.array([0 if len(c) == 5 else 1 for c in self.cells],
                                    dtype=float)
        elif color_by == "Mock temperature":
            # Clean synthetic field: latitude gradient + a faint zonal wave.
            # Same formula as the `tas` field in tools/make_test_nc.py.
            c = self.centers / np.linalg.norm(self.centers, axis=1, keepdims=True)
            lat = np.arcsin(np.clip(c[:, 2], -1, 1))
            lon = np.arctan2(c[:, 1], c[:, 0])
            self.scalars = 250.0 + 50.0 * np.cos(lat) + 5.0 * np.cos(2 * lon)
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

            self.scalars = T
        elif color_by in file_fields and self.file_path:
            from .loader import read_field
            self.scalars = read_field(
                self.file_path, color_by,
                time_index=self._file_state.time_index,
                level_index=self._file_state.level_index,
            )
        else:
            self.scalars = None

    def _on_color_by(self, name):
        st = self.state
        st.color_by = name
        # Enable/disable cmap-related widgets on the tab that emitted the change.
        tab = self.active_tab
        if hasattr(tab, "display"):
            tab.display.center_cb.setEnabled(name != "None")
            tab.display.bar_cb.setEnabled(name != "None")
            tab.display.cmap_box.setEnabled(name != "None")
        # configure the time + level sliders for the active field (File tab only).
        meta = self._file_state.file_fields.get(name) if tab is self.panel.file_tab else None
        if tab is self.panel.file_tab:
            if meta and meta.get("time_varying"):
                self.panel.file_tab.set_time_steps(meta["shape"][0])
            else:
                self.panel.file_tab.set_time_steps(0)
            self._file_state.time_index = 0
            n_levels = meta.get("n_levels", 0) if meta else 0
            if n_levels > 1 and self._file_state.file_levels is not None:
                self.panel.file_tab.set_levels(self._file_state.file_levels)
            else:
                self.panel.file_tab.set_levels(None)
            self._file_state.level_index = 0
        self._refresh_scalars()
        self._mesh = self._to_polydata()
        self.picker.invalidate_locator()
        self._build_scene()

    def _on_colorbar(self, on):
        self.state.colorbar_on = on
        self._build_scene()

    def _on_center_zero(self, on):
        self.state.center_zero = on
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
        self.picker.invalidate_locator()
        self.picker.clear_highlight()
        self._build_scene()
        self._update_status()

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
            from .loader import load_grid, read_levels
            verts, cells, centers, fields = load_grid(path)
            levels = read_levels(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
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
        self.file_path = None
        self._file_state.file_fields = {}
        self._file_state.file_levels = None
        self._file_cache = None
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
        if meta and meta.get("time_varying"):
            n_t = meta["shape"][0]
            self.panel.file_tab.set_time_steps(n_t)
            # Clamp the saved index in case the caller is recycling state
            # across a different field/file (e.g. via _on_open_file).
            t_idx = min(max(self._file_state.time_index, 0), n_t - 1)
            self._file_state.time_index = t_idx
            slider = self.panel.file_tab.display.time_slider
            slider.blockSignals(True)
            slider.setValue(t_idx)
            slider.blockSignals(False)
            self.panel.file_tab.set_time_label(t_idx, n_t)
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
            self.panel.file_tab.display._update_level_label(l_idx)
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
        self.panel.file_tab.display.cmap_box.setEnabled(enable)
        self._apply_mesh_change()

    def _on_tab_changed(self, idx: int):
        """Tab is the active mesh source — swap the rendered scene accordingly."""
        if idx == Tab.ICO:
            self._regen_synthetic()
        elif idx == Tab.LONLAT:
            self._regen_lonlat()
        elif idx == Tab.FILE:
            if self._file_cache is not None:
                self._activate_file_view()
            else:
                # No file loaded — show a plain empty sphere instead of
                # whatever was last rendered. The File tab's overlay
                # settings are ignored in this state since there's no
                # geographic data to overlay onto.
                self._render_empty_sphere()
        # Per-tab colour overrides may differ → refresh swatches.
        self._sync_color_buttons()
        # Auto-rotate is per-tab state but the timer is window-level — sync
        # the timer to the new active tab's spin_on flag.
        if self.state.spin_on:
            self.playback.start_spin()
        else:
            self.playback.stop_spin()
        self._update_status()

    def _render_empty_sphere(self):
        """Render a plain blank sphere (no cells, no overlays).

        Used when a tab is active but has no mesh to show yet — the
        File tab before any NetCDF is loaded, and the LonLat placeholder
        tab. Conveys "this is the canvas, populate it" without leaking
        stale geometry from another tab into the view.
        """
        self.plotter.clear()
        self.plotter.set_background(THEMES[self.theme_name]["bg"])
        # pv.Sphere uses theta/phi tessellation, which leaves visible latitude
        # rings even with smooth_shading on. Icosphere has no pole singularity
        # and no axis-aligned strips, so the surface reads as a clean sphere.
        sphere = pv.Icosphere(radius=1.0, nsub=5)
        self.plotter.add_mesh(sphere, name="empty",
                              color="#777777", show_edges=False,
                              smooth_shading=True, reset_camera=False)
        self._mesh = None
        self.scalars = None
        self.picker.invalidate_locator()
        self.picker.clear_highlight()
        self.plotter.render()
        self._update_status()

    def _on_time_changed(self, idx):
        if idx == self._file_state.time_index:
            return
        self._file_state.time_index = idx
        meta = self._file_state.file_fields.get(self._file_state.color_by)
        if meta:
            self.panel.file_tab.set_time_label(idx, meta["shape"][0])
        self._refresh_scalars()
        self._mesh = self._to_polydata()
        self.picker.invalidate_locator()
        self._build_scene()

    def _on_level_changed(self, idx):
        if idx == self._file_state.level_index:
            return
        self._file_state.level_index = idx
        self._refresh_scalars()
        self._mesh = self._to_polydata()
        self.picker.invalidate_locator()
        self._build_scene()

    def _on_play_toggled(self, on):
        self.playback.toggle_play(on)

    def _on_play_speed_changed(self, ms):
        self.playback.set_speed(ms)

    def _on_spin(self, on):
        if on:
            self.playback.start_spin()
        else:
            self.playback.stop_spin()
        self.state.spin_on = on

    def _on_screenshot(self):
        self.transparent_export = _export.save_screenshot(
            self, self.plotter, transparent=self.transparent_export)

    def _on_vector_export(self):
        _export.save_vector(self, self.plotter)


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
        from .loader import load_grid, read_levels
        f_verts, f_cells, f_centers, fields = load_grid(file_path)
        levels = read_levels(file_path)
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
        }
        w.panel.file_tab.set_file_loaded(True)
        w._sync_file_info(file_path)
        w._activate_file_view()
        w.panel.tabs.setCurrentIndex(Tab.FILE)
    w.show()
    sys.exit(app.exec())
