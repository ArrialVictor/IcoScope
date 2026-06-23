"""Per-tab widgets for the IcoScope side panel.

Each tab (Ico, LonLat, File) is self-contained: it owns its own copy of the
display controls (Coloring, Overlays, Animation, Export) and re-emits their
signals at the tab level so the main window can wire connections per-tab.

The shared ``_DisplayBlock`` lives in :mod:`icoscope.display_block`;
``IcoTab`` and ``LonLatTab`` instantiate it without the time slider,
``FileTab`` with.
"""
from enum import IntEnum
from os.path import basename

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .display_block import _DisplayBlock


class Tab(IntEnum):
    """Side-panel tab indices.

    The integer values match the ``addTab`` order in
    :class:`icoscope.controls.ControlPanel`.
    """

    ICO = 0
    LONLAT = 1
    FILE = 2


class _ScrollArea(QScrollArea):
    """QScrollArea wrapping a tab's content so overflow becomes scrollable.

    Reports the wrapped widget's sizeHint so it cooperates with
    ``_AdaptiveTabWidget``'s height-following logic — when the panel has
    enough vertical space the scroll area takes its content's natural
    size (no scrollbar). When the panel is shorter than the content
    needs, the scroll area accepts the shorter allocation and shows a
    vertical scrollbar.
    """

    def __init__(self, content: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidget(content)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def sizeHint(self):
        w = self.widget()
        return w.sizeHint() if w is not None else super().sizeHint()


class _AdaptiveTabWidget(QTabWidget):
    """QTabWidget whose vertical size hint follows only the active tab's content.

    The stock QTabWidget reserves vertical space for the tallest tab. Overriding
    sizeHint / minimumSizeHint lets the panel shrink to fit the active tab.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.currentChanged.connect(lambda _i: self.updateGeometry())

    def sizeHint(self):
        return self._hint_for_current(use_min=False)

    def minimumSizeHint(self):
        return self._hint_for_current(use_min=True)

    def _hint_for_current(self, use_min: bool):
        page = self.currentWidget()
        base = super().minimumSizeHint() if use_min else super().sizeHint()
        if page is None:
            return base
        page_hint = page.minimumSizeHint() if use_min else page.sizeHint()
        bar_h = self.tabBar().sizeHint().height()
        base.setHeight(page_hint.height() + bar_h + 8)   # 8px frame padding
        return base


def _forward_signals(tab, block, names):
    """Expose ``block``'s signals as attributes of ``tab`` (no re-emit wrapper).

    Qt signals on the inner widget are valid signal objects; assigning them
    onto the tab gives callers ``tab.coastlines_toggled.connect(...)`` ergonomics
    without manually rewiring every connection.
    """
    for name in names:
        setattr(tab, name, getattr(block, name))


_DISPLAY_SIGNALS_BASE = (
    "color_by_changed", "cmap_changed", "colorbar_toggled", "center_zero_toggled",
    "coastlines_toggled", "graticule_toggled", "edges_toggled",
    "coast_color_changed", "grat_color_changed", "edge_color_changed",
    "coast_width_changed", "grat_width_changed", "edge_width_changed",
    "autorotate_toggled",
    "screenshot_clicked", "vector_export_clicked",
)
_DISPLAY_SIGNALS_TIME = ("time_changed", "play_toggled", "play_speed_changed",
                         "level_changed")


class IcoTab(QWidget):
    """Ico tab: synthetic Goldberg grid params + display controls (no time)."""

    n_changed           = Signal(int)
    relax_iters_changed = Signal(int)
    zoom_changed        = Signal(float, float, float)   # factor, lon_deg, lat_deg

    def __init__(self, cmaps: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)

        # ── Grid params ────────────────────────────
        gs = QGroupBox("Grid")
        gl = QFormLayout(gs)
        gl.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.n_box = QSpinBox()
        self.n_box.setRange(1, 100)
        self.n_box.setValue(40)
        self.n_box.setKeyboardTracking(False)
        self.n_box.valueChanged.connect(self.n_changed)
        gl.addRow("Cell frequency", self.n_box)

        self.relax_iters_box = QSpinBox()
        self.relax_iters_box.setRange(0, 1000)
        self.relax_iters_box.setSingleStep(10)
        self.relax_iters_box.setValue(200)
        self.relax_iters_box.setKeyboardTracking(False)
        self.relax_iters_box.valueChanged.connect(self.relax_iters_changed)
        gl.addRow("Max relax iter", self.relax_iters_box)

        v.addWidget(gs)

        # ── Synthetic zoom (Schmidt) ───────────────
        zg = QGroupBox("Synthetic zoom (Schmidt)")
        zl = QFormLayout(zg)
        zl.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.zoom_factor_box = QDoubleSpinBox()
        self.zoom_factor_box.setRange(0.1, 10.0)
        self.zoom_factor_box.setSingleStep(0.1)
        self.zoom_factor_box.setDecimals(2)
        self.zoom_factor_box.setValue(1.0)
        self.zoom_factor_box.setKeyboardTracking(False)
        self.zoom_factor_box.valueChanged.connect(self._on_zoom_spinbox_changed)
        zl.addRow("Factor", self.zoom_factor_box)

        self.zoom_lon_box = QDoubleSpinBox()
        self.zoom_lon_box.setRange(-180.0, 180.0)
        self.zoom_lon_box.setSingleStep(5.0)
        self.zoom_lon_box.setDecimals(2)
        self.zoom_lon_box.setValue(0.0)
        self.zoom_lon_box.setKeyboardTracking(False)
        self.zoom_lon_box.valueChanged.connect(self._on_zoom_spinbox_changed)
        zl.addRow("Center lon (°)", self.zoom_lon_box)

        self.zoom_lat_box = QDoubleSpinBox()
        self.zoom_lat_box.setRange(-90.0, 90.0)
        self.zoom_lat_box.setSingleStep(5.0)
        self.zoom_lat_box.setDecimals(2)
        self.zoom_lat_box.setValue(45.0)
        self.zoom_lat_box.setKeyboardTracking(False)
        self.zoom_lat_box.valueChanged.connect(self._on_zoom_spinbox_changed)
        zl.addRow("Center lat (°)", self.zoom_lat_box)

        self.zoom_toggle_btn = QPushButton("Activate zoom")
        self._zoom_active = False
        self.zoom_toggle_btn.clicked.connect(self._toggle_zoom)
        zl.addRow(self.zoom_toggle_btn)

        v.addWidget(zg)

        # ── Display block ──────────────────────────
        self.display = _DisplayBlock(cmaps, with_time=False)
        v.addWidget(self.display)
        _forward_signals(self, self.display, _DISPLAY_SIGNALS_BASE)

        v.addStretch(1)

    # ── proxy display methods ────────────────────────────────────────────

    def set_cmap(self, name: str) -> None:
        """Forward to the inner display block."""
        self.display.set_cmap(name)

    def set_color_by(self, name: str) -> None:
        """Forward to the inner display block."""
        self.display.set_color_by(name)

    def set_color_by_items(self, items: list[str]) -> None:
        """Forward to the inner display block."""
        self.display.set_color_by_items(items)

    def set_edge_color(self, hex_str: str) -> None:
        """Forward to the inner display block."""
        self.display.set_edge_color(hex_str)

    def set_coast_color(self, hex_str: str) -> None:
        """Forward to the inner display block."""
        self.display.set_coast_color(hex_str)

    def set_grat_color(self, hex_str: str) -> None:
        """Forward to the inner display block."""
        self.display.set_grat_color(hex_str)

    # ── zoom helpers ─────────────────────────────────────────────────────

    def _toggle_zoom(self):
        """Activate or deactivate the synthetic zoom (button-as-toggle)."""
        if self._zoom_active:
            self._zoom_active = False
            self.zoom_toggle_btn.setText("Activate zoom")
            self.zoom_changed.emit(
                1.0,
                self.zoom_lon_box.value(),
                self.zoom_lat_box.value(),
            )
        else:
            self._zoom_active = True
            self.zoom_toggle_btn.setText("Deactivate zoom")
            self.zoom_changed.emit(
                self.zoom_factor_box.value(),
                self.zoom_lon_box.value(),
                self.zoom_lat_box.value(),
            )

    def _on_zoom_spinbox_changed(self, _value):
        """Live re-apply when any zoom spinbox is edited while zoom is active."""
        if self._zoom_active:
            self.zoom_changed.emit(
                self.zoom_factor_box.value(),
                self.zoom_lon_box.value(),
                self.zoom_lat_box.value(),
            )

    def set_zoom(self, factor: float, lon: float, lat: float) -> None:
        """Sync the zoom spinboxes (and toggle state) to the given values."""
        for box, val in ((self.zoom_factor_box, factor),
                         (self.zoom_lon_box, lon),
                         (self.zoom_lat_box, lat)):
            box.blockSignals(True)
            box.setValue(float(val))
            box.blockSignals(False)
        active = abs(float(factor) - 1.0) >= 1e-12
        self._zoom_active = active
        self.zoom_toggle_btn.setText("Deactivate zoom" if active else "Activate zoom")


class LonLatTab(QWidget):
    """LonLat tab: synthetic dyn3d-style regular lat-lon mesh + display controls."""

    iim_changed = Signal(int)
    jjm_changed = Signal(int)
    # 8 params: clon, clat, grossismx, grossismy, dzoomx, dzoomy, taux, tauy
    lmdz_zoom_changed = Signal(float, float, float, float, float, float, float, float)

    def __init__(self, cmaps: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)

        # ── Grid params ────────────────────────────
        gs = QGroupBox("Grid")
        gl = QFormLayout(gs)
        gl.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.iim_box = QSpinBox()
        self.iim_box.setRange(2, 720)
        self.iim_box.setValue(96)
        self.iim_box.setKeyboardTracking(False)
        self.iim_box.valueChanged.connect(self.iim_changed)
        gl.addRow("Longitudes (iim)", self.iim_box)

        self.jjm_box = QSpinBox()
        self.jjm_box.setRange(1, 360)
        self.jjm_box.setValue(95)
        self.jjm_box.setKeyboardTracking(False)
        self.jjm_box.valueChanged.connect(self.jjm_changed)
        gl.addRow("Latitude bands (jjm)", self.jjm_box)

        v.addWidget(gs)

        # ── Synthetic zoom (LMDZ tanh) ─────────────
        zg = QGroupBox("Synthetic zoom (LMDZ tanh)")
        zl = QFormLayout(zg)
        zl.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        def _mkbox(lo, hi, step, decimals, value):
            b = QDoubleSpinBox()
            b.setRange(lo, hi)
            b.setSingleStep(step)
            b.setDecimals(decimals)
            b.setValue(value)
            b.setKeyboardTracking(False)
            b.valueChanged.connect(self._on_lmdz_spinbox_changed)
            return b

        self.lmdz_clon_box = _mkbox(-180.0, 180.0, 5.0, 2, 0.0)
        zl.addRow("Center lon (°)", self.lmdz_clon_box)
        self.lmdz_clat_box = _mkbox(-90.0, 90.0, 5.0, 2, 0.0)
        zl.addRow("Center lat (°)", self.lmdz_clat_box)
        self.lmdz_grossismx_box = _mkbox(1.0, 20.0, 0.1, 2, 1.0)
        zl.addRow("Grossism x", self.lmdz_grossismx_box)
        self.lmdz_grossismy_box = _mkbox(1.0, 20.0, 0.1, 2, 1.0)
        zl.addRow("Grossism y", self.lmdz_grossismy_box)
        self.lmdz_dzoomx_box = _mkbox(0.0, 0.5, 0.01, 3, 0.0)
        zl.addRow("Dzoom x", self.lmdz_dzoomx_box)
        self.lmdz_dzoomy_box = _mkbox(0.0, 0.5, 0.01, 3, 0.0)
        zl.addRow("Dzoom y", self.lmdz_dzoomy_box)
        self.lmdz_taux_box = _mkbox(0.1, 20.0, 0.5, 2, 3.0)
        zl.addRow("Tau x", self.lmdz_taux_box)
        self.lmdz_tauy_box = _mkbox(0.1, 20.0, 0.5, 2, 3.0)
        zl.addRow("Tau y", self.lmdz_tauy_box)

        self.lmdz_zoom_toggle_btn = QPushButton("Activate zoom")
        self._lmdz_zoom_active = False
        self.lmdz_zoom_toggle_btn.clicked.connect(self._toggle_lmdz_zoom)
        zl.addRow(self.lmdz_zoom_toggle_btn)

        v.addWidget(zg)

        # ── Display block ──────────────────────────
        self.display = _DisplayBlock(cmaps, with_time=False)
        v.addWidget(self.display)
        _forward_signals(self, self.display, _DISPLAY_SIGNALS_BASE)

        v.addStretch(1)

    # ── LMDZ zoom helpers ────────────────────────────────────────────────

    def _current_lmdz_values(self):
        return (
            self.lmdz_clon_box.value(),
            self.lmdz_clat_box.value(),
            self.lmdz_grossismx_box.value(),
            self.lmdz_grossismy_box.value(),
            self.lmdz_dzoomx_box.value(),
            self.lmdz_dzoomy_box.value(),
            self.lmdz_taux_box.value(),
            self.lmdz_tauy_box.value(),
        )

    def _toggle_lmdz_zoom(self, *_args):
        """Activate or deactivate the LMDZ tanh zoom (button-as-toggle)."""
        if self._lmdz_zoom_active:
            self._lmdz_zoom_active = False
            self.lmdz_zoom_toggle_btn.setText("Activate zoom")
            # Identity emission: grossism = 1 forces uniform fast path.
            vals = self._current_lmdz_values()
            self.lmdz_zoom_changed.emit(
                vals[0], vals[1], 1.0, 1.0, vals[4], vals[5], vals[6], vals[7],
            )
        else:
            self._lmdz_zoom_active = True
            self.lmdz_zoom_toggle_btn.setText("Deactivate zoom")
            self.lmdz_zoom_changed.emit(*self._current_lmdz_values())

    @property
    def lmdz_zoom_active(self) -> bool:
        """Whether the LMDZ-zoom toggle is currently on."""
        return self._lmdz_zoom_active

    def set_lmdz_zoom_toggle_enabled(self, enabled: bool):
        """Enable or disable the Activate/Deactivate zoom button.

        Used by the main window to block activation when the current
        spinbox combination would fail LMDZ's 2β-G>0 validity check.
        """
        self.lmdz_zoom_toggle_btn.setEnabled(enabled)

    def _on_lmdz_spinbox_changed(self, _value):
        """Re-emit on any LMDZ-zoom spinbox edit, regardless of toggle state.

        The app's slot validates the combination either way — when active it
        rebuilds the mesh (and reverts on error); when inactive it only
        runs the cheap validity check and shows the red error message.
        """
        self.lmdz_zoom_changed.emit(*self._current_lmdz_values())

    def set_lmdz_zoom(
        self,
        clon: float, clat: float,
        grossismx: float, grossismy: float,
        dzoomx: float, dzoomy: float,
        taux: float, tauy: float,
    ) -> None:
        """Sync the LMDZ-zoom spinboxes and toggle state to the given values."""
        # Drop focus before the revert: the line-editor's uncommitted text
        # would otherwise paint over the value we set until focus leaves.
        for b in (self.lmdz_clon_box, self.lmdz_clat_box,
                  self.lmdz_grossismx_box, self.lmdz_grossismy_box,
                  self.lmdz_dzoomx_box, self.lmdz_dzoomy_box,
                  self.lmdz_taux_box, self.lmdz_tauy_box):
            if b.hasFocus():
                b.clearFocus()
        for box, val in (
            (self.lmdz_clon_box, clon),
            (self.lmdz_clat_box, clat),
            (self.lmdz_grossismx_box, grossismx),
            (self.lmdz_grossismy_box, grossismy),
            (self.lmdz_dzoomx_box, dzoomx),
            (self.lmdz_dzoomy_box, dzoomy),
            (self.lmdz_taux_box, taux),
            (self.lmdz_tauy_box, tauy),
        ):
            box.blockSignals(True)
            box.setValue(float(val))
            box.blockSignals(False)
        active = (
            abs(float(grossismx) - 1.0) >= 1e-12
            or abs(float(grossismy) - 1.0) >= 1e-12
        )
        self._lmdz_zoom_active = active
        self.lmdz_zoom_toggle_btn.setText(
            "Deactivate zoom" if active else "Activate zoom"
        )

    # ── proxy display methods ────────────────────────────────────────────

    def set_cmap(self, name: str) -> None:
        """Forward to the inner display block."""
        self.display.set_cmap(name)

    def set_color_by(self, name: str) -> None:
        """Forward to the inner display block."""
        self.display.set_color_by(name)

    def set_color_by_items(self, items: list[str]) -> None:
        """Forward to the inner display block."""
        self.display.set_color_by_items(items)

    def set_edge_color(self, hex_str: str) -> None:
        """Forward to the inner display block."""
        self.display.set_edge_color(hex_str)

    def set_coast_color(self, hex_str: str) -> None:
        """Forward to the inner display block."""
        self.display.set_coast_color(hex_str)

    def set_grat_color(self, hex_str: str) -> None:
        """Forward to the inner display block."""
        self.display.set_grat_color(hex_str)


class FileTab(QWidget):
    """File tab: Open/Unload + file summary + display controls (with time)."""

    open_file_clicked  = Signal()
    close_file_clicked = Signal()

    def __init__(self, cmaps: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        # Open ↔ Unload toggle: same button, label/behavior switches with state.
        self.file_btn = QPushButton("Open NetCDF…")
        self._file_btn_mode = "open"
        self.file_btn.clicked.connect(self._on_file_btn_clicked)
        v.addWidget(self.file_btn)

        # File summary block — hidden until a file is loaded. All three labels
        # wrap so a long filename or CF title doesn't stretch the panel sideways.
        self.file_name_label = QLabel("")
        self.file_name_label.setStyleSheet("font-weight: bold; padding-top: 6px;")
        self.file_name_label.setWordWrap(True)
        self.file_name_label.setVisible(False)
        v.addWidget(self.file_name_label)

        self.file_stats_label = QLabel("")
        self.file_stats_label.setStyleSheet("padding-top: 4px;")
        self.file_stats_label.setWordWrap(True)
        self.file_stats_label.setVisible(False)
        v.addWidget(self.file_stats_label)

        self.file_attrs_label = QLabel("")
        self.file_attrs_label.setStyleSheet("color: #888; font-size: 11px; padding-top: 4px;")
        self.file_attrs_label.setWordWrap(True)
        self.file_attrs_label.setVisible(False)
        v.addWidget(self.file_attrs_label)

        # ── Display block (with time slider) ───────
        self.display = _DisplayBlock(cmaps, with_time=True)
        # File tab's Color by is only ever the loaded file's fields. Replace
        # the default synthetic options that _DisplayBlock pre-populates.
        self.display.set_color_by_items(["None"])
        v.addWidget(self.display)
        _forward_signals(self, self.display,
                         _DISPLAY_SIGNALS_BASE + _DISPLAY_SIGNALS_TIME)

        v.addStretch(1)

    # ── proxy display methods ────────────────────────────────────────────

    def set_cmap(self, name: str) -> None:
        """Forward to the inner display block."""
        self.display.set_cmap(name)

    def set_color_by(self, name: str) -> None:
        """Forward to the inner display block."""
        self.display.set_color_by(name)

    def set_color_by_items(self, items: list[str]) -> None:
        """Forward to the inner display block."""
        self.display.set_color_by_items(items)

    def set_edge_color(self, hex_str: str) -> None:
        """Forward to the inner display block."""
        self.display.set_edge_color(hex_str)

    def set_coast_color(self, hex_str: str) -> None:
        """Forward to the inner display block."""
        self.display.set_coast_color(hex_str)

    def set_grat_color(self, hex_str: str) -> None:
        """Forward to the inner display block."""
        self.display.set_grat_color(hex_str)

    def set_time_axis(self, n_steps: int, times=None) -> None:
        """Forward to the inner display block."""
        self.display.set_time_axis(n_steps, times)

    def set_time_steps(self, n_steps: int) -> None:
        """Forward to the inner display block (no datetime info; legacy alias)."""
        self.display.set_time_steps(n_steps)

    def set_time_label(self, idx: int) -> None:
        """Forward to the inner display block."""
        self.display.set_time_label(idx)

    def set_levels(self, levels_pa) -> None:
        """Forward to the inner display block."""
        self.display.set_levels(levels_pa)

    def set_level_label(self, idx: int) -> None:
        """Forward to the inner display block."""
        self.display.set_level_label(idx)

    # ── file controls ────────────────────────────────────────────────────

    def set_file_loaded(self, loaded: bool):
        """Swap the file button between 'Open NetCDF…' and 'Unload NetCDF'."""
        self._file_btn_mode = "close" if loaded else "open"
        self.file_btn.setText("Unload NetCDF" if loaded else "Open NetCDF…")

    def set_file_info(
        self,
        path: str = "",
        n_cells: int = 0,
        n_fields: int = 0,
        n_time_steps: int = 0,
        attrs: dict | None = None,
    ) -> None:
        """Populate the File tab's summary block.

        Pass empty path (or call with no args) to clear everything back to the
        unloaded state. ``attrs`` is the NetCDF global-attributes dict; only
        ``title`` and ``source`` are surfaced if present.
        """
        if not path:
            for lbl in (self.file_name_label, self.file_stats_label,
                        self.file_attrs_label):
                lbl.setText("")
                lbl.setVisible(False)
            self.file_name_label.setToolTip("")
            self.updateGeometry()
            return

        # Qt word-wrap only breaks on whitespace + hyphens, so a long run of
        # underscore-joined tokens (typical of ICOLMDZ histday filenames) stays
        # as one unbreakable word. Inject zero-width spaces after underscores
        # and dots so the wrap algorithm has somewhere to break. Tooltip keeps
        # the verbatim path.
        name = basename(path)
        zwsp = "\u200b"
        wrapped = name.replace("_", "_" + zwsp).replace(".", "." + zwsp)
        self.file_name_label.setText(wrapped)
        self.file_name_label.setToolTip(path)
        self.file_name_label.setVisible(True)

        stats = []
        if n_cells:
            stats.append(f"{n_cells:,} cells")
        if n_fields:
            stats.append(f"{n_fields} field{'s' if n_fields != 1 else ''}")
        if n_time_steps > 1:
            stats.append(f"{n_time_steps} time steps")
        self.file_stats_label.setText(" · ".join(stats))
        self.file_stats_label.setVisible(bool(stats))

        attr_lines = []
        attrs = attrs or {}
        for key in ("title", "source"):
            val = attrs.get(key) or attrs.get(key.capitalize())
            if val:
                attr_lines.append(f"{key.capitalize()}: {val}")
        self.file_attrs_label.setText("\n".join(attr_lines))
        self.file_attrs_label.setVisible(bool(attr_lines))

        self.updateGeometry()

    def _on_file_btn_clicked(self):
        if self._file_btn_mode == "open":
            self.open_file_clicked.emit()
        else:
            self.close_file_clicked.emit()
