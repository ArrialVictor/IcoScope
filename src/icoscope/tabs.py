"""Per-tab widgets for the IcoScope side panel.

Each tab (Ico, LonLat, File) is self-contained: it owns its own copy of the
display controls (Coloring, Overlays, Animation, Export) and re-emits their
signals at the tab level so the main window can wire connections per-tab.

The shared internal ``_DisplayBlock`` builds the standard display controls;
``IcoTab`` instantiates it without the time slider, ``FileTab`` with.
"""
from os.path import basename

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .controls import ColorButton, _expand

# Color-by options shown for the synthetic Ico mesh (no NetCDF loaded). Once a
# file is loaded, the File tab's combo is replaced by "None" + the file's own
# field names; the Ico tab keeps this list independently.
SYNTHETIC_COLOR_BY = [
    "None", "Latitude", "Cell kind", "Mock temperature", "Realistic temperature",
]


class _AdaptiveTabWidget(QTabWidget):
    """QTabWidget whose vertical size hint follows only the active tab's content.

    The stock QTabWidget reserves vertical space for the tallest tab. Overriding
    sizeHint / minimumSizeHint lets the panel shrink to fit the active tab.
    """

    def __init__(self, parent=None):
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


class _DisplayBlock(QWidget):
    """Standard display controls: Coloring + Overlays + Animation + Export.

    Reused inside IcoTab and FileTab. The ``with_time`` flag controls whether
    the Animation group includes the time slider, play button, and speed
    spinbox (only meaningful for the File tab where a time-varying field can
    be active).
    """

    # Coloring
    cmap_changed        = Signal(str)
    color_by_changed    = Signal(str)
    colorbar_toggled    = Signal(bool)
    center_zero_toggled = Signal(bool)
    edge_color_changed  = Signal(str)
    coast_color_changed = Signal(str)
    grat_color_changed  = Signal(str)

    # Overlays
    coastlines_toggled  = Signal(bool)
    graticule_toggled   = Signal(bool)
    edges_toggled       = Signal(bool)
    coast_width_changed = Signal(float)
    grat_width_changed  = Signal(float)
    edge_width_changed  = Signal(float)

    # Animation
    autorotate_toggled  = Signal(bool)

    # Time (only emitted if with_time=True)
    time_changed        = Signal(int)
    play_toggled        = Signal(bool)
    play_speed_changed  = Signal(int)

    # Export
    screenshot_clicked     = Signal()
    vector_export_clicked  = Signal()

    def __init__(self, cmaps, with_time: bool, parent=None):
        super().__init__(parent)
        self.with_time = with_time
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._build_coloring_group(cmaps))
        v.addWidget(self._build_overlays_group())
        v.addWidget(self._build_animation_group())
        v.addWidget(self._build_export_group())

    # ── group builders ────────────────────────────────────────────────────

    def _build_coloring_group(self, cmaps) -> QGroupBox:
        col = QGroupBox("Coloring")
        cf = QFormLayout(col)
        cf.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.color_by_box = _expand(QComboBox())
        self.color_by_box.addItems(SYNTHETIC_COLOR_BY)
        self.color_by_box.currentTextChanged.connect(self.color_by_changed)
        cf.addRow("Color by", self.color_by_box)

        self.cmap_box = _expand(QComboBox())
        self.cmap_box.addItems(cmaps)
        self.cmap_box.currentTextChanged.connect(self.cmap_changed)
        cf.addRow("Colormap", self.cmap_box)

        self.center_cb = QCheckBox("Symmetric scale around 0")
        self.center_cb.setToolTip(
            "Sets the colorbar range to ±max(|values|) so the midpoint sits "
            "at zero.\nUseful for anomalies, vorticity, wind components, or "
            "any signed field."
        )
        self.center_cb.toggled.connect(self.center_zero_toggled)
        cf.addRow(self.center_cb)

        self.bar_cb = QCheckBox("Colorbar")
        self.bar_cb.setChecked(True)
        self.bar_cb.toggled.connect(self.colorbar_toggled)
        cf.addRow(self.bar_cb)

        # color-by defaults to "None" → cmap-related widgets start disabled
        self.center_cb.setEnabled(False)
        self.bar_cb.setEnabled(False)
        self.cmap_box.setEnabled(False)

        return col

    def _build_overlays_group(self) -> QGroupBox:
        ov = QGroupBox("Overlays")
        ol = QVBoxLayout(ov)
        ol.setSpacing(2)

        def toggle_color_row(label, signal_toggle, signal_color, signal_width,
                             default_width, checked=False):
            cb = QCheckBox(label)
            cb.setChecked(checked)
            cb.toggled.connect(signal_toggle)
            width = QDoubleSpinBox()
            width.setRange(0.2, 10.0)
            width.setSingleStep(0.2)
            width.setDecimals(1)
            width.setValue(default_width)
            width.setKeyboardTracking(False)
            width.setFixedWidth(54)
            width.valueChanged.connect(signal_width)
            btn = ColorButton("#ffffff")
            btn.color_changed.connect(signal_color)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addWidget(cb, stretch=1)
            row.addWidget(width)
            row.addWidget(btn)
            wrap = QWidget()
            wrap.setLayout(row)
            ol.addWidget(wrap)
            return cb, btn, width

        self.coast_cb, self.coast_btn, self.coast_width = toggle_color_row(
            "Coastlines", self.coastlines_toggled, self.coast_color_changed,
            self.coast_width_changed, default_width=1.2)
        self.grat_cb, self.grat_btn, self.grat_width = toggle_color_row(
            "Graticule (30°)", self.graticule_toggled, self.grat_color_changed,
            self.grat_width_changed, default_width=0.6)
        self.edges_cb, self.edge_btn, self.edge_width = toggle_color_row(
            "Cell edges", self.edges_toggled, self.edge_color_changed,
            self.edge_width_changed, default_width=0.6, checked=True)

        return ov

    def _build_animation_group(self) -> QGroupBox:
        anim = QGroupBox("Animation")
        al = QVBoxLayout(anim)
        self.spin_cb = QCheckBox("Auto-rotate")
        self.spin_cb.toggled.connect(self.autorotate_toggled)
        al.addWidget(self.spin_cb)

        if not self.with_time:
            return anim

        ROW_H = 22
        SLIDER_STYLE = (
            "QSlider::groove:horizontal {"
            "  height: 4px; background: #888; border-radius: 2px; }"
            "QSlider::handle:horizontal {"
            "  width: 12px; margin: -6px 0; background: #ccc;"
            "  border: 1px solid #555; border-radius: 6px; }"
        )

        self.time_row = QWidget()
        trow = QHBoxLayout(self.time_row)
        trow.setContentsMargins(0, 0, 0, 0)
        trow.setSpacing(6)
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(28, ROW_H)
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._on_play_toggled)
        trow.addWidget(self.play_btn)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.setFixedHeight(ROW_H)
        self.time_slider.setStyleSheet(SLIDER_STYLE)
        self.time_slider.valueChanged.connect(self.time_changed)
        trow.addWidget(self.time_slider, stretch=1)
        self.time_label = QLabel("—")
        self.time_label.setFixedSize(40, ROW_H)
        self.time_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
        trow.addWidget(self.time_label)
        al.addWidget(self.time_row)
        self.time_row.setVisible(False)

        self.speed_row = QWidget()
        srow = QHBoxLayout(self.speed_row)
        srow.setContentsMargins(0, 0, 0, 0)
        srow.setSpacing(6)
        speed_lbl = QLabel("Step")
        speed_lbl.setFixedHeight(ROW_H)
        srow.addWidget(speed_lbl)
        self.speed_box = QSpinBox()
        self.speed_box.setRange(50, 5000)
        self.speed_box.setSingleStep(50)
        self.speed_box.setValue(500)
        self.speed_box.setSuffix(" ms")
        self.speed_box.setFixedHeight(ROW_H)
        self.speed_box.setKeyboardTracking(False)
        self.speed_box.valueChanged.connect(self.play_speed_changed)
        srow.addWidget(self.speed_box, stretch=1)
        al.addWidget(self.speed_row)
        self.speed_row.setVisible(False)

        return anim

    def _build_export_group(self) -> QGroupBox:
        exp = QGroupBox("Export")
        elay = QHBoxLayout(exp)
        self.shot_btn = QPushButton("Save as PNG…")
        self.shot_btn.clicked.connect(self.screenshot_clicked)
        elay.addWidget(self.shot_btn)
        self.vec_btn = QPushButton("Save as SVG…")
        self.vec_btn.clicked.connect(self.vector_export_clicked)
        elay.addWidget(self.vec_btn)
        return exp

    # ── public setters ────────────────────────────────────────────────────

    def set_cmap(self, name):
        """Select *name* in the colormap combo box (no-op if not listed)."""
        i = self.cmap_box.findText(name)
        if i >= 0:
            self.cmap_box.setCurrentIndex(i)

    def set_color_by(self, name):
        """Select *name* in the color-by combo box (no-op if not listed)."""
        i = self.color_by_box.findText(name)
        if i >= 0:
            self.color_by_box.setCurrentIndex(i)

    def set_color_by_items(self, items):
        """Repopulate the color-by combo while preserving the current selection."""
        self.color_by_box.blockSignals(True)
        cur = self.color_by_box.currentText()
        self.color_by_box.clear()
        self.color_by_box.addItems(items)
        i = self.color_by_box.findText(cur)
        self.color_by_box.setCurrentIndex(i if i >= 0 else 0)
        self.color_by_box.blockSignals(False)

    def set_edge_color(self, hex_str):
        """Set the edge-color swatch."""
        self.edge_btn.set_color(hex_str)

    def set_coast_color(self, hex_str):
        """Set the coastline-color swatch."""
        self.coast_btn.set_color(hex_str)

    def set_grat_color(self, hex_str):
        """Set the graticule-color swatch."""
        self.grat_btn.set_color(hex_str)

    def set_time_steps(self, n_steps):
        """Configure the time slider for a time-varying field, or hide it."""
        if not self.with_time:
            return
        if n_steps and n_steps > 1:
            self.time_row.setVisible(True)
            self.speed_row.setVisible(True)
            self.time_slider.blockSignals(True)
            self.time_slider.setRange(0, n_steps - 1)
            self.time_slider.setValue(0)
            self.time_slider.blockSignals(False)
            self.time_label.setText(f"1/{n_steps}")
            self.play_btn.setChecked(False)
        else:
            self.time_row.setVisible(False)
            self.speed_row.setVisible(False)
            self.play_btn.setChecked(False)

    def set_time_label(self, idx, total):
        """Update the ``i/N`` label next to the time slider."""
        if self.with_time:
            self.time_label.setText(f"{idx+1}/{total}")

    def _on_play_toggled(self, on):
        self.play_btn.setText("⏸" if on else "▶")
        self.play_toggled.emit(on)


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
_DISPLAY_SIGNALS_TIME = ("time_changed", "play_toggled", "play_speed_changed")


class IcoTab(QWidget):
    """Ico tab: synthetic Goldberg grid params + display controls (no time)."""

    n_changed           = Signal(int)
    relax_iters_changed = Signal(int)
    zoom_changed        = Signal(float, float, float)   # factor, lon_deg, lat_deg

    def __init__(self, cmaps, parent=None):
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

    def set_cmap(self, name):
        """Forward to the inner display block."""
        self.display.set_cmap(name)

    def set_color_by(self, name):
        """Forward to the inner display block."""
        self.display.set_color_by(name)

    def set_color_by_items(self, items):
        """Forward to the inner display block."""
        self.display.set_color_by_items(items)

    def set_edge_color(self, hex_str):
        """Forward to the inner display block."""
        self.display.set_edge_color(hex_str)

    def set_coast_color(self, hex_str):
        """Forward to the inner display block."""
        self.display.set_coast_color(hex_str)

    def set_grat_color(self, hex_str):
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

    def set_zoom(self, factor, lon, lat):
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

    def __init__(self, cmaps, parent=None):
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

    def _toggle_lmdz_zoom(self):
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

    def _on_lmdz_spinbox_changed(self, _value):
        """Live re-apply when any LMDZ-zoom spinbox is edited while active."""
        if self._lmdz_zoom_active:
            self.lmdz_zoom_changed.emit(*self._current_lmdz_values())

    def set_lmdz_zoom(self, clon, clat, grossismx, grossismy,
                      dzoomx, dzoomy, taux, tauy):
        """Sync the LMDZ-zoom spinboxes and toggle state to the given values."""
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

    def set_cmap(self, name):
        """Forward to the inner display block."""
        self.display.set_cmap(name)

    def set_color_by(self, name):
        """Forward to the inner display block."""
        self.display.set_color_by(name)

    def set_color_by_items(self, items):
        """Forward to the inner display block."""
        self.display.set_color_by_items(items)

    def set_edge_color(self, hex_str):
        """Forward to the inner display block."""
        self.display.set_edge_color(hex_str)

    def set_coast_color(self, hex_str):
        """Forward to the inner display block."""
        self.display.set_coast_color(hex_str)

    def set_grat_color(self, hex_str):
        """Forward to the inner display block."""
        self.display.set_grat_color(hex_str)


class FileTab(QWidget):
    """File tab: Open/Unload + file summary + display controls (with time)."""

    open_file_clicked  = Signal()
    close_file_clicked = Signal()

    def __init__(self, cmaps, parent=None):
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

        # File summary block — hidden until a file is loaded.
        self.file_name_label = QLabel("")
        self.file_name_label.setStyleSheet("font-weight: bold; padding-top: 6px;")
        self.file_name_label.setVisible(False)
        v.addWidget(self.file_name_label)

        self.file_stats_label = QLabel("")
        self.file_stats_label.setStyleSheet("padding-top: 4px;")
        self.file_stats_label.setVisible(False)
        v.addWidget(self.file_stats_label)

        self.file_attrs_label = QLabel("")
        self.file_attrs_label.setStyleSheet("color: #888; font-size: 11px; padding-top: 4px;")
        self.file_attrs_label.setVisible(False)
        v.addWidget(self.file_attrs_label)

        # ── Display block (with time slider) ───────
        self.display = _DisplayBlock(cmaps, with_time=True)
        v.addWidget(self.display)
        _forward_signals(self, self.display,
                         _DISPLAY_SIGNALS_BASE + _DISPLAY_SIGNALS_TIME)

        v.addStretch(1)

    # ── proxy display methods ────────────────────────────────────────────

    def set_cmap(self, name):
        """Forward to the inner display block."""
        self.display.set_cmap(name)

    def set_color_by(self, name):
        """Forward to the inner display block."""
        self.display.set_color_by(name)

    def set_color_by_items(self, items):
        """Forward to the inner display block."""
        self.display.set_color_by_items(items)

    def set_edge_color(self, hex_str):
        """Forward to the inner display block."""
        self.display.set_edge_color(hex_str)

    def set_coast_color(self, hex_str):
        """Forward to the inner display block."""
        self.display.set_coast_color(hex_str)

    def set_grat_color(self, hex_str):
        """Forward to the inner display block."""
        self.display.set_grat_color(hex_str)

    def set_time_steps(self, n_steps):
        """Forward to the inner display block."""
        self.display.set_time_steps(n_steps)

    def set_time_label(self, idx, total):
        """Forward to the inner display block."""
        self.display.set_time_label(idx, total)

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
    ):
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

        self.file_name_label.setText(basename(path))
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
