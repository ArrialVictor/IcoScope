"""Qt side panel."""
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
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


def _expand(w):
    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    return w


class ColorButton(QPushButton):
    """A small square button showing a color swatch; click → QColorDialog."""

    color_changed = Signal(str)

    def __init__(self, initial="#ffffff", parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 18)
        self._color = QColor(initial)
        self._apply()
        self.clicked.connect(self._open_dialog)

    def _apply(self):
        self.setStyleSheet(
            f"background-color: {self._color.name()}; border: 1px solid #888;"
        )

    def _open_dialog(self):
        c = QColorDialog.getColor(self._color, self, "Pick color")
        if c.isValid():
            self.set_color(c.name())
            self.color_changed.emit(c.name())

    def set_color(self, hex_str):
        """Set the current swatch color from a ``#RRGGBB`` string."""
        self._color = QColor(hex_str)
        self._apply()


class ControlPanel(QWidget):
    """Right-side panel: every signal the main app connects to lives here.

    Top half is a tab strip — `Ico`, `LonLat`, `File` — that picks the mesh
    source and shows its source-specific parameters. Bottom half is shared
    display controls (Coloring, Overlays, Animation, Export) that apply to
    whatever is currently rendered.
    """

    # Coloring
    theme_changed       = Signal(str)
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

    # Time (for time-varying fields)
    time_changed        = Signal(int)
    play_toggled        = Signal(bool)
    play_speed_changed  = Signal(int)   # ms per step

    # Grid (Ico tab)
    n_changed           = Signal(int)
    relax_iters_changed = Signal(int)

    # File (File tab)
    open_file_clicked   = Signal()
    close_file_clicked  = Signal()

    # Synthetic zoom (Schmidt-style, DYNAMICO parameters; Ico tab)
    zoom_changed        = Signal(float, float, float)   # factor, lon_deg, lat_deg

    # Export
    screenshot_clicked     = Signal()
    vector_export_clicked  = Signal()

    def __init__(self, themes, cmaps, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        outer = QVBoxLayout(self)

        # ── Tab strip: mesh source ─────────────────
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_ico_tab(), "Ico")
        self.tabs.addTab(self._build_lonlat_tab(), "LonLat")
        self.tabs.addTab(self._build_file_tab(), "File")
        outer.addWidget(self.tabs)

        # ── Shared display section ─────────────────
        outer.addWidget(self._build_coloring_group(themes, cmaps))
        outer.addWidget(self._build_overlays_group())
        outer.addWidget(self._build_animation_group())
        outer.addWidget(self._build_export_group())

        outer.addStretch(1)

    # ── Tab builders ──────────────────────────────────────────────────────

    def _build_ico_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        v.setContentsMargins(6, 6, 6, 6)

        # Grid params
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

        # Synthetic zoom (Schmidt)
        zg = QGroupBox("Synthetic zoom (Schmidt)")
        zl = QFormLayout(zg)
        zl.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.zoom_factor_box = QDoubleSpinBox()
        self.zoom_factor_box.setRange(0.1, 10.0)
        self.zoom_factor_box.setSingleStep(0.1)
        self.zoom_factor_box.setDecimals(2)
        self.zoom_factor_box.setValue(1.0)
        self.zoom_factor_box.setKeyboardTracking(False)
        zl.addRow("Factor", self.zoom_factor_box)

        self.zoom_lon_box = QDoubleSpinBox()
        self.zoom_lon_box.setRange(-180.0, 180.0)
        self.zoom_lon_box.setSingleStep(5.0)
        self.zoom_lon_box.setDecimals(2)
        self.zoom_lon_box.setValue(0.0)
        self.zoom_lon_box.setKeyboardTracking(False)
        zl.addRow("Center lon (°)", self.zoom_lon_box)

        self.zoom_lat_box = QDoubleSpinBox()
        self.zoom_lat_box.setRange(-90.0, 90.0)
        self.zoom_lat_box.setSingleStep(5.0)
        self.zoom_lat_box.setDecimals(2)
        self.zoom_lat_box.setValue(45.0)
        self.zoom_lat_box.setKeyboardTracking(False)
        zl.addRow("Center lat (°)", self.zoom_lat_box)

        # Single Apply button so the three params are picked up together
        # (avoids three successive mesh rebuilds while the user is editing).
        self.zoom_apply_btn = QPushButton("Apply zoom")
        self.zoom_apply_btn.clicked.connect(self._emit_zoom)
        zl.addRow(self.zoom_apply_btn)

        v.addWidget(zg)
        v.addStretch(1)
        return tab

    def _build_lonlat_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        v.setContentsMargins(6, 6, 6, 6)
        placeholder = QLabel("LonLat mesh (coming soon)")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #888; font-style: italic; padding: 24px;")
        v.addWidget(placeholder)
        v.addStretch(1)
        return tab

    def _build_file_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        v.setContentsMargins(6, 6, 6, 6)

        # Open ↔ Unload toggle: same button, label/behavior switches with state.
        self.file_btn = QPushButton("Open NetCDF…")
        self._file_btn_mode = "open"
        self.file_btn.clicked.connect(self._on_file_btn_clicked)
        v.addWidget(self.file_btn)

        self.file_path_label = QLabel("")
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setStyleSheet("color: #888; font-size: 10px; padding-top: 4px;")
        v.addWidget(self.file_path_label)

        v.addStretch(1)
        return tab

    # ── Shared-section builders ───────────────────────────────────────────

    def _build_coloring_group(self, themes, cmaps) -> QGroupBox:
        col = QGroupBox("Coloring")
        cf = QFormLayout(col)
        cf.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.theme_box = _expand(QComboBox())
        self.theme_box.addItems(themes)
        self.theme_box.currentTextChanged.connect(self.theme_changed)
        cf.addRow("Theme", self.theme_box)

        self.color_by_box = _expand(QComboBox())
        # initial items; replaced at runtime via set_color_by_items()
        self.color_by_box.addItems(
            ["None", "Latitude", "Cell kind",
             "Mock temperature", "Realistic temperature"]
        )
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

        return col

    def _build_overlays_group(self) -> QGroupBox:
        ov = QGroupBox("Overlays")
        ol = QVBoxLayout(ov)
        ol.setSpacing(2)

        # builds a row:  [ ] <label>   [width]   <ColorButton>
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

        # Time row: play / slider / step label (hidden unless time-varying field)
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

        # Speed row: ms per step (modifiable spinbox)
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

    # ── Public setters (called by app.py) ────────────────────────────────

    def set_theme(self, name):
        """Select *name* in the theme combo box (no-op if not listed)."""
        i = self.theme_box.findText(name)
        if i >= 0:
            self.theme_box.setCurrentIndex(i)

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

    def set_file_path(self, path: str):
        """Show the loaded NetCDF path in the File tab (empty string clears it)."""
        self.file_path_label.setText(path or "")

    def disable_n(self, disabled=True):
        """Lock the Ico-tab controls and flip the file button to 'Unload'.

        Also locks the synthetic-zoom controls, which only apply to the
        Goldberg generator (loaded NetCDF files use their own geometry).
        """
        self.n_box.setEnabled(not disabled)
        self.relax_iters_box.setEnabled(not disabled)
        self.zoom_factor_box.setEnabled(not disabled)
        self.zoom_lon_box.setEnabled(not disabled)
        self.zoom_lat_box.setEnabled(not disabled)
        self.zoom_apply_btn.setEnabled(not disabled)
        # switch the file button's role
        self._file_btn_mode = "close" if disabled else "open"
        self.file_btn.setText("Unload NetCDF" if disabled else "Open NetCDF…")

    def _emit_zoom(self):
        """Emit ``zoom_changed`` with the current factor / lon / lat values."""
        self.zoom_changed.emit(
            self.zoom_factor_box.value(),
            self.zoom_lon_box.value(),
            self.zoom_lat_box.value(),
        )

    def set_zoom(self, factor, lon, lat):
        """Sync the zoom spinboxes to the given values without emitting."""
        for box, val in ((self.zoom_factor_box, factor),
                         (self.zoom_lon_box, lon),
                         (self.zoom_lat_box, lat)):
            box.blockSignals(True)
            box.setValue(float(val))
            box.blockSignals(False)

    def _on_file_btn_clicked(self):
        if self._file_btn_mode == "open":
            self.open_file_clicked.emit()
        else:
            self.close_file_clicked.emit()

    def set_time_steps(self, n_steps):
        """Configure the time slider for a time-varying field, or hide it."""
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
        self.time_label.setText(f"{idx+1}/{total}")

    def _on_play_toggled(self, on):
        self.play_btn.setText("⏸" if on else "▶")
        self.play_toggled.emit(on)
