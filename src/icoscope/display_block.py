"""Standard display controls shared between the IcoScope tabs.

``_DisplayBlock`` builds the Coloring + Overlays + Animation + Export
sections that every tab (Ico, LonLat, File) exposes. The ``with_time``
flag controls whether Animation includes the time slider, play button,
and speed spinbox — only the File tab passes ``with_time=True``.

Kept private (leading underscore) because tabs are responsible for
forwarding its signals and exposing its setters; consumers outside
:mod:`icoscope.tabs` should not touch a display block directly.
"""
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
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .widgets import ColorButton, _expand

# Color-by options shown for a synthetic mesh (no NetCDF loaded). The File
# tab overrides this with the loaded file's actual field names.
SYNTHETIC_COLOR_BY = [
    "None", "Latitude", "Cell kind", "Mock temperature", "Realistic temperature",
]


class _DisplayBlock(QWidget):
    """Standard display controls: Coloring + Overlays + Animation + Export.

    The ``with_time`` flag controls whether the Animation group includes the
    time slider, play button, and speed spinbox (only meaningful for the
    File tab where a time-varying field can be active).
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

    def __init__(self, cmaps: list[str], with_time: bool,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.with_time = with_time
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._build_coloring_group(cmaps))
        v.addWidget(self._build_overlays_group())
        v.addWidget(self._build_animation_group())
        v.addWidget(self._build_export_group())

    # ── group builders ────────────────────────────────────────────────────

    def _build_coloring_group(self, cmaps: list[str]) -> QGroupBox:
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

    def set_cmap(self, name: str) -> None:
        """Select *name* in the colormap combo box (no-op if not listed)."""
        i = self.cmap_box.findText(name)
        if i >= 0:
            self.cmap_box.setCurrentIndex(i)

    def set_color_by(self, name: str) -> None:
        """Select *name* in the color-by combo box (no-op if not listed)."""
        i = self.color_by_box.findText(name)
        if i >= 0:
            self.color_by_box.blockSignals(True)
            self.color_by_box.setCurrentIndex(i)
            self.color_by_box.blockSignals(False)

    def set_color_by_items(self, items: list[str]) -> None:
        """Repopulate the color-by combo while preserving the current selection."""
        self.color_by_box.blockSignals(True)
        cur = self.color_by_box.currentText()
        self.color_by_box.clear()
        self.color_by_box.addItems(items)
        i = self.color_by_box.findText(cur)
        self.color_by_box.setCurrentIndex(i if i >= 0 else 0)
        self.color_by_box.blockSignals(False)

    def set_edge_color(self, hex_str: str) -> None:
        """Set the edge-color swatch."""
        self.edge_btn.set_color(hex_str)

    def set_coast_color(self, hex_str: str) -> None:
        """Set the coastline-color swatch."""
        self.coast_btn.set_color(hex_str)

    def set_grat_color(self, hex_str: str) -> None:
        """Set the graticule-color swatch."""
        self.grat_btn.set_color(hex_str)

    def set_time_steps(self, n_steps: int) -> None:
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

    def set_time_label(self, idx: int, total: int) -> None:
        """Update the ``i/N`` label next to the time slider."""
        if self.with_time:
            self.time_label.setText(f"{idx+1}/{total}")

    def _on_play_toggled(self, on):
        self.play_btn.setText("⏸" if on else "▶")
        self.play_toggled.emit(on)
