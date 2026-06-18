"""Custom save dialog for PNG screenshots: path + quality + transparency in one window."""
from __future__ import annotations

import os

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

# Mapping from human label → PyVista scale multiplier.
QUALITY_PRESETS = [
    ("Standard (1× — window resolution)",            1),
    ("High (2× — for slides / web)",                 2),
    ("Ultra (4× — for printing / posters)",          4),
]


class PngExportDialog(QDialog):
    """Single window combining file path, quality preset, and transparency."""

    def __init__(self, parent, default_filename: str, default_transparent: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Save PNG screenshot")
        self.setMinimumWidth(460)

        # ── path picker ──
        self.path_edit = QLineEdit(os.path.join(os.getcwd(), default_filename))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(browse)

        # ── quality ──
        self.quality_box = QComboBox()
        for label, _scale in QUALITY_PRESETS:
            self.quality_box.addItem(label)
        self.quality_box.setCurrentIndex(0)

        # ── transparency ──
        self.transparent_cb = QCheckBox("Transparent background")
        self.transparent_cb.setChecked(default_transparent)
        self.transparent_cb.setToolTip(
            "Save with no background fill. Useful for compositing into slides or papers."
        )

        # ── buttons ──
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        # rename Save button so it reads naturally
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save PNG")

        # ── layout ──
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignVCenter | Qt.AlignRight)
        form.setFormAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(10)
        # remove inner margins on the path row so its baseline matches the label
        path_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("File:", _wrap(path_row))
        form.addRow("Quality:", self.quality_box)
        form.addRow("", self.transparent_cb)

        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(buttons)

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PNG", self.path_edit.text(), "PNG image (*.png)"
        )
        if path:
            self.path_edit.setText(path)

    def result(self) -> tuple[str, int, bool]:
        """Return (path, scale_multiplier, transparent_background)."""
        path = self.path_edit.text().strip()
        scale = QUALITY_PRESETS[self.quality_box.currentIndex()][1]
        transparent = self.transparent_cb.isChecked()
        return path, scale, transparent


def _wrap(layout):
    from qtpy.QtWidgets import QWidget
    w = QWidget()
    w.setLayout(layout)
    return w
