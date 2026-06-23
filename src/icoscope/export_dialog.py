"""Unified export dialog — PNG composite, PNG per-pane, SVG per-pane."""
from __future__ import annotations

import os
from dataclasses import dataclass

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Human label → PyVista scale multiplier for PNG renders.
QUALITY_PRESETS = [
    ("Standard (1× — window resolution)",   1),
    ("High (2× — for slides / web)",        2),
    ("Ultra (4× — for printing / posters)", 4),
]


@dataclass
class ExportChoice:
    """User's selections from the dialog, returned to the export handler."""

    base_path: str
    scale: int
    transparent: bool
    png_composite: bool
    png_per_pane: bool
    svg_per_pane: bool


class ExportDialog(QDialog):
    """Pick output formats + quality in one window.

    When ``n_panes == 1`` the "composite" vs "per-pane" distinction
    collapses — only a single "PNG" / "SVG" pair is offered. When
    ``n_panes > 1`` the user can mix composite + per-pane PNGs and a
    set of per-pane SVGs; the dialog shows a live preview of which files
    will land on disk.
    """

    def __init__(self, parent, *, default_base: str, n_panes: int, defaults):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(500)
        self.n_panes = n_panes

        # ── base path picker ──
        self.path_edit = QLineEdit(os.path.join(os.getcwd(), default_base))
        self.path_edit.textChanged.connect(self._refresh_counts)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, stretch=1)
        path_row.addWidget(browse)

        # ── output checkboxes ──
        # png_per_pane_cb only exists in multi-pane mode; result() returns
        # False for it in single-pane so the writer's `if png_per_pane` branch
        # never fires on a layout that has no individual panes to write.
        self.png_per_pane_cb: QCheckBox | None = None
        if n_panes > 1:
            self.png_composite_cb = QCheckBox("PNG composite (all panes, one image)")
            self.png_per_pane_cb = QCheckBox("PNG individual panes (one per pane)")
            self.svg_per_pane_cb = QCheckBox("SVG individual panes (one per pane)")
            self.png_composite_cb.setChecked(defaults.png_composite)
            self.png_per_pane_cb.setChecked(defaults.png_per_pane)
            self.svg_per_pane_cb.setChecked(defaults.svg_per_pane)
        else:
            self.png_composite_cb = QCheckBox("PNG")
            self.svg_per_pane_cb = QCheckBox("SVG")
            self.png_composite_cb.setChecked(defaults.png_composite)
            self.svg_per_pane_cb.setChecked(defaults.svg_per_pane)

        for cb in (self.png_composite_cb, self.png_per_pane_cb, self.svg_per_pane_cb):
            if cb is not None:
                cb.toggled.connect(self._refresh_counts)

        # Per-row tuples of (checkbox, count_label, file-list closure). Each
        # row sits in its own HBox so the count label hugs the right edge —
        # like grouped settings rows in a typical preferences dialog.
        outputs_box = QVBoxLayout()
        outputs_box.setSpacing(4)
        self._rows = []

        def add_row(cb: QCheckBox, files_fn):
            count = QLabel()
            # Scope the rule to QLabel — bare "color: gray" cascades into the
            # widget's QToolTip and makes tooltip text unreadable on dark themes.
            count.setStyleSheet("QLabel { color: gray; }")
            count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(cb, stretch=1)
            row.addWidget(count)
            outputs_box.addLayout(row)
            self._rows.append((cb, count, files_fn))

        if n_panes > 1:
            add_row(self.png_composite_cb,
                    lambda base: [f"{os.path.basename(base)}.png"])
            add_row(self.png_per_pane_cb,
                    lambda base: [f"{os.path.basename(base)}_pane{i+1}.png"
                                  for i in range(self.n_panes)])
            add_row(self.svg_per_pane_cb,
                    lambda base: [f"{os.path.basename(base)}_pane{i+1}.svg"
                                  for i in range(self.n_panes)])
        else:
            add_row(self.png_composite_cb, lambda base: [f"{base}.png"])
            add_row(self.svg_per_pane_cb, lambda base: [f"{base}.svg"])

        outputs_wrap = QWidget()
        outputs_wrap.setLayout(outputs_box)

        # ── PNG quality + transparency ──
        self.quality_box = QComboBox()
        for label, _ in QUALITY_PRESETS:
            self.quality_box.addItem(label)
        scale_index = next(
            (i for i, (_, s) in enumerate(QUALITY_PRESETS) if s == defaults.scale),
            0,
        )
        self.quality_box.setCurrentIndex(scale_index)

        self.transparent_cb = QCheckBox("Transparent background (PNG only)")
        self.transparent_cb.setChecked(defaults.transparent)
        self.transparent_cb.setToolTip(
            "Save PNG with no background fill. Useful for compositing into slides."
        )

        # ── buttons ──
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save")

        # ── layout ──
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignVCenter | Qt.AlignRight)
        form.setFormAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(10)
        path_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("File:", _wrap(path_row))
        form.addRow("Outputs:", outputs_wrap)
        form.addRow("PNG quality:", self.quality_box)
        form.addRow("", self.transparent_cb)

        outer = QVBoxLayout(self)
        outer.addLayout(form)
        outer.addWidget(self.buttons)

        self._refresh_counts()

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Choose base file name", self.path_edit.text(),
            "All files (*)",
        )
        if path:
            # Strip a trailing .png / .svg if the user picked one — the dialog
            # treats the field as a base name, not a final file name.
            root, ext = os.path.splitext(path)
            if ext.lower() in (".png", ".svg"):
                path = root
            self.path_edit.setText(path)

    def _refresh_counts(self):
        base = self.path_edit.text().strip()
        anything_checked = False
        for cb, count_label, files_fn in self._rows:
            files = files_fn(base) if base else []
            n = len(files)
            count_label.setText(f"{n} file" if n == 1 else f"{n} files")
            # Per-row tooltip on both checkbox and count label, so hovering
            # either reveals exactly which files that row will write.
            tip = "\n".join(files) if files else ""
            cb.setToolTip(tip)
            count_label.setToolTip(tip)
            if cb.isChecked():
                anything_checked = True
        save_btn = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        save_btn.setEnabled(anything_checked and bool(base))

    def result(self) -> ExportChoice:
        """Snapshot the user's selections into an :class:`ExportChoice`."""
        base = self.path_edit.text().strip()
        # Trim any redundant extension the user typed in — the writer adds
        # its own per-format suffix.
        root, ext = os.path.splitext(base)
        if ext.lower() in (".png", ".svg"):
            base = root
        scale = QUALITY_PRESETS[self.quality_box.currentIndex()][1]
        return ExportChoice(
            base_path=base,
            scale=scale,
            transparent=self.transparent_cb.isChecked(),
            png_composite=self.png_composite_cb.isChecked(),
            png_per_pane=(self.png_per_pane_cb.isChecked()
                          if self.png_per_pane_cb is not None else False),
            svg_per_pane=self.svg_per_pane_cb.isChecked(),
        )


def _wrap(layout):
    w = QWidget()
    w.setLayout(layout)
    return w
