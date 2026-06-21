"""Reusable Qt widgets shared across the side panel modules.

Lives in its own module so ``controls.py`` and ``tabs.py`` can both import
from here without forming a cycle.
"""
from qtpy.QtCore import Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QColorDialog, QPushButton, QSizePolicy, QWidget


def _expand(w: QWidget) -> QWidget:
    """Set ``w``'s horizontal size policy to Expanding (no vertical change)."""
    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    return w


class ColorButton(QPushButton):
    """A small square button showing a color swatch; click → QColorDialog."""

    color_changed = Signal(str)

    def __init__(self, initial: str = "#ffffff", parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(24, 18)
        self._color = QColor(initial)
        self._apply()
        self.clicked.connect(self._open_dialog)

    def _apply(self) -> None:
        self.setStyleSheet(
            f"background-color: {self._color.name()}; border: 1px solid #888;"
        )

    def _open_dialog(self) -> None:
        c = QColorDialog.getColor(self._color, self, "Pick color")
        if c.isValid():
            self.set_color(c.name())
            self.color_changed.emit(c.name())

    def set_color(self, hex_str: str) -> None:
        """Set the current swatch color from a ``#RRGGBB`` string."""
        self._color = QColor(hex_str)
        self._apply()
