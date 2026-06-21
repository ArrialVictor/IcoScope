"""Qt side panel — thin container that hosts the per-tab widgets.

Each tab is fully self-contained (its own display controls, its own state);
this module just composes them inside an adaptive tab strip. The reusable
``ColorButton`` and ``_expand`` helper live here so ``tabs.py`` can import them
without circularity.
"""
from qtpy.QtCore import Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QColorDialog,
    QPushButton,
    QSizePolicy,
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
    """Right-side panel: a tab strip of three independent tabs.

    Each tab (``ico_tab``, ``lonlat_tab``, ``file_tab``) owns its own display
    controls and signals. The main window connects per-tab — there is no
    shared section below the tabs.
    """

    def __init__(self, cmaps, parent=None):
        super().__init__(parent)
        # late import: tabs.py imports ColorButton from this module
        from .tabs import FileTab, IcoTab, LonLatTab, _AdaptiveTabWidget, _ScrollArea

        self.setFixedWidth(320)
        outer = QVBoxLayout(self)

        self.tabs = _AdaptiveTabWidget()
        self.ico_tab = IcoTab(cmaps)
        self.lonlat_tab = LonLatTab(cmaps)
        self.file_tab = FileTab(cmaps)
        # Wrap each tab in a scroll area — LonLat in particular now has 10+
        # zoom params plus the display block, easily exceeding the window
        # height. Adaptive sizing still works (the scroll area reports the
        # wrapped widget's sizeHint) so short tabs don't grow.
        self.tabs.addTab(_ScrollArea(self.ico_tab), "Ico")
        self.tabs.addTab(_ScrollArea(self.lonlat_tab), "LonLat")
        self.tabs.addTab(_ScrollArea(self.file_tab), "File")
        outer.addWidget(self.tabs)

        outer.addStretch(1)
