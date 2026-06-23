"""Qt side panel — thin container that hosts the per-tab widgets.

Each tab is fully self-contained (its own display controls, its own state);
this module just composes them inside an adaptive tab strip. Reusable
helper widgets (``ColorButton``, ``_expand``) live in ``widgets.py``.
"""
from qtpy.QtWidgets import QVBoxLayout, QWidget

from .tabs import FileTab, IcoTab, LonLatTab, _AdaptiveTabWidget, _ScrollArea


class ControlPanel(QWidget):
    """Right-side panel: a tab strip of three independent tabs.

    Each tab (``ico_tab``, ``lonlat_tab``, ``file_tab``) owns its own display
    controls and signals. The main window connects per-tab — there is no
    shared section below the tabs.
    """

    DEFAULT_WIDTH = 320
    MIN_WIDTH = 220

    def __init__(self, cmaps: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        # Was setFixedWidth(320); now resizable via the splitter in MainWindow.
        # Keep a sensible minimum so the controls don't collapse to unreadable.
        self.setMinimumWidth(self.MIN_WIDTH)
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
        # Tabs widget claims all vertical space — without this, the active
        # tab would be sized to its sizeHint with empty room below, leaving
        # no room to scroll long tabs (File with vertical fields, LonLat
        # with its zoom params) inside the existing scroll-area wrapping.
        outer.addWidget(self.tabs, stretch=1)
