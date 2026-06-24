"""Pane widgets for the multi-pane viewport.

A :class:`Pane` wraps a single :class:`pyvistaqt.QtInteractor` with a 2-px
border that toggles between transparent (unselected) and an accent colour
(selected), plus a VTK 2-D text-actor banner used to surface "showing
nearest" cursor-out-of-range messages. :class:`PaneContainer` holds up to
four panes in a splitter-based layout (1 / 1×2 / 2×2 are configured in
:meth:`PaneContainer.set_layout`) and is the central viewport that the
main window mounts in its horizontal splitter.
"""
from __future__ import annotations

from pyvistaqt import QtInteractor
from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtWidgets import QFrame, QSplitter, QVBoxLayout, QWidget


class Pane(QFrame):
    """A single sphere viewport — `QtInteractor` wrapped in a selection frame.

    The frame's stylesheet swaps between transparent and an accent border
    when :meth:`set_selected` toggles. Clicking the pane emits
    :attr:`clicked` with its index so the main window can update the
    selection model.
    """

    clicked = Signal(int)

    _STYLE_UNSELECTED = "Pane { border: 2px solid transparent; }"
    # Soft muted-amber rather than full safety-cone orange — visible enough
    # to identify the focused pane without dominating the viewport.
    _STYLE_SELECTED = "Pane { border: 2px solid #d4a060; }"

    def __init__(self, idx: int, parent: QWidget | None = None):
        super().__init__(parent)
        # Object name + targeted selector means the border only applies to
        # the Pane frame itself, not to every child widget (QFrame's default
        # stylesheet behaviour leaks otherwise).
        self.setObjectName("Pane")
        self.idx = idx
        self.setStyleSheet(self._STYLE_UNSELECTED)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        # Selection is most naturally triggered by clicking the sphere itself,
        # but the VTK widget captures its own mouse events for rotate / pan /
        # zoom — they never reach our mousePressEvent. Install an event
        # filter so we also see those clicks (without consuming them — VTK
        # still gets the event so the camera gesture works as before).
        self.plotter.interactor.installEventFilter(self)
        # Banner text actor — surfaces "Showing <nearest> (cursor at <cursor>)"
        # when the master time cursor falls outside this pane's field's axis
        # range. Drawn as a VTK 2D text actor (rendered inside the scene), not
        # a Qt QLabel: QLabels parented to QtInteractor don't reliably
        # composite over the native VTK render on macOS. ``None`` means no
        # banner is currently shown.
        self._banner_text: str | None = None

    def set_selected(self, selected: bool) -> None:
        """Toggle the selection border."""
        self.setStyleSheet(
            self._STYLE_SELECTED if selected else self._STYLE_UNSELECTED
        )

    def set_banner(self, text: str | None) -> None:
        """Show ``text`` in the bottom-left overlay, or hide the banner.

        Implemented as a VTK 2D text actor (``plotter.add_text`` /
        ``remove_actor``) so it reliably composites over the rendered
        sphere on every platform. Idempotent: re-passing the same text
        is a no-op (avoids actor-churn flicker during fast scrubs).
        """
        if text == self._banner_text:
            return
        self.plotter.remove_actor("banner", reset_camera=False, render=False)
        self._banner_text = text
        if text:
            # Top-left, just inside the pane. PyVista's ``position`` accepts
            # either a named slot ("upper_left", etc.) or pixel coords; named
            # slot keeps the banner anchored if the pane resizes.
            self.plotter.add_text(
                text, name="banner", position="upper_left",
                font_size=10, color="white",
                shadow=True,
            )
        self.plotter.render()

    @property
    def banner_visible(self) -> bool:
        """Whether a banner is currently displayed (for tests)."""
        return self._banner_text is not None

    @property
    def banner_text(self) -> str:
        """Current banner text (empty string when hidden), for tests."""
        return self._banner_text or ""

    def mousePressEvent(self, ev):
        """Forward the click to the parent then announce who was clicked."""
        super().mousePressEvent(ev)
        self.clicked.emit(self.idx)

    def eventFilter(self, obj, event):
        """Catch mouse-press events on the VTK widget to drive selection."""
        if (obj is self.plotter.interactor
                and event.type() == QEvent.MouseButtonPress):
            self.clicked.emit(self.idx)
        return False   # don't consume — let VTK handle the click too


class PaneContainer(QWidget):
    """Holds up to four panes in a splitter layout (1 / 1×2 / 2×2).

    All four pane widgets are instantiated upfront and reused across
    layout changes; :meth:`set_layout` toggles visibility and rebuilds
    the inner splitter geometry but never destroys a pane, so each
    pane's plotter, camera, and per-pane state (banner text, picked
    cell, etc.) survives a layout switch. The View → Pane layout menu
    drives the active layout.
    """

    pane_clicked = Signal(int)

    # n_panes → (rows, cols) for the splitter layout
    _LAYOUTS = {1: (1, 1), 2: (1, 2), 4: (2, 2)}
    MAX_PANES = 4

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._panes: list[Pane] = []
        self._n_visible = 0
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._splitter: QSplitter | None = None
        # All panes are created upfront — keeps state alive across layout
        # changes and avoids reinstantiating heavy QtInteractor widgets on
        # every switch. Layout decides which ones are visible.
        for i in range(self.MAX_PANES):
            pane = Pane(i)
            pane.clicked.connect(self.pane_clicked)
            pane.setVisible(False)
            self._panes.append(pane)
        self.set_layout(1)

    def set_layout(self, n_panes: int) -> None:
        """Reconfigure the container to show ``n_panes`` (must be 1, 2, or 4).

        Panes beyond the visible count stay alive but hidden — their
        state survives so switching back to a wider layout restores them.
        """
        if n_panes not in self._LAYOUTS:
            raise ValueError(
                f"n_panes must be one of {sorted(self._LAYOUTS)}; got {n_panes}"
            )
        # Reparent every pane out of any existing splitter so we can rebuild.
        for pane in self._panes:
            pane.setParent(self)
            pane.setVisible(False)
        if self._splitter is not None:
            self._outer.removeWidget(self._splitter)
            self._splitter.deleteLater()
            self._splitter = None

        rows, cols = self._LAYOUTS[n_panes]
        # Slim handle so the ridge between viewports doesn't dominate the
        # comparison. 2 px feels right at typical screen DPI; the user can
        # still grab and drag to rebalance.
        handle_w = 2
        if rows == 1:
            # 1 or 1×2 — single horizontal splitter
            self._splitter = QSplitter(Qt.Horizontal)
            self._splitter.setHandleWidth(handle_w)
            for i in range(n_panes):
                self._splitter.addWidget(self._panes[i])
        else:
            # 2×2 — vertical splitter of two horizontal splitters
            self._splitter = QSplitter(Qt.Vertical)
            self._splitter.setHandleWidth(handle_w)
            for row in range(rows):
                row_split = QSplitter(Qt.Horizontal)
                row_split.setHandleWidth(handle_w)
                for col in range(cols):
                    idx = row * cols + col
                    row_split.addWidget(self._panes[idx])
                self._splitter.addWidget(row_split)
        for i in range(n_panes):
            self._panes[i].setVisible(True)
        self._outer.addWidget(self._splitter)
        # QSplitter distributes space proportionally to children's sizeHints
        # by default, which can leave panes unequal at startup before the
        # first interaction. Force an equal split for predictable layout.
        if rows == 1:
            self._splitter.setSizes([1000] * n_panes)
        else:
            self._splitter.setSizes([1000] * rows)
            for child_idx in range(self._splitter.count()):
                inner = self._splitter.widget(child_idx)
                if isinstance(inner, QSplitter):
                    inner.setSizes([1000] * cols)
        self._n_visible = n_panes

    @property
    def n_visible(self) -> int:
        """How many panes are currently in the active layout (1, 2, or 4)."""
        return self._n_visible

    def pane(self, idx: int) -> Pane:
        """Return the ``idx``-th pane (visible or hidden)."""
        return self._panes[idx]

    def panes(self) -> list[Pane]:
        """Return all panes (length :attr:`MAX_PANES`)."""
        return list(self._panes)

    def visible_panes(self) -> list[Pane]:
        """Return only the panes in the current layout."""
        return self._panes[: self._n_visible]

    def set_selected(self, idx: int | None) -> None:
        """Mark one pane as selected (or pass ``None`` to deselect all)."""
        for i, pane in enumerate(self._panes):
            pane.set_selected(i == idx)
