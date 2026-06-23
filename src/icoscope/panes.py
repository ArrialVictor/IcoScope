"""Pane widgets for the multi-pane scaffold.

A :class:`Pane` wraps a single :class:`pyvistaqt.QtInteractor` with a 2-px
border that toggles between transparent (unselected) and an accent colour
(selected). :class:`PaneContainer` holds N panes in a splitter-based
layout (1 / 1×2 / 2×2 are configured in :meth:`PaneContainer.set_layout`).

Stage 2 of the multi-pane scaffold (see ``_design/multi-pane-layout-
scaffold.md``) introduces these widgets but the main window still hosts
exactly one pane; layout switching lands in stage 3.
"""
from __future__ import annotations

from pyvistaqt import QtInteractor
from qtpy.QtCore import Qt, Signal
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
    _STYLE_SELECTED = "Pane { border: 2px solid #ff8800; }"

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

    def set_selected(self, selected: bool) -> None:
        """Toggle the selection border."""
        self.setStyleSheet(
            self._STYLE_SELECTED if selected else self._STYLE_UNSELECTED
        )

    def mousePressEvent(self, ev):
        """Forward the click to the parent then announce who was clicked."""
        super().mousePressEvent(ev)
        self.clicked.emit(self.idx)


class PaneContainer(QWidget):
    """Holds N panes in a splitter layout (1 / 1×2 / 2×2).

    The full pane list is created lazily on first :meth:`set_layout` call
    and grown as needed — switching from 1 to 2×2 instantiates new panes;
    switching back to 1 hides slots 1..3 but keeps them in memory so their
    state survives.

    Stage 2 only ever populates one pane (single-pane). :meth:`set_layout`
    handles the multi-pane wiring but the main window doesn't call it with
    n>1 until stage 3 of the scaffold PR.
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
        if rows == 1:
            # 1 or 1×2 — single horizontal splitter
            self._splitter = QSplitter(Qt.Horizontal)
            for i in range(n_panes):
                self._splitter.addWidget(self._panes[i])
        else:
            # 2×2 — vertical splitter of two horizontal splitters
            self._splitter = QSplitter(Qt.Vertical)
            for row in range(rows):
                row_split = QSplitter(Qt.Horizontal)
                for col in range(cols):
                    idx = row * cols + col
                    row_split.addWidget(self._panes[idx])
                self._splitter.addWidget(row_split)
        for i in range(n_panes):
            self._panes[i].setVisible(True)
        self._outer.addWidget(self._splitter)
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
