"""Bottom timeline strip — per-pane sample tracks with a shared cursor.

Multi-pane comparison view, Phase 1: replaces (when active) the
side-panel time slider with N horizontally-stacked tracks, one per
visible pane. Each track paints its time samples as dots on a single
shared horizontal axis (the union of every visible pane's datetime
domain), so a file mixing a daily field with a monthly field shows the
daily samples clustered in a sub-range while the monthly ones spread
across — the axis mismatch is visually obvious. A single vertical
cursor crosses every track; clicking or dragging on any track moves
the cursor and emits :attr:`TimelineStrip.cursor_changed` with the
chosen datetime, which the window resolves to every pane via the
existing master-cursor sync.

Per-track value displays, per-track time locks, and pane-selection
via track click are deferred to Phase 2.
"""
from __future__ import annotations

from collections.abc import Sequence

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QFontMetrics, QPainter, QPen
from qtpy.QtWidgets import QPushButton, QSizePolicy, QVBoxLayout, QWidget

# Track height and total-strip height tuned to land under the viewports
# without dominating the layout. Update both together if the row height
# grows; the strip's setFixedHeight uses N × TRACK_HEIGHT + padding.
TRACK_HEIGHT = 28
# Track region widths, left to right:
#   [lock button] [label] [plot area with dots + cursor] [value text]
LOCK_WIDTH = 72       # toggle button on the very left — fits "Unlock" text
LABEL_WIDTH = 96      # field-name label (clickable → select pane)
VALUE_WIDTH = 96      # right-hand current-pick value
HORIZONTAL_PADDING = 12


class Track(QWidget):
    """A single per-pane track: dots for samples + a vertical cursor line.

    Tracks share the same horizontal datetime range (set by the parent
    :class:`TimelineStrip`) so the cursor x-coordinate is consistent
    across every track in the strip.

    Parameters
    ----------
    label
        Short string shown left of the track (typically the field name).

    The mouse press is split by region:

    - Click on the lock area (very left) → :attr:`lock_clicked`.
    - Click on the label area → :attr:`label_clicked`.
    - Click on the plot area → :attr:`clicked_at` (drag continues this).

    Signals
    -------
    clicked_at
        Emitted with a fraction in ``[0, 1]`` mapped back to the shared
        datetime domain by the parent. The parent decides what to do
        with it (currently: set the master cursor).
    label_clicked
        Emitted with no payload when the user clicks the field-name
        label — the parent reports it upward as "select this pane".
    lock_clicked
        Emitted with no payload when the user clicks the lock area —
        the parent toggles the pane's time_locked state.
    """

    clicked_at = Signal(float)
    label_clicked = Signal()
    lock_clicked = Signal()

    def __init__(self, label: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._label = label
        self._times: list = []          # sample datetimes for this pane
        self._domain_t0 = None          # parent-supplied range
        self._domain_t1 = None
        self._cursor_t = None
        self._value_text = ""           # current-pick value, "" when no pick
        self._locked = False
        self.setFixedHeight(TRACK_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(False)
        # Lock as a real QPushButton with imperative text ("Lock" when
        # unlocked = "click to lock"; "Unlock" when locked = "click to
        # unlock"). Emojis didn't render legibly at this size.
        self._lock_btn = QPushButton("Lock", self)
        self._lock_btn.setCheckable(True)
        self._lock_btn.setFixedSize(LOCK_WIDTH - 4, TRACK_HEIGHT - 6)
        self._lock_btn.move(2, 3)
        self._lock_btn.setToolTip(
            "Lock this pane to its current time — the master cursor will "
            "skip it on subsequent scrubs."
        )
        # Don't toggle state on press — the window decides whether to
        # honour the toggle (and pushes the new state back via
        # set_locked). Block the default toggle so visual state is
        # always authoritatively set by set_locked.
        self._lock_btn.clicked.connect(self._on_lock_button_clicked)

    def set_samples(self, times: Sequence) -> None:
        """Replace the dot positions for this track."""
        self._times = list(times)
        self.update()

    def set_domain(self, t0, t1) -> None:
        """Set the shared horizontal datetime range (called by parent)."""
        self._domain_t0 = t0
        self._domain_t1 = t1
        self.update()

    def set_cursor(self, t) -> None:
        """Move the vertical cursor line to datetime ``t`` (or hide if None)."""
        self._cursor_t = t
        self.update()

    def set_label(self, label: str) -> None:
        """Update the left-hand label text."""
        self._label = label
        self.update()

    def set_value(self, text: str) -> None:
        """Update the right-hand value text (empty string hides it)."""
        self._value_text = text or ""
        self.update()

    def set_locked(self, locked: bool) -> None:
        """Set the lock visual + state (caller decides the semantics)."""
        self._locked = bool(locked)
        self._lock_btn.blockSignals(True)
        self._lock_btn.setChecked(self._locked)
        self._lock_btn.setText("Unlock" if self._locked else "Lock")
        self._lock_btn.blockSignals(False)
        self.update()

    def _on_lock_button_clicked(self) -> None:
        """Fire ``lock_clicked`` and revert the button's auto-toggle.

        The window is the only authority on the visual state — it
        decides whether to honour the toggle and pushes the new state
        back via :meth:`set_locked`.
        """
        self._lock_btn.blockSignals(True)
        self._lock_btn.setChecked(self._locked)
        self._lock_btn.blockSignals(False)
        self.lock_clicked.emit()

    @property
    def locked(self) -> bool:
        """Current lock state (for tests)."""
        return self._locked

    @property
    def value_text(self) -> str:
        """Current value-column text (for tests)."""
        return self._value_text

    def _plot_x0(self) -> int:
        """Left edge of the plot area in widget pixels."""
        return LOCK_WIDTH + LABEL_WIDTH + HORIZONTAL_PADDING

    def _plot_w(self) -> int:
        """Width of the plot area (between left-pad and value column)."""
        return (self.width() - LOCK_WIDTH - LABEL_WIDTH - VALUE_WIDTH
                - 2 * HORIZONTAL_PADDING)

    def _x_for(self, t) -> float | None:
        """Map a datetime ``t`` to an x pixel within the plot area."""
        if (t is None or self._domain_t0 is None
                or self._domain_t1 is None):
            return None
        total = self._domain_t1 - self._domain_t0
        # Some datetime types (cftime) return timedelta-like; coerce to a
        # float ratio via total_seconds when available, otherwise rely on
        # the type supporting / by another delta of the same kind.
        try:
            ratio = (t - self._domain_t0).total_seconds() / total.total_seconds()
        except AttributeError:
            ratio = (t - self._domain_t0) / total
        ratio = max(0.0, min(1.0, float(ratio)))
        return self._plot_x0() + ratio * self._plot_w()

    def paintEvent(self, _event) -> None:
        """Custom paint: label, baseline, sample dots, cursor, value.

        The lock control is a real QPushButton child widget (positioned
        in ``__init__``), not painted here.
        """
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        fm = QFontMetrics(p.font())

        # Label, vertically centred and elided to fit. Slightly brighter when
        # the pane is unlocked to telegraph that the label is clickable.
        p.setPen(QColor("#cccccc"))
        elided = fm.elidedText(self._label, Qt.ElideRight, LABEL_WIDTH - 8)
        p.drawText(LOCK_WIDTH + 4, 0, LABEL_WIDTH - 8, self.height(),
                   Qt.AlignVCenter | Qt.AlignRight, elided)

        # Baseline — thin horizontal line across the plot area.
        plot_x0 = self._plot_x0()
        plot_x1 = plot_x0 + self._plot_w()
        baseline_y = self.height() // 2
        p.setPen(QPen(QColor("#555555"), 1))
        p.drawLine(plot_x0, baseline_y, plot_x1, baseline_y)

        # Sample dots — only when we have a domain (the strip always sets
        # one before showing). Skip silently otherwise.
        if self._domain_t0 is not None and self._times:
            p.setBrush(QColor("#d4a060"))
            p.setPen(Qt.NoPen)
            for t in self._times:
                x = self._x_for(t)
                if x is None:
                    continue
                p.drawEllipse(int(x) - 2, baseline_y - 2, 4, 4)

        # Cursor — vertical line at the cursor datetime.
        if self._cursor_t is not None:
            x = self._x_for(self._cursor_t)
            if x is not None:
                p.setPen(QPen(QColor("#ff7a3a"), 2))
                p.drawLine(int(x), 2, int(x), self.height() - 2)

        # Right-hand current-pick value (empty when no pick).
        if self._value_text:
            value_x = self.width() - VALUE_WIDTH - 2
            p.setPen(QColor("#cccccc"))
            elided = fm.elidedText(self._value_text, Qt.ElideRight,
                                   VALUE_WIDTH - 4)
            p.drawText(value_x, 0, VALUE_WIDTH - 4, self.height(),
                       Qt.AlignVCenter | Qt.AlignRight, elided)

    def mousePressEvent(self, event) -> None:
        """Route the click to the right region: label / plot.

        Lock clicks come from the QPushButton's own ``clicked`` signal
        (Qt swallows mouse events on child widgets before they reach the
        parent), so we only handle label + plot here.
        """
        x = event.position().x() if hasattr(event, "position") else event.x()
        if x < LOCK_WIDTH + LABEL_WIDTH:
            # Anywhere left of the plot baseline that isn't the lock
            # button is the label area — select this pane.
            self.label_clicked.emit()
            return
        plot_w = self._plot_w()
        if plot_w <= 0:
            return
        frac = max(0.0, min(1.0, (x - self._plot_x0()) / plot_w))
        self.clicked_at.emit(float(frac))

    def mouseMoveEvent(self, event) -> None:
        """Drag on the plot area continues the cursor update."""
        if not (event.buttons() & Qt.LeftButton):
            return
        x = event.position().x() if hasattr(event, "position") else event.x()
        # Only the plot region scrubs — dragging across the label / lock
        # shouldn't keep firing cursor updates.
        if x < LOCK_WIDTH + LABEL_WIDTH:
            return
        plot_w = self._plot_w()
        if plot_w <= 0:
            return
        frac = max(0.0, min(1.0, (x - self._plot_x0()) / plot_w))
        self.clicked_at.emit(float(frac))


class TimelineStrip(QWidget):
    """Vertical stack of N :class:`Track` widgets sharing a datetime domain.

    Configured by the main window via :meth:`set_panes` whenever the
    visible-pane set changes (layout switch, color-by change, file open).

    Signals
    -------
    cursor_changed
        Absolute datetime the user clicked / dragged to on any track —
        the window propagates it to every pane via the master-cursor sync.
    pane_selected
        Pane index whose label area the user clicked — alternative to
        clicking the sphere; the window calls ``_select_pane`` with it.
    lock_toggle_requested
        Pane index whose lock icon the user clicked. The window inverts
        that pane's ``time_locked`` state.
    """

    cursor_changed = Signal(object)
    pane_selected = Signal(int)
    lock_toggle_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._tracks: list[Track] = []
        self._domain_t0 = None
        self._domain_t1 = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(2)
        self.setVisible(False)

    def set_panes(self, panes: Sequence[tuple[str, Sequence]]) -> None:
        """Configure the strip from a list of ``(label, sample_times)`` pairs.

        Pass an empty list to hide the strip (no time-varying panes).
        The domain is the union of every pane's first/last sample; if
        any pane has no samples (e.g. its color_by is non-time-varying),
        skip it in the domain calculation but still draw its label.
        """
        if not panes:
            self.setVisible(False)
            return

        # Drop any extra tracks beyond the new pane count.
        while len(self._tracks) > len(panes):
            track = self._tracks.pop()
            self._layout.removeWidget(track)
            track.deleteLater()
        # Add tracks if we're growing. Wire the per-region signals up here
        # so the index in our list always matches the pane index the
        # window cares about.
        while len(self._tracks) < len(panes):
            idx = len(self._tracks)
            track = Track("", self)
            track.clicked_at.connect(self._on_track_clicked)
            track.label_clicked.connect(
                lambda i=idx: self.pane_selected.emit(i))
            track.lock_clicked.connect(
                lambda i=idx: self.lock_toggle_requested.emit(i))
            self._tracks.append(track)
            self._layout.addWidget(track)

        # Compute the union datetime domain from every pane with samples.
        all_times = [t for _, times in panes for t in times if t is not None]
        if all_times:
            self._domain_t0 = min(all_times)
            self._domain_t1 = max(all_times)
        else:
            self._domain_t0 = None
            self._domain_t1 = None

        for track, (label, times) in zip(self._tracks, panes, strict=True):
            track.set_label(label)
            track.set_samples(times)
            track.set_domain(self._domain_t0, self._domain_t1)

        # Fix the overall strip height so the splitter doesn't resize it.
        self.setFixedHeight(
            len(panes) * TRACK_HEIGHT + len(panes) * 2  # spacing
            + 8                                          # outer margins
        )
        self.setVisible(True)

    def set_cursors(self, cursors: Sequence) -> None:
        """Set each track's cursor independently.

        Locked panes keep their cursor at their pinned datetime
        regardless of the master cursor's position, so the cursor bar
        on a locked track stays where the data is — not where the
        master cursor moved to. Length mismatch is silently clamped
        to the shorter.
        """
        for track, t in zip(self._tracks, cursors, strict=False):
            track.set_cursor(t)

    def set_pane_values(self, values: Sequence[str]) -> None:
        """Update each track's right-hand value text.

        Pass an empty string for a track that has no pick / no value;
        the value column hides on empty. Length mismatch is silently
        clamped to the shorter — caller is responsible for passing one
        entry per visible pane.
        """
        for track, text in zip(self._tracks, values, strict=False):
            track.set_value(text)

    def set_pane_locked(self, pane_idx: int, locked: bool) -> None:
        """Update the lock icon on track ``pane_idx``."""
        if 0 <= pane_idx < len(self._tracks):
            self._tracks[pane_idx].set_locked(locked)

    def _on_track_clicked(self, frac: float) -> None:
        """Convert a track-fraction back to a datetime and emit upward."""
        if self._domain_t0 is None or self._domain_t1 is None:
            return
        delta = self._domain_t1 - self._domain_t0
        try:
            t = self._domain_t0 + delta * frac
        except TypeError:
            # cftime may not support delta * float; fall back via total seconds.
            seconds = delta.total_seconds() * frac
            from datetime import timedelta
            t = self._domain_t0 + timedelta(seconds=seconds)
        self.cursor_changed.emit(t)
