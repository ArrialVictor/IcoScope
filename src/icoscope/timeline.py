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
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

# Track height and total-strip height tuned to land under the viewports
# without dominating the layout. Update both together if the row height
# grows; the strip's setFixedHeight uses N × TRACK_HEIGHT + padding.
TRACK_HEIGHT = 28
LABEL_WIDTH = 96   # pixels reserved on the left for the track label
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

    Signals
    -------
    clicked_at
        Emitted with a fraction in ``[0, 1]`` mapped back to the shared
        datetime domain by the parent. The parent decides what to do
        with it (currently: set the master cursor).
    """

    clicked_at = Signal(float)

    def __init__(self, label: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._label = label
        self._times: list = []          # sample datetimes for this pane
        self._domain_t0 = None          # parent-supplied range
        self._domain_t1 = None
        self._cursor_t = None
        self.setFixedHeight(TRACK_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(False)

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
        plot_w = self.width() - LABEL_WIDTH - 2 * HORIZONTAL_PADDING
        return LABEL_WIDTH + HORIZONTAL_PADDING + ratio * plot_w

    def paintEvent(self, _event) -> None:
        """Custom paint: label, baseline, sample dots, cursor line."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Left-hand label, vertically centred and elided to fit.
        p.setPen(QColor("#cccccc"))
        fm = QFontMetrics(p.font())
        elided = fm.elidedText(self._label, Qt.ElideRight, LABEL_WIDTH - 8)
        p.drawText(4, 0, LABEL_WIDTH - 8, self.height(),
                   Qt.AlignVCenter | Qt.AlignRight, elided)

        # Baseline — thin horizontal line across the plot area.
        plot_x0 = LABEL_WIDTH + HORIZONTAL_PADDING
        plot_x1 = self.width() - HORIZONTAL_PADDING
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

    def mousePressEvent(self, event) -> None:
        """Map the click x to a fraction in [0, 1] and emit ``clicked_at``."""
        x = event.position().x() if hasattr(event, "position") else event.x()
        plot_x0 = LABEL_WIDTH + HORIZONTAL_PADDING
        plot_w = self.width() - LABEL_WIDTH - 2 * HORIZONTAL_PADDING
        if plot_w <= 0:
            return
        frac = max(0.0, min(1.0, (x - plot_x0) / plot_w))
        self.clicked_at.emit(float(frac))

    def mouseMoveEvent(self, event) -> None:
        """Treat drag as a continuous cursor update (same path as click)."""
        if event.buttons() & Qt.LeftButton:
            self.mousePressEvent(event)


class TimelineStrip(QWidget):
    """Vertical stack of N :class:`Track` widgets sharing a datetime domain.

    Configured by the main window via :meth:`set_panes` whenever the
    visible-pane set changes (layout switch, color-by change, file open).
    Emits :attr:`cursor_changed` with the absolute datetime the user
    clicked / dragged to on any track; the window then propagates the
    cursor to every pane through the existing master-cursor sync.
    """

    cursor_changed = Signal(object)   # datetime — sent on any track click/drag

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
        # Add tracks if we're growing.
        while len(self._tracks) < len(panes):
            track = Track("", self)
            track.clicked_at.connect(self._on_track_clicked)
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

    def set_cursor(self, t) -> None:
        """Move the shared cursor on every track to datetime ``t``."""
        for track in self._tracks:
            track.set_cursor(t)

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
