"""Regression tests for the audit-cleanup PR A correctness fixes.

Covers:

- TimelineStrip domain reset when every pane becomes time-static
  (otherwise Playback._play_step advances against stale bounds).
- Locked pane keeps its pinned time_index across a color_by change
  (the lock contract must hold on every entry point, not just master-cursor moves).
- ``clim_symmetric`` snapshot includes hidden panes so layout shrink ->
  toggle -> expand doesn't lose / corrupt per-pane ``center_zero``.
"""
from __future__ import annotations

from qtpy.QtCore import QCoreApplication


def _set_field(win, pane_idx: int, field: str) -> None:
    win._select_pane(pane_idx)
    QCoreApplication.processEvents()
    win._on_color_by(field)
    QCoreApplication.processEvents()


def test_timeline_domain_resets_when_no_time_varying_panes(make_main_window):
    """Switching every pane to a non-time-varying field must null the strip's domain.

    Otherwise the playback timer keeps advancing against the previously
    cached ``_domain_t0/_t1``, even though no pane has any time samples
    to map onto.
    """
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")
    _set_field(win, 1, "tas_t")

    # Domain is populated as soon as a time-varying field is up.
    assert win._timeline_strip.domain[0] is not None
    assert win._timeline_strip.domain[1] is not None

    # Drop both panes to 'None' — strip hides + domain must reset.
    _set_field(win, 0, "None")
    _set_field(win, 1, "None")

    assert win._timeline_strip.domain[0] is None
    assert win._timeline_strip.domain[1] is None
    assert not win._timeline_strip.isVisible()


def test_locked_pane_keeps_time_index_on_color_by_change(make_main_window):
    """A locked pane's ``time_index`` must not be reassigned when its field changes.

    Lock contract: the pane is pinned to a specific datetime; nothing
    should silently move it. ``_on_color_by`` previously unconditionally
    resolved the master cursor against the new field's axis and wrote
    the result back, breaking the lock at the moment the user swapped
    fields on the locked pane.
    """
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")
    _set_field(win, 1, "tas_t")

    # Advance master cursor so pane 1 sits at a non-zero index, then lock it.
    times = win._times_for(win._file_state.file_fields["tas_t"])
    win._timeline_strip.cursor_changed.emit(times[3])
    QCoreApplication.processEvents()
    win._select_pane(1)
    QCoreApplication.processEvents()
    win._on_timeline_lock_toggled(1)
    QCoreApplication.processEvents()
    pinned_idx = win.state.panes[1].time_index
    assert win.state.panes[1].time_locked is True
    assert pinned_idx != 0

    # Move master forward, then change the locked pane's field. The
    # field swap must NOT realign the locked pane to the cursor.
    win._timeline_strip.cursor_changed.emit(times[7])
    QCoreApplication.processEvents()
    _set_field(win, 1, "tas_daily")

    assert win.state.panes[1].time_locked is True, \
        "lock state itself must survive the field change"
    assert win.state.panes[1].time_index == pinned_idx, \
        "locked pane's time_index must not be reassigned on color_by change"


def test_clim_symmetric_carry_back_covers_hidden_panes(make_main_window):
    """Toggling shared OFF must carry the shared value back to every pane.

    Previously the carry-back loop iterated only visible panes, so a
    pane that was visible when shared was ON (and thus rendering with
    the shared symmetric value) but hidden at the moment of the OFF
    toggle would retain a stale per-pane ``center_zero``. Re-expanding
    the layout afterwards then showed the wrong colourbar.
    """
    win = make_main_window()
    # Start with 4 panes so all four register a per-pane center_zero=False.
    win._on_pane_layout(4)
    QCoreApplication.processEvents()
    for i in range(4):
        _set_field(win, i, "tas_anomaly")
    assert all(p.center_zero is False for p in win.state.panes[:4])

    # Make tas_anomaly symmetric in shared mode (one write -> all panes
    # use it via the shared dict).
    win._select_pane(0)
    QCoreApplication.processEvents()
    win._on_center_zero(True)
    QCoreApplication.processEvents()
    assert win._file_state.clim_symmetric.get("tas_anomaly") is True

    # Shrink to 2 panes — pane 3 and pane 4 are hidden but their
    # PaneState entries still exist in win.state.panes[2:4].
    win._on_pane_layout(2)
    QCoreApplication.processEvents()

    # Toggle shared OFF — carry-back must hit ALL panes including hidden.
    win._on_clim_shared_toggled(False)
    QCoreApplication.processEvents()

    for i, pane in enumerate(win.state.panes[:4]):
        assert pane.center_zero is True, (
            f"pane {i + 1}: carry-back must reach hidden panes too "
            f"(got center_zero={pane.center_zero})"
        )
