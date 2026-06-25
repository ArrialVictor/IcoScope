"""Slider scrub on a locked pane must respect the lock contract.

PR #35 (audit PR A) fixed ``_on_color_by`` to skip the time_index
reassignment when ``pane.time_locked`` is True. The architecture
audit found the same contract was being violated by ``_on_time_changed``
— dragging the slider on a locked pane bypassed the lock. This file
pins the fix down so a future refactor can't reintroduce the gap.
"""
from __future__ import annotations

from qtpy.QtCore import QCoreApplication


def test_slider_scrub_on_locked_pane_keeps_pinned_time_index(make_main_window, set_field):
    """Direct slider drag on a locked pane is a no-op on time_index."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    set_field(win, 1, "tas_t")

    # Advance the active pane to a non-zero sample, then lock it.
    times = win._times_for(win._file_state.file_fields["tas_t"])
    win._timeline_strip.cursor_changed.emit(times[3])
    QCoreApplication.processEvents()
    win._select_pane(0)
    QCoreApplication.processEvents()
    win._on_timeline_lock_toggled(0)
    QCoreApplication.processEvents()
    pinned_idx = win.state.panes[0].time_index
    assert win.state.panes[0].time_locked is True
    assert pinned_idx != 0

    # Drive _on_time_changed as the slider does — the lock contract
    # says the pane stays pinned even when the side-panel slider is
    # scrubbed directly.
    target_idx = pinned_idx + 4
    win._on_time_changed(target_idx)
    QCoreApplication.processEvents()

    assert win.state.panes[0].time_index == pinned_idx, (
        f"locked pane time_index must not move on slider scrub "
        f"(pinned {pinned_idx}, slider asked for {target_idx}, "
        f"got {win.state.panes[0].time_index})"
    )


def test_slider_position_snaps_back_on_locked_pane(make_main_window, set_field):
    """After a no-op scrub on a locked pane, the slider widget shows the pinned index."""
    win = make_main_window()
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    times = win._times_for(win._file_state.file_fields["tas_t"])
    win._timeline_strip.cursor_changed.emit(times[2])
    QCoreApplication.processEvents()
    win._on_timeline_lock_toggled(0)
    QCoreApplication.processEvents()
    pinned_idx = win.state.panes[0].time_index

    # Slider drag would normally fire valueChanged → _on_time_changed
    # with the new index. The handler must snap the slider back so
    # widget and state agree.
    win._on_time_changed(pinned_idx + 5)
    QCoreApplication.processEvents()
    slider = win.panel.file_tab.display_pane.time_slider
    assert slider.value() == pinned_idx, (
        f"slider widget must snap back to pinned index after a locked-pane "
        f"scrub (slider says {slider.value()}, pinned {pinned_idx})"
    )


def test_slider_scrub_works_normally_when_unlocked(make_main_window, set_field):
    """The new guard must NOT affect normal (unlocked) slider scrubs."""
    win = make_main_window()
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    assert win.state.panes[0].time_locked is False

    win._on_time_changed(5)
    QCoreApplication.processEvents()
    assert win.state.panes[0].time_index == 5, (
        "unlocked pane must still respond to slider scrub"
    )
