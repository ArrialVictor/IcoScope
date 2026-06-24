"""Loop checkbox controls wrap-vs-stop behaviour at the end of the time range.

Playback is now cursor-stride: each tick advances the master cursor by
``stride = (PLAYBACK_TICK_MS / speed_ms) * unit_seconds`` and
``_set_master_cursor`` propagates to every visible pane. End-of-axis is
the union of all visible panes' axes (the strip's ``_domain_t1``).

Drives ``_play_step`` synchronously rather than waiting on a real QTimer
firing in the test runner — letting the timer actually fire under
pytest's fixture lifecycle crashes Qt on macOS.
"""
from __future__ import annotations

from qtpy.QtCore import QCoreApplication, QTimer


def _setup_time_field(win) -> int:
    """Single-pane File-tab with the 12-step monthly tas_t field. Returns N."""
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    win._select_pane(0)
    QCoreApplication.processEvents()
    win._on_color_by("tas_t")
    QCoreApplication.processEvents()
    return win._file_state.file_fields["tas_t"]["shape"][0]


def _prime_playback(win) -> None:
    """Hand ``_play_step`` a stopped timer it can call ``.stop()`` on."""
    win.playback._play_timer = QTimer(win)
    bar = win._timeline_strip.playback_bar
    bar.play_btn.blockSignals(True)
    bar.play_btn.setChecked(True)
    bar.play_btn.blockSignals(False)


def test_loop_on_wraps_at_end_of_axis(make_main_window):
    """With Loop ON, _play_step past the last datetime wraps to the first."""
    win = make_main_window()
    _setup_time_field(win)
    _prime_playback(win)
    win._file_state.loop_playback = True

    # Jump the cursor to the union's last datetime, then take one stride.
    t0 = win._timeline_strip._domain_t0
    t1 = win._timeline_strip._domain_t1
    win._set_master_cursor(t1)
    QCoreApplication.processEvents()
    win.playback._play_step()
    QCoreApplication.processEvents()

    # Stride advances past t1 → loop on wraps to t0.
    assert win._file_state.time_cursor == t0, \
        f"loop ON should wrap to t0, got {win._file_state.time_cursor}"
    assert win._timeline_strip.playback_bar.play_btn.isChecked(), \
        "loop ON should not stop playback at the end"


def test_loop_off_stops_at_end_of_axis(make_main_window):
    """With Loop OFF, _play_step past the last datetime stops + unchecks Play."""
    win = make_main_window()
    _setup_time_field(win)
    _prime_playback(win)
    win._file_state.loop_playback = False

    t1 = win._timeline_strip._domain_t1
    win._set_master_cursor(t1)
    QCoreApplication.processEvents()
    win.playback._play_step()
    QCoreApplication.processEvents()

    # Stride past t1 → loop off lands exactly on t1, stops, unchecks Play.
    assert win._file_state.time_cursor == t1, \
        "loop OFF must clamp the cursor at the union's last sample"
    assert not win._timeline_strip.playback_bar.play_btn.isChecked(), \
        "loop OFF must auto-uncheck Play at the end of the range"


def test_loop_checkbox_toggle_updates_file_state(make_main_window):
    """Toggling the strip's Loop checkbox writes through to _file_state."""
    win = make_main_window()
    _setup_time_field(win)
    cb = win._timeline_strip.playback_bar.loop_cb
    assert cb.isChecked()                                  # default ON
    assert win._file_state.loop_playback is True

    cb.setChecked(False)
    QCoreApplication.processEvents()
    assert win._file_state.loop_playback is False


def test_speed_change_updates_file_state(make_main_window):
    """Changing the strip's speed spinbox or unit combo updates _file_state."""
    win = make_main_window()
    _setup_time_field(win)
    bar = win._timeline_strip.playback_bar

    bar.speed_box.setValue(250)
    QCoreApplication.processEvents()
    assert win._file_state.playback_speed_value == 250

    bar.unit_combo.setCurrentText("month")
    QCoreApplication.processEvents()
    assert win._file_state.playback_speed_unit == "month"
