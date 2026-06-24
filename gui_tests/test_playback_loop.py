"""Playback stops at the end of the time range + speed control updates state.

Loop-and-restart was removed (PR series — climate researchers prefer
the explicit end-of-data signal over silent restart), so playback now
always clamps to the last sample and unchecks the Play button.

Drives ``_play_step`` synchronously rather than waiting on a real
QTimer firing in the test runner — letting the timer actually fire
under pytest's fixture lifecycle crashes Qt on macOS.
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


def test_play_stops_at_end_of_axis(make_main_window):
    """_play_step past the union's last datetime clamps and unchecks Play."""
    win = make_main_window()
    _setup_time_field(win)
    _prime_playback(win)

    t1 = win._timeline_strip._domain_t1
    win._set_master_cursor(t1)
    QCoreApplication.processEvents()
    win.playback._play_step()
    QCoreApplication.processEvents()

    assert win._file_state.time_cursor == t1, \
        "playback must clamp the cursor at the union's last sample"
    assert not win._timeline_strip.playback_bar.play_btn.isChecked(), \
        "playback must auto-uncheck Play at the end of the range"


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
