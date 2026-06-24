"""Loop checkbox controls wrap-vs-stop behaviour at the end of the time axis.

Drives ``_play_step`` synchronously rather than waiting on a real QTimer
firing in the test runner — letting the timer actually fire under
pytest's fixture lifecycle crashes Qt on macOS. The wrap-vs-stop
decision lives entirely inside ``_play_step``'s body, so a direct call
plus a stopped timer is sufficient.
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
    """Hand ``_play_step`` a stopped timer it can call ``.stop()`` on.

    Avoids ``play_btn.setChecked(True)`` (which would start a real QTimer
    and is the variant that crashes Qt under pytest).
    """
    win.playback._play_timer = QTimer(win)
    win.panel.file_tab.display_pane.play_btn.blockSignals(True)
    win.panel.file_tab.display_pane.play_btn.setChecked(True)
    win.panel.file_tab.display_pane.play_btn.blockSignals(False)


def test_loop_on_wraps_at_end_of_axis(make_main_window):
    """With Loop ON, _play_step at the last frame wraps back to index 0."""
    win = make_main_window()
    n = _setup_time_field(win)
    _prime_playback(win)
    win._file_state.loop_playback = True

    # Jump to the last frame programmatically, then run one playback step.
    win.state.panes[0].time_index = n - 1
    win.panel.file_tab.display_pane.time_slider.setValue(n - 1)
    QCoreApplication.processEvents()
    win.playback._play_step()
    QCoreApplication.processEvents()

    assert win.state.panes[0].time_index == 0, \
        f"loop ON should wrap to 0, got {win.state.panes[0].time_index}"
    assert win.panel.file_tab.display_pane.play_btn.isChecked(), \
        "loop ON should not stop playback at the end"


def test_loop_off_stops_at_end_of_axis(make_main_window):
    """With Loop OFF, _play_step at the last frame stops + unchecks Play."""
    win = make_main_window()
    n = _setup_time_field(win)
    _prime_playback(win)
    win._file_state.loop_playback = False

    win.state.panes[0].time_index = n - 1
    win.panel.file_tab.display_pane.time_slider.setValue(n - 1)
    QCoreApplication.processEvents()
    win.playback._play_step()
    QCoreApplication.processEvents()

    assert win.state.panes[0].time_index == n - 1, \
        f"loop OFF must hold the last frame, got {win.state.panes[0].time_index}"
    assert not win.panel.file_tab.display_pane.play_btn.isChecked(), \
        "loop OFF must auto-uncheck Play at the end of the axis"


def test_loop_checkbox_toggle_updates_file_state(make_main_window):
    """Clicking the Loop checkbox writes through to _file_state.loop_playback."""
    win = make_main_window()
    _setup_time_field(win)
    cb = win.panel.file_tab.display_pane.loop_cb
    assert cb.isChecked()                                  # default ON
    assert win._file_state.loop_playback is True

    cb.setChecked(False)
    QCoreApplication.processEvents()
    assert win._file_state.loop_playback is False
