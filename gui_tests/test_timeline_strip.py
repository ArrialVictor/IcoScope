"""Bottom timeline strip — visibility rules + cursor propagation."""
from __future__ import annotations

from datetime import timedelta

from qtpy.QtCore import QCoreApplication


def test_strip_hidden_until_time_field_selected(make_main_window, set_field):
    """Strip stays hidden when no visible pane shows a time-varying field."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas")          # static
    set_field(win, 1, "precip")       # static
    assert not win._timeline_strip.isVisible()


def test_strip_shows_one_track_per_visible_pane(make_main_window, set_field):
    """2-pane layout with time-varying fields → 2 tracks render."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    set_field(win, 1, "tas_daily")
    assert win._timeline_strip.isVisible()
    assert len(win._timeline_strip.tracks) == 2
    # Labels are 1-indexed pane number + color_by.
    assert win._timeline_strip.tracks[0].label == "1: tas_t"
    assert win._timeline_strip.tracks[1].label == "2: tas_daily"


def test_strip_hides_on_ico_tab(make_main_window, set_field):
    """Switching away from the File tab hides the strip."""
    from icoscope.tabs import Tab
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    set_field(win, 1, "tas_daily")
    assert win._timeline_strip.isVisible()
    win.panel.tabs.setCurrentIndex(Tab.ICO)
    QCoreApplication.processEvents()
    assert not win._timeline_strip.isVisible()


def test_strip_drag_propagates_cursor_to_panes(make_main_window, set_field):
    """Emitting cursor_changed from the strip moves both panes' time_index."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")       # 12 monthly samples
    set_field(win, 1, "tas_daily")   # 60 daily samples
    win._select_pane(0)
    QCoreApplication.processEvents()

    # Pull pane 0's monthly axis and pick a known datetime — sample #1 of
    # tas_t is day 30, which is inside the daily axis (0..59) too.
    times = win._times_for(win._file_state.file_fields["tas_t"])
    target = times[1]

    # Simulate the strip emitting cursor_changed (as if user dragged).
    win._timeline_strip.cursor_changed.emit(target)
    QCoreApplication.processEvents()

    assert win.state.panes[0].time_index == 1
    assert win.state.panes[1].time_index == 30   # day 30 → daily index 30
    assert win._file_state.time_cursor == target


def test_strip_domain_is_union_of_pane_axes(make_main_window, set_field):
    """Strip's shared domain spans min(first) → max(last) over visible panes."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")       # year-long
    set_field(win, 1, "tas_daily")   # first 60 days

    monthly = win._times_for(win._file_state.file_fields["tas_t"])
    daily = win._times_for(win._file_state.file_fields["tas_daily"])
    expected_start = min(monthly[0], daily[0])
    expected_end = max(monthly[-1], daily[-1])
    assert win._timeline_strip._domain_t0 == expected_start
    assert win._timeline_strip._domain_t1 == expected_end
    # Sanity: monthly axis (Jan–Dec) should be wider than daily (Jan–Feb).
    assert (expected_end - expected_start) > timedelta(days=300)
