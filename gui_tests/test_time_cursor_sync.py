"""Master datetime cursor propagates across panes with different time axes.

The synthetic NetCDF has two distinct time dims:
    - ``time`` (monthly cadence, 12 samples, days 0..330 of 2020)
    - ``time_counter`` (daily cadence, 60 samples, days 0..59 of 2020)

Two panes, one field on each axis. Scrubbing pane 0's slider should
update the cursor and propagate the nearest sample to pane 1. Some
cursor positions land in pane 1's range (banner hidden), others land
outside (banner shown).
"""
from __future__ import annotations

from qtpy.QtCore import QCoreApplication


def _set_field(win, pane_idx: int, field: str) -> None:
    win.state.panes[pane_idx].color_by = field
    win._refresh_scalars(pane_idx)
    win._update_scalars_only(pane_idx)


def test_cursor_propagates_to_pane_on_different_axis(make_main_window):
    """Pane 0 monthly, pane 1 daily — scrub pane 0, pane 1 follows in days."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")        # 12 monthly samples
    _set_field(win, 1, "tas_daily")    # 60 daily samples
    win._select_pane(0)
    QCoreApplication.processEvents()

    # Scrub pane 0 to index 1: monthly[1] = day 30 of 2020.
    # Daily axis has day 30 at index 30 — pane 1 should land there.
    win._on_time_changed(1)
    QCoreApplication.processEvents()
    assert win.state.panes[1].time_index == 30, (
        f"expected pane 1 at day-30 (index 30), got "
        f"{win.state.panes[1].time_index}")

    # Cursor at day 30 is still inside the daily axis (0..59) so the
    # banner should be hidden on pane 1.
    pane1_widget = win._pane_container.pane(1)
    assert not pane1_widget.banner_visible


def test_cursor_out_of_range_shows_banner(make_main_window):
    """Scrub pane 0 to month 5 (day 150) — past the daily axis end (day 59)."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")
    _set_field(win, 1, "tas_daily")
    win._select_pane(0)
    QCoreApplication.processEvents()

    # Monthly[5] = day 150 of 2020 — far past day 59.
    win._on_time_changed(5)
    QCoreApplication.processEvents()

    # Pane 1's daily axis is clamped to its last sample.
    assert win.state.panes[1].time_index == 59
    pane1_widget = win._pane_container.pane(1)
    assert pane1_widget.banner_visible, \
        "out-of-range cursor should surface the banner"
    text = pane1_widget.banner_text
    assert "Showing" in text and "cursor at" in text, \
        f"unexpected banner text: {text!r}"


def test_cursor_resolves_on_color_by_change(make_main_window):
    """Switching a pane's field to a different axis re-resolves the cursor."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")
    _set_field(win, 1, "tas_t")
    win._select_pane(0)
    QCoreApplication.processEvents()
    win._on_time_changed(3)            # cursor = day 90
    QCoreApplication.processEvents()
    assert win.state.panes[1].time_index == 3

    # Now switch pane 1 to the daily field. Cursor (day 90) is past
    # day 59 → clamp to last sample + show banner.
    win._select_pane(1)
    QCoreApplication.processEvents()
    win._on_color_by("tas_daily")
    QCoreApplication.processEvents()
    assert win.state.panes[1].time_index == 59
    assert win._pane_container.pane(1).banner_visible


def test_file_unload_clears_cursor_and_banners(make_main_window):
    """File unload retires the cursor and hides every pane's banner."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")
    _set_field(win, 1, "tas_daily")
    win._select_pane(0)
    QCoreApplication.processEvents()
    win._on_time_changed(5)            # out-of-range for pane 1
    QCoreApplication.processEvents()
    assert win._pane_container.pane(1).banner_visible

    win._on_close_file()
    QCoreApplication.processEvents()
    assert win._file_state.time_cursor is None
    assert not win._pane_container.pane(1).banner_visible
