"""Switching panes refreshes the time/level rows to match the new pane.

Regression test: previously `_sync_pane_widgets` only called
`set_time_axis` / `set_levels` when the new pane's field WAS time-varying
or had levels — switching from a time-varying field to a static one
left the time row + play button visible, leading to the user pressing
play and seeing it auto-revert because `_play_step` can't advance a
static field.
"""
from __future__ import annotations

from qtpy.QtCore import QCoreApplication


def _set_field(win, pane_idx: int, field: str) -> None:
    """Configure pane ``pane_idx`` to display ``field`` via the proper slot."""
    win._select_pane(pane_idx)
    QCoreApplication.processEvents()
    win._on_color_by(field)
    QCoreApplication.processEvents()


def test_switch_from_time_to_static_pane_hides_time_row(make_main_window):
    """Pane 1 has tas_t (time-varying); pane 0 has tas (static).

    Selecting pane 0 must hide the time row + play button so the user
    can't kick off a playback that immediately stops.
    """
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas")          # static
    _set_field(win, 1, "tas_t")        # 12-step monthly axis

    display_pane = win.panel.file_tab.display_pane

    # While pane 1 is selected, time row must be visible.
    assert display_pane.time_row.isVisible(), \
        "time row should be visible for pane 1's time-varying field"

    # Switch to pane 0 — time row must hide.
    win._select_pane(0)
    QCoreApplication.processEvents()
    assert not display_pane.time_row.isVisible(), \
        "time row should hide when switching to a static-field pane"
    assert not display_pane.play_btn.isVisible(), \
        "play button should hide along with the time row"


def test_switch_back_to_time_pane_reshows_time_row(make_main_window):
    """Going pane 0 (static) → pane 1 (time) must re-show the time row."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas")
    _set_field(win, 1, "tas_t")

    win._select_pane(0)
    QCoreApplication.processEvents()
    win._select_pane(1)
    QCoreApplication.processEvents()

    display_pane = win.panel.file_tab.display_pane
    assert display_pane.time_row.isVisible()
    assert display_pane.time_slider.maximum() == 11   # 12 monthly samples
