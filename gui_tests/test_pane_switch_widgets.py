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


def test_speed_box_change_propagates_to_play_speed_signal(make_main_window):
    """speed_box.valueChanged must reach the FileTab proxy → app slot.

    Pre-existing bug: ``play_speed_changed`` was listed under the global
    block's signal proxy, but the speed_box widget itself is built only
    when ``with_time=True``, which the global block forces False — so
    the widget lives on the pane block. The signal fired but no proxy
    forwarded it, so changing the value never reached ``Playback.set_speed``.

    Verified by hooking a probe onto the tab's ``play_speed_changed``
    signal and checking it fires when the spinbox value changes.
    """
    win = make_main_window()
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")        # time-varying so the speed row exists

    received: list[int] = []
    win.panel.file_tab.play_speed_changed.connect(received.append)

    win.panel.file_tab.display_pane.speed_box.setValue(120)
    QCoreApplication.processEvents()

    assert received == [120], (
        f"expected play_speed_changed to fire with 120, got {received}")


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
