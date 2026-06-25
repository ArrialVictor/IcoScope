"""Camera-sync mirror is deferred so VTK ModifiedEvent bursts coalesce.

The runtime perf audit flagged ``_on_camera_modified`` for doing a full
mirror + render of every other visible pane on every VTK
ModifiedEvent. A single user mouse-move can fire several ModifiedEvents
(SetPosition / SetFocalPoint / SetViewUp inside VTK's interactor); the
fix defers the mirror to the next Qt event-loop tick so they collapse
into one operation per iteration.

These tests pin down the deferred-mirror contract:
- Mirror does NOT happen synchronously inside _on_camera_modified.
- After processEvents, the most recent source pane's state wins.
- Calling _on_camera_modified N times before yielding still results in
  exactly one _flush_camera_sync execution.
"""
from __future__ import annotations

from qtpy.QtCore import QCoreApplication


def _pane_camera_position(win, idx):
    cam = win._pane_container.pane(idx).plotter.renderer.GetActiveCamera()
    return cam.GetPosition()


def test_camera_modified_defers_mirror_until_next_tick(make_main_window, set_field):
    """_on_camera_modified does NOT mirror synchronously; processEvents() does."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")

    baseline_pos1 = _pane_camera_position(win, 1)
    src_cam = win._pane_container.pane(0).plotter.renderer.GetActiveCamera()
    src_cam.SetPosition(9.0, 4.0, 1.5)
    src_cam.SetFocalPoint(0.0, 0.0, 0.0)

    # Trigger the observer manually; with the deferred-mirror change,
    # pane 1 must NOT have moved yet (the mirror is queued for the next
    # Qt tick).
    win._on_camera_modified(0)
    assert _pane_camera_position(win, 1) == baseline_pos1, (
        "mirror must be deferred; pane 1 should not move synchronously"
    )
    assert win._camera_sync_pending_src == 0

    QCoreApplication.processEvents()
    # Now the timer has fired and the mirror has been applied.
    assert _pane_camera_position(win, 1) == _pane_camera_position(win, 0)
    assert win._camera_sync_pending_src is None


def test_burst_of_modifications_coalesces_to_one_mirror(make_main_window, set_field):
    """Multiple _on_camera_modified calls before yielding → one flush, last wins."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")

    src_cam = win._pane_container.pane(0).plotter.renderer.GetActiveCamera()
    # Three back-to-back modifications, each fires _on_camera_modified
    # via the VTK observer chain when SetPosition runs. The pending_src
    # slot just keeps getting overwritten with the same value (0) — no
    # extra flush gets scheduled.
    for x in (3.0, 5.0, 7.0):
        src_cam.SetPosition(x, x, x)
        win._on_camera_modified(0)
    final_pos = src_cam.GetPosition()

    assert win._camera_sync_pending_src == 0

    QCoreApplication.processEvents()
    # Pane 1 ends at the LAST source position, not any of the intermediates.
    assert _pane_camera_position(win, 1) == final_pos


def test_sync_off_during_pending_mirror_skips_the_flush(make_main_window, set_field):
    """Toggling sync OFF between schedule and flush must cancel the mirror."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")

    baseline_pos1 = _pane_camera_position(win, 1)
    src_cam = win._pane_container.pane(0).plotter.renderer.GetActiveCamera()
    src_cam.SetPosition(6.0, 6.0, 6.0)
    win._on_camera_modified(0)        # schedules the flush
    assert win._camera_sync_pending_src == 0

    # User toggles sync OFF before the timer fires.
    win._on_camera_sync(False)
    QCoreApplication.processEvents()

    # Pane 1 must not have been mirrored — sync is off.
    assert _pane_camera_position(win, 1) == baseline_pos1
