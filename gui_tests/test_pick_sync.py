"""Cross-pane pick sync follows the camera-sync toggle.

In sync mode every visible pane should highlight the same cell; in
desync mode only the clicked pane should. The selected pane updates
in both modes.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QCoreApplication


def _nearest_cell(win, lon=0.0, lat=0.0):
    centers = np.asarray(win.centers)
    lats = np.degrees(np.arcsin(np.clip(centers[:, 2], -1, 1)))
    lons = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
    return int(np.argmin((lats - lat) ** 2 + (lons - lon) ** 2))


def _has_highlight(win, pane_idx):
    return "highlight" in win._pickers[pane_idx].plotter.renderer.actors


def _setup_four(win):
    # Layout expansion must happen first — _on_pane_layout grows the
    # state.panes list to 4 entries; assigning panes[3].color_by before
    # then is an IndexError.
    win._on_pane_layout(4)
    QCoreApplication.processEvents()
    for idx, field in {0: "tas", 1: "tas_anomaly", 2: "precip",
                       3: "tas"}.items():
        win.state.panes[idx].color_by = field
        win._refresh_scalars(idx)
        win._update_scalars_only(idx)
    QCoreApplication.processEvents()


def test_sync_on_highlights_every_visible_pane(make_main_window):
    win = make_main_window()
    _setup_four(win)
    win._on_camera_sync(True)
    QCoreApplication.processEvents()
    cell = _nearest_cell(win)

    win._on_pane_pick(1, cell, lon=0.0, lat=0.0)
    QCoreApplication.processEvents()

    assert win._selected_pane == 1
    assert all(_has_highlight(win, i) for i in range(4)), \
        "sync mode should highlight every visible pane"


def test_sync_off_highlights_only_clicked_pane(make_main_window):
    win = make_main_window()
    _setup_four(win)
    win._on_camera_sync(False)
    QCoreApplication.processEvents()
    cell = _nearest_cell(win)

    win._on_pane_pick(2, cell, lon=0.0, lat=0.0)
    QCoreApplication.processEvents()

    assert win._selected_pane == 2
    assert _has_highlight(win, 2)
    assert not any(_has_highlight(win, i) for i in (0, 1, 3)), \
        "desync mode should leave other panes untouched"


def test_pick_miss_clears_everything_and_deselects(make_main_window):
    win = make_main_window()
    _setup_four(win)
    cell = _nearest_cell(win)
    win._on_pane_pick(1, cell, lon=0.0, lat=0.0)
    QCoreApplication.processEvents()
    assert win._selected_pane == 1
    assert _has_highlight(win, 1)

    win._on_pane_pick(1, None)
    QCoreApplication.processEvents()

    assert win._selected_pane is None, \
        "miss should drop pane selection (matches Escape)"
    assert not any(_has_highlight(win, i) for i in range(4))
    assert not win.value_label.isVisible()
