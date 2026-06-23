"""Status-bar pick state clears on mesh change, file unload, layout change.

Each of these is a regression that surfaced (or would have surfaced)
during code review of the multi-pane work — exactly the kind of bug
that's invisible to the unit suite because it lives in widget side
effects.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QCoreApplication


def _nearest_cell(win, lon: float = 0.0, lat: float = 0.0) -> int:
    centers = np.asarray(win.centers)
    lats = np.degrees(np.arcsin(np.clip(centers[:, 2], -1, 1)))
    lons = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
    return int(np.argmin((lats - lat) ** 2 + (lons - lon) ** 2))


def _pick(win, pane_idx: int = 0, lon: float = 0.0, lat: float = 0.0):
    win._on_pane_pick(pane_idx, _nearest_cell(win, lon, lat),
                      lon=lon, lat=lat)
    QCoreApplication.processEvents()


def _has_highlight(win, pane_idx: int) -> bool:
    actors = win._pickers[pane_idx].plotter.renderer.actors
    return "highlight" in actors


def _status_is_clear(win) -> bool:
    return (not win.value_label.text()
            and not win.value_label.isVisible())


def test_mesh_change_clears_pick_status(make_main_window):
    """_apply_mesh_change must wipe highlight + lon/lat + value widgets."""
    win = make_main_window()
    win.state.panes[0].color_by = "tas"
    win._refresh_scalars(0)
    win._update_scalars_only(0)
    _pick(win)

    assert _has_highlight(win, 0)
    assert win.value_label.isVisible()

    # Simulate any mesh-changing operation (parameter spinbox, grid swap …).
    win._apply_mesh_change()
    QCoreApplication.processEvents()

    assert not _has_highlight(win, 0), "stale highlight after mesh change"
    assert _status_is_clear(win), \
        f"value_label still visible: {win.value_label.text()!r}"


def test_layout_change_clears_stale_highlights(make_main_window):
    """Layout switch retires the prior pick so re-revealed panes are clean."""
    win = make_main_window()
    # Expand first — _on_pane_layout grows state.panes to length 4;
    # writing pane[3].color_by before then is an IndexError.
    win._on_pane_layout(4)
    QCoreApplication.processEvents()
    for idx, field in {0: "tas", 1: "tas_anomaly", 2: "precip",
                       3: "tas"}.items():
        win.state.panes[idx].color_by = field
        win._refresh_scalars(idx)
        win._update_scalars_only(idx)
    _pick(win, pane_idx=1)

    # Sync mode is the default — every visible pane should hold a highlight.
    assert all(_has_highlight(win, i) for i in range(4))

    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    assert _status_is_clear(win), "status should be cleared on layout shrink"

    win._on_pane_layout(4)
    QCoreApplication.processEvents()
    assert not any(_has_highlight(win, i) for i in range(4)), \
        "panes re-revealed with stale highlight outlines"


def test_escape_clears_pick_and_deselects_pane(make_main_window):
    """Escape is the canonical full-deselect."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    win.state.panes[0].color_by = "tas"
    win.state.panes[1].color_by = "tas_anomaly"  # noqa: index ok after layout=2
    for i in (0, 1):
        win._refresh_scalars(i)
        win._update_scalars_only(i)
    _pick(win, pane_idx=1)
    assert win._selected_pane == 1
    assert _has_highlight(win, 1)

    win._on_escape()
    QCoreApplication.processEvents()
    assert win._selected_pane is None
    assert not any(_has_highlight(win, i) for i in range(2))
    assert _status_is_clear(win)
