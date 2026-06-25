"""Regression tests for the audit-cleanup PR B multi-pane state reads.

Covers the call sites that previously routed through ``_TabState``'s
``panes[0]`` back-compat properties and silently produced wrong-pane
behaviour in multi-pane mode:

- Picker units / cell value resolve against the picked pane's field,
  not pane 1's.
- Theme change updates every pane's cmap, not just pane 1's.
- Open NetCDF resets every pane (including hidden ones), so re-expanding
  the layout after a file switch never shows a stale field from the
  previous file.
"""
from __future__ import annotations

from qtpy.QtCore import QCoreApplication


def _set_field(win, pane_idx: int, field: str) -> None:
    # _select_pane is synchronous (no signals fired), so a processEvents
    # between it and _on_color_by drains nothing. One drain at the end is
    # enough — saves ~50-100ms per call across the GUI test suite.
    win._select_pane(pane_idx)
    win._on_color_by(field)
    QCoreApplication.processEvents()


def _pick_target_cell(win) -> int:
    import numpy as np
    centers = np.asarray(win.centers)
    lats = np.degrees(np.arcsin(np.clip(centers[:, 2], -1, 1)))
    lons = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
    return int(np.argmin(lats ** 2 + lons ** 2))


def test_pick_on_non_active_pane_uses_its_own_field_units(make_main_window):
    """Picking pane 2 must surface pane 2's units, not pane 1's.

    Before the fix, ``_current_color_by_units`` and ``_set_cell_value``
    read ``self.state.color_by`` which routed to pane 0, so picking on
    pane 2 displayed pane 1's field's units alongside pane 2's scalar
    value — wrong number-and-label combination.
    """
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")
    _set_field(win, 1, "vort_t")
    cell = _pick_target_cell(win)

    # Pick on pane 2 — selection promotes pane 2 to active.
    win._on_pane_pick(1, cell, lon=0.0, lat=0.0)
    QCoreApplication.processEvents()

    from icoscope.formatters import pretty_units
    tas_units = pretty_units(
        win._file_state.file_fields["tas_t"].get("units", ""))
    anom_units = pretty_units(
        win._file_state.file_fields["vort_t"].get("units", ""))
    # Sanity: the two fields must actually differ in units for this test
    # to be meaningful (otherwise the bug would be invisible).
    assert tas_units != anom_units, (
        "test fixture assumption: tas_t and vort_t must have "
        "distinguishable units"
    )

    label_text = win.value_label.text()
    assert anom_units in label_text, (
        f"pick on pane 2 (color_by=vort_t, units={anom_units!r}) "
        f"must show pane 2's units; got {label_text!r}"
    )
    assert tas_units not in label_text, (
        f"pick on pane 2 must NOT show pane 1's units ({tas_units!r}); "
        f"got {label_text!r}"
    )


def test_theme_change_updates_every_pane_cmap(make_main_window):
    """Switching theme propagates the suggested cmap to every pane on the tab.

    Before the fix, ``_on_theme`` wrote through ``tab_state.cmap``
    which only touched pane 0; panes 2–4 kept the old theme's cmap.
    """
    from icoscope.themes import THEMES

    win = make_main_window()
    win._on_pane_layout(4)
    QCoreApplication.processEvents()
    assert len(win._file_state.panes) >= 4

    # Pick a theme whose default cmap differs from whatever is current.
    current_cmap = win._file_state.panes[0].cmap
    other = next(
        name for name, t in THEMES.items() if t["cmap"] != current_cmap
    )
    win._on_theme(other)
    QCoreApplication.processEvents()

    expected = THEMES[other]["cmap"]
    for i, pane in enumerate(win._file_state.panes[:4]):
        assert pane.cmap == expected, (
            f"pane {i + 1}: theme change must update cmap on every pane "
            f"(expected {expected!r}, got {pane.cmap!r})"
        )


def test_file_close_resets_color_by_on_every_pane(make_main_window):
    """Closing the file clears color_by on every pane, not just pane 1.

    Otherwise a hidden pane (e.g. pane 4 after the user shrunk to 2x2)
    retains its prior field name; re-expanding the layout surfaces a
    stale field that no longer exists in the current load.
    """
    win = make_main_window()
    win._on_pane_layout(4)
    QCoreApplication.processEvents()
    for i in range(4):
        _set_field(win, i, "tas_t")
    assert all(p.color_by == "tas_t" for p in win._file_state.panes[:4])

    # Shrink the visible layout — panes 3 and 4 are now hidden but
    # their PaneState entries still hold color_by="tas_t".
    win._on_pane_layout(2)
    QCoreApplication.processEvents()

    win._on_close_file()
    QCoreApplication.processEvents()

    for i, pane in enumerate(win._file_state.panes[:4]):
        assert pane.color_by == "None", (
            f"pane {i + 1}: close-file must reset color_by on every pane "
            f"(got {pane.color_by!r}); hidden panes were previously skipped"
        )
