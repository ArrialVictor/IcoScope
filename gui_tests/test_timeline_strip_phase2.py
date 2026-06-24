"""Timeline strip Phase 2 — per-track value display, lock, label-click."""
from __future__ import annotations

from qtpy.QtCore import QCoreApplication


def _set_field(win, pane_idx: int, field: str) -> None:
    win._select_pane(pane_idx)
    QCoreApplication.processEvents()
    win._on_color_by(field)
    QCoreApplication.processEvents()


def _setup_two_panes(win) -> int:
    """2-pane File-tab, pane 1=tas_t (monthly), pane 2=tas_daily. Returns cell idx."""
    import numpy as np
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")
    _set_field(win, 1, "tas_daily")
    # Equator-ish cell for a stable, comparable value.
    centers = np.asarray(win.centers)
    lats = np.degrees(np.arcsin(np.clip(centers[:, 2], -1, 1)))
    lons = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
    return int(np.argmin(lats ** 2 + lons ** 2))


def test_pick_populates_value_column_on_every_track(make_main_window):
    """A cell pick surfaces a per-pane value on every track's value column."""
    win = make_main_window()
    cell = _setup_two_panes(win)
    win._select_pane(0)
    QCoreApplication.processEvents()

    # Empty until first pick.
    for track in win._timeline_strip._tracks:
        assert track.value_text == ""

    win._on_pane_pick(0, cell, lon=0.0, lat=0.0)
    QCoreApplication.processEvents()

    tracks = win._timeline_strip._tracks
    assert tracks[0].value_text, "pane 1 track value must populate on pick"
    assert tracks[1].value_text, "pane 2 track value must populate on pick"
    # Both fields are temperature-ish (~280-310 K) → both labels should
    # contain a "K" unit token. Belt-and-braces sanity check.
    assert "K" in tracks[0].value_text
    assert "K" in tracks[1].value_text


def test_lock_prevents_cursor_propagation(make_main_window):
    """Locking pane 2 pins its time_index while the cursor advances."""
    win = make_main_window()
    _setup_two_panes(win)
    win._select_pane(0)
    QCoreApplication.processEvents()

    # Lock pane 2 at its initial time_index (=0).
    win._on_timeline_lock_toggled(1)
    QCoreApplication.processEvents()
    assert win.state.panes[1].time_locked is True

    # Pull a known datetime from pane 1's axis (sample #1 = day 30).
    times = win._times_for(win._file_state.file_fields["tas_t"])
    target = times[1]
    win._timeline_strip.cursor_changed.emit(target)
    QCoreApplication.processEvents()

    # Pane 1 followed; pane 2 stayed pinned.
    assert win.state.panes[0].time_index == 1
    assert win.state.panes[1].time_index == 0, \
        "locked pane must not move with the cursor"


def test_label_click_selects_pane(make_main_window):
    """Clicking a track's label region routes to _select_pane."""
    win = make_main_window()
    _setup_two_panes(win)
    win._select_pane(0)
    QCoreApplication.processEvents()
    assert win._selected_pane == 0

    # The TimelineStrip emits pane_selected → connected to _on_pane_clicked.
    win._timeline_strip.pane_selected.emit(1)
    QCoreApplication.processEvents()
    assert win._selected_pane == 1


def test_lock_state_persists_across_layout_refresh(make_main_window):
    """Rebuilding tracks (e.g. color_by change) must preserve lock visuals."""
    win = make_main_window()
    _setup_two_panes(win)
    win._on_timeline_lock_toggled(1)
    QCoreApplication.processEvents()
    assert win._timeline_strip._tracks[1].locked is True

    # Force a rebuild by changing pane 1's color_by — _refresh_timeline_strip
    # runs and re-creates the tracks if pane count changed (it didn't, but
    # the lock-restore loop runs unconditionally).
    _set_field(win, 0, "tas_t")
    QCoreApplication.processEvents()
    assert win._timeline_strip._tracks[1].locked is True, \
        "lock visual must survive a strip rebuild"


def test_locked_track_cursor_stays_at_pinned_time(make_main_window):
    """The cursor bar on a locked track must not follow the master cursor.

    Regression: previously the strip pushed one shared cursor to every
    track, so a locked pane's data stayed pinned but its cursor bar
    drifted with the master — confusingly out of sync with the data.
    """
    win = make_main_window()
    _setup_two_panes(win)
    win._select_pane(0)
    QCoreApplication.processEvents()

    # Lock pane 2 at its initial time_index (=0).
    win._on_timeline_lock_toggled(1)
    QCoreApplication.processEvents()
    pinned = win._times_for(win._file_state.file_fields["tas_daily"])[0]

    # Drag the master cursor to sample #5 of pane 1's monthly axis.
    times = win._times_for(win._file_state.file_fields["tas_t"])
    win._timeline_strip.cursor_changed.emit(times[5])
    QCoreApplication.processEvents()

    pane1_track = win._timeline_strip._tracks[0]
    pane2_track = win._timeline_strip._tracks[1]
    assert pane1_track._cursor_t == times[5], \
        "unlocked track cursor must follow the master"
    assert pane2_track._cursor_t == pinned, \
        "locked track cursor must stay at the pinned datetime"


def test_unlocking_jumps_track_cursor_to_master(make_main_window):
    """Unlocking a previously-locked track snaps it back to the master cursor."""
    win = make_main_window()
    _setup_two_panes(win)
    win._select_pane(0)
    QCoreApplication.processEvents()

    win._on_timeline_lock_toggled(1)
    QCoreApplication.processEvents()
    times = win._times_for(win._file_state.file_fields["tas_t"])
    win._timeline_strip.cursor_changed.emit(times[3])
    QCoreApplication.processEvents()
    assert win._timeline_strip._tracks[1]._cursor_t != times[3]

    win._on_timeline_lock_toggled(1)         # unlock
    QCoreApplication.processEvents()
    assert win._timeline_strip._tracks[1]._cursor_t == times[3], \
        "unlocking must snap the track cursor back to the master"


def test_cursor_clears_value_column(make_main_window):
    """Empty-click / Escape clears the pick → value column hides."""
    win = make_main_window()
    cell = _setup_two_panes(win)
    win._select_pane(0)
    QCoreApplication.processEvents()
    win._on_pane_pick(0, cell, lon=0.0, lat=0.0)
    QCoreApplication.processEvents()
    assert win._timeline_strip._tracks[0].value_text != ""

    # Escape drops the pick + clears every track's value column.
    win._on_escape()
    QCoreApplication.processEvents()
    for track in win._timeline_strip._tracks:
        assert track.value_text == "", \
            "Escape must clear per-track value columns too"
