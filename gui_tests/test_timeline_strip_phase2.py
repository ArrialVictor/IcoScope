"""Timeline strip Phase 2 — per-track value display, lock, label-click.

Setup pattern
-------------
Every test in this module needs the same 2-pane File-tab configuration
(pane 0 = ``tas_t`` monthly, pane 1 = ``tas_daily``). The per-test cost
of building that — Qt window + 2× PyVista ``add_mesh`` for each field
swap — was 15-20 s before this refactor. The fixtures below share one
configured window for the whole module and reset per-test mutations
(cursor, lock state, pick state) before each test runs.

If a future test in this file needs a *different* setup, it can still
take ``make_main_window`` directly and bypass the module fixture; the
autouse reset only runs when ``phase2_setup`` is in the dependency
graph.
"""
from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import QCoreApplication


@pytest.fixture(scope="module")
def phase2_setup(make_module_window, set_field):
    """Module-scoped 2-pane window. Returns ``(win, cell_idx)``.

    Built once and shared across every test in this module. Per-test
    mutations are undone by the autouse :func:`_reset_phase2_state`
    fixture below.
    """
    win = make_module_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    set_field(win, 1, "tas_daily")
    # Equator-ish cell for a stable, comparable value.
    centers = np.asarray(win.centers)
    lats = np.degrees(np.arcsin(np.clip(centers[:, 2], -1, 1)))
    lons = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
    cell = int(np.argmin(lats ** 2 + lons ** 2))
    return win, cell


@pytest.fixture(autouse=True)
def _reset_phase2_state(phase2_setup):
    """Put the shared window back to pristine state before each test.

    Tests can mutate: pick state, pane selection, lock state on either
    pane, master cursor, per-pane ``time_index``. The reset undoes all
    of these without touching ``color_by`` (so the shared 2-pane field
    configuration survives).

    Done as ``yield``-before-body so a previous test's crash can't
    contaminate the current one — every test starts from a known
    reference state.
    """
    win, _ = phase2_setup
    # Clear pick + selection (also clears per-track value_text via
    # _refresh_timeline_pane_values).
    win._clear_pick_state(render=False, deselect_pane=True)
    # Unlock both panes if any test left them locked.
    for i in range(2):
        if win.state.panes[i].time_locked:
            win._on_timeline_lock_toggled(i)
    # Reset master cursor → clears banners + per-track cursor visuals.
    win._set_master_cursor(None)
    # _set_master_cursor doesn't touch time_index when cursor is None,
    # so reset the per-pane indices by hand.
    win.state.panes[0].time_index = 0
    win.state.panes[1].time_index = 0
    win._render_visible_panes()
    QCoreApplication.processEvents()
    yield


def test_pick_populates_value_column_on_every_track(phase2_setup):
    """A cell pick surfaces a per-pane value on every track's value column."""
    win, cell = phase2_setup
    win._select_pane(0)
    QCoreApplication.processEvents()

    # Empty until first pick.
    for track in win._timeline_strip.tracks:
        assert track.value_text == ""

    win._on_pane_pick(0, cell, lon=0.0, lat=0.0)
    QCoreApplication.processEvents()

    tracks = win._timeline_strip.tracks
    assert tracks[0].value_text, "pane 1 track value must populate on pick"
    assert tracks[1].value_text, "pane 2 track value must populate on pick"
    # Both fields are temperature-ish (~280-310 K) → both labels should
    # contain a "K" unit token. Belt-and-braces sanity check.
    assert "K" in tracks[0].value_text
    assert "K" in tracks[1].value_text


def test_lock_prevents_cursor_propagation(phase2_setup):
    """Locking pane 2 pins its time_index while the cursor advances."""
    win, _ = phase2_setup
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


def test_label_click_selects_pane(phase2_setup):
    """Clicking a track's label region routes to _select_pane."""
    win, _ = phase2_setup
    win._select_pane(0)
    QCoreApplication.processEvents()
    assert win._selected_pane == 0

    # The TimelineStrip emits pane_selected → connected to _on_pane_clicked.
    win._timeline_strip.pane_selected.emit(1)
    QCoreApplication.processEvents()
    assert win._selected_pane == 1


def test_lock_state_persists_across_layout_refresh(phase2_setup, set_field):
    """Rebuilding tracks (e.g. color_by change) must preserve lock visuals."""
    win, _ = phase2_setup
    win._on_timeline_lock_toggled(1)
    QCoreApplication.processEvents()
    assert win._timeline_strip.tracks[1].locked is True

    # Force a rebuild by re-setting pane 1's color_by — _refresh_timeline_strip
    # runs and re-creates the tracks if pane count changed (it didn't, but
    # the lock-restore loop runs unconditionally).
    set_field(win, 0, "tas_t")
    QCoreApplication.processEvents()
    assert win._timeline_strip.tracks[1].locked is True, \
        "lock visual must survive a strip rebuild"


def test_locked_track_cursor_stays_at_pinned_time(phase2_setup):
    """The cursor bar on a locked track must not follow the master cursor.

    Regression: previously the strip pushed one shared cursor to every
    track, so a locked pane's data stayed pinned but its cursor bar
    drifted with the master — confusingly out of sync with the data.
    """
    win, _ = phase2_setup
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

    pane1_track = win._timeline_strip.tracks[0]
    pane2_track = win._timeline_strip.tracks[1]
    assert pane1_track.cursor == times[5], \
        "unlocked track cursor must follow the master"
    assert pane2_track.cursor == pinned, \
        "locked track cursor must stay at the pinned datetime"


def test_unlocking_jumps_track_cursor_and_data_to_master(phase2_setup):
    """Unlocking snaps both the cursor visual AND the pane data to master."""
    win, _ = phase2_setup
    win._select_pane(0)
    QCoreApplication.processEvents()

    win._on_timeline_lock_toggled(1)
    QCoreApplication.processEvents()
    # While locked, drag the master cursor to monthly sample #1 (=day 30),
    # which inside pane 2's daily axis is index 30.
    times = win._times_for(win._file_state.file_fields["tas_t"])
    win._timeline_strip.cursor_changed.emit(times[1])
    QCoreApplication.processEvents()
    assert win.state.panes[1].time_index == 0, "locked pane stays at 0"
    assert win._timeline_strip.tracks[1].cursor != times[1]

    win._on_timeline_lock_toggled(1)         # unlock
    QCoreApplication.processEvents()
    assert win.state.panes[1].time_index == 30, \
        "unlocking must propagate the master cursor to time_index"
    assert win._timeline_strip.tracks[1].cursor == times[1], \
        "unlocking must snap the track cursor back to the master"


def test_cursor_clears_value_column(phase2_setup):
    """Empty-click / Escape clears the pick → value column hides."""
    win, cell = phase2_setup
    win._select_pane(0)
    QCoreApplication.processEvents()
    win._on_pane_pick(0, cell, lon=0.0, lat=0.0)
    QCoreApplication.processEvents()
    assert win._timeline_strip.tracks[0].value_text != ""

    # Escape drops the pick + clears every track's value column.
    win._on_escape()
    QCoreApplication.processEvents()
    for track in win._timeline_strip.tracks:
        assert track.value_text == "", \
            "Escape must clear per-track value columns too"
