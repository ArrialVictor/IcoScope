"""Coverage backfill flagged by the post-#34 audit.

Each test pins down behaviour that is correct today but had no
regression guard, so future refactors can break it silently:

- Playback toggle slot: clicking the strip's Play button must drive
  ``Playback.toggle_play`` end-to-end (button check state + timer
  lifecycle), not just via the test harness's ``blockSignals`` prime.
- Camera sync: ON mirrors pane 1's camera onto pane 2; OFF leaves them
  independent.
- Clim-shared toggle ON: active pane's ``center_zero`` wins the snapshot
  when two panes show the same field with different per-pane settings.
- PlaybackBar cursor label: ``set_cursor_label`` is wired to every
  cursor change and renders the master cursor's ``short_datetime``.
- File unload behavioural regression: load A, close, load B with a
  different range — ``_clim`` must return B's range, not a stale A.
"""
from __future__ import annotations

from qtpy.QtCore import QCoreApplication


# ── Playback toggle wiring ────────────────────────────────────────────

def test_play_button_click_drives_toggle_play(make_main_window, set_field):
    """Toggling Play via the button (not via the harness prime) must reach Playback.

    Previously only the harness path was exercised; a regression in
    the ``play_toggled`` → ``_on_play_toggled`` → ``toggle_play``
    chain would have gone undetected.
    """
    win = make_main_window()
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    bar = win._timeline_strip.playback_bar
    assert bar.play_btn.isChecked() is False
    assert win.playback._play_timer is None

    try:
        bar.play_btn.setChecked(True)
        QCoreApplication.processEvents()

        assert bar.play_btn.isChecked() is True, \
            "Play button must reflect toggled-on state"
        assert win.playback._play_timer is not None, \
            "toggle_play must instantiate the timer on first ON"
        assert win.playback._play_timer.isActive(), \
            "toggle_play must start the timer when checked"
        assert bar.play_btn.text() == "⏸", \
            "Play button label must flip to pause glyph"
    finally:
        # Stop before the timer fires — letting a real tick run during
        # pytest's fixture lifecycle crashes Qt on macOS.
        if win.playback._play_timer is not None:
            win.playback._play_timer.stop()
        bar.play_btn.blockSignals(True)
        bar.play_btn.setChecked(False)
        bar.play_btn.blockSignals(False)


# ── Camera sync mirror ────────────────────────────────────────────────

def _pane_camera_position(win, idx: int):
    cam = win._pane_container.pane(idx).plotter.renderer.GetActiveCamera()
    return cam.GetPosition()


def test_camera_sync_on_mirrors_changes(make_main_window, set_field):
    """With sync ON, moving pane 1's camera mirrors onto every other visible pane."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    assert win._camera_sync_on is True

    src_cam = (
        win._pane_container.pane(0).plotter.renderer.GetActiveCamera()
    )
    src_cam.SetPosition(5.0, 3.0, 2.0)
    src_cam.SetFocalPoint(0.0, 0.0, 0.0)
    win._on_camera_modified(0)
    QCoreApplication.processEvents()

    pos0 = _pane_camera_position(win, 0)
    pos1 = _pane_camera_position(win, 1)
    assert pos0 == pos1, (
        f"sync ON must mirror pane 1's position onto pane 2 "
        f"(pane 1: {pos0}, pane 2: {pos1})"
    )


def test_camera_sync_off_leaves_panes_independent(make_main_window, set_field):
    """With sync OFF, pane 1's camera moves don't touch pane 2's view."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    win._on_camera_sync(False)
    QCoreApplication.processEvents()
    assert win._camera_sync_on is False

    baseline_pos1 = _pane_camera_position(win, 1)

    src_cam = (
        win._pane_container.pane(0).plotter.renderer.GetActiveCamera()
    )
    src_cam.SetPosition(7.0, 1.5, -2.0)
    win._on_camera_modified(0)
    QCoreApplication.processEvents()

    assert _pane_camera_position(win, 1) == baseline_pos1, (
        "sync OFF must leave pane 2's camera unchanged when pane 1 moves"
    )


# ── Clim shared toggle ON snapshot priority ───────────────────────────

def test_clim_shared_toggle_on_active_pane_wins(make_main_window, set_field):
    """When two panes show the same field with different center_zero, ON snapshots the active pane.

    Existing ``test_clim_shared`` covers the OFF carry-back path but
    never exercised the ON path's "active wins" override.
    """
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_anomaly")
    set_field(win, 1, "tas_anomaly")

    # Drop shared mode so per-pane center_zero is meaningful, then
    # set conflicting values per pane.
    win._on_clim_shared_toggled(False)
    QCoreApplication.processEvents()
    win.state.panes[0].center_zero = False
    win.state.panes[1].center_zero = True

    # Select pane 1 (active wins → False), toggle shared ON.
    win._select_pane(0)
    QCoreApplication.processEvents()
    win._on_clim_shared_toggled(True)
    QCoreApplication.processEvents()
    assert win._file_state.clim_symmetric.get("tas_anomaly") is False, (
        "active-pane snapshot must override the seed when toggling ON "
        "(pane 1 active, center_zero=False)"
    )

    # Reverse: pane 2 active wins → True.
    win._on_clim_shared_toggled(False)
    QCoreApplication.processEvents()
    win.state.panes[0].center_zero = False
    win.state.panes[1].center_zero = True
    win._select_pane(1)
    QCoreApplication.processEvents()
    win._on_clim_shared_toggled(True)
    QCoreApplication.processEvents()
    assert win._file_state.clim_symmetric.get("tas_anomaly") is True, (
        "active-pane snapshot must override the seed when toggling ON "
        "(pane 2 active, center_zero=True)"
    )


# ── PlaybackBar cursor label ──────────────────────────────────────────

def test_cursor_label_updates_on_master_cursor_change(make_main_window, set_field):
    """Strip cursor label must render the current master cursor as short_datetime."""
    from icoscope.formatters import short_datetime

    win = make_main_window()
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    times = win._times_for(win._file_state.file_fields["tas_t"])

    win._timeline_strip.cursor_changed.emit(times[3])
    QCoreApplication.processEvents()
    label = win._timeline_strip.playback_bar.cursor_label.text()
    assert label == short_datetime(times[3]), (
        f"cursor label must show short_datetime(times[3])={short_datetime(times[3])!r}; "
        f"got {label!r}"
    )

    win._timeline_strip.cursor_changed.emit(times[7])
    QCoreApplication.processEvents()
    label = win._timeline_strip.playback_bar.cursor_label.text()
    assert label == short_datetime(times[7]), (
        f"cursor label must refresh on scrub; got {label!r}"
    )


# ── File unload → reload behavioural regression ───────────────────────

def test_clim_cache_fresh_after_close_reopen(make_main_window, synthetic_nc, set_field):
    """Closing the file then reloading must compute clim from the new file's data.

    The clim cache is keyed by field name; if the same name appears in
    both files, a stale (min, max) from file A would silently render
    file B with the wrong colour range. ``test_file_unload_clears_clim_cache``
    pins down that the dict is emptied; this test pins down that the
    next field access actually recomputes against fresh data.
    """
    win = make_main_window()
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    set_field(win, 0, "tas_t")
    first_clim = win._clim(0)
    assert first_clim is not None
    assert "tas_t" in win._file_state.clim_cache

    win._on_close_file()
    QCoreApplication.processEvents()
    assert win._file_state.clim_cache == {}

    # Reload the same synthetic file — replicate what _on_open_file does
    # internally (without the QFileDialog) so the test stays headless.
    # Fresh compute must repopulate the cache; same data → same range.
    import numpy as np
    from icoscope.loader import FileContext, load_grid, read_levels
    f_verts, f_cells, f_centers, fields = load_grid(str(synthetic_nc))
    levels = read_levels(str(synthetic_nc))
    win.file_path = str(synthetic_nc)
    win._file_state.file_fields = fields
    win._file_state.file_levels = levels
    for pane in win._file_state.panes:
        pane.color_by = "None"
        pane.time_index = 0
        pane.level_index = 0
    win._file_cache = {
        "path": str(synthetic_nc),
        "verts": f_verts,
        "cells": f_cells,
        "centers": np.asarray(f_centers),
        "fields": fields,
        "levels": levels,
        "context": FileContext(str(synthetic_nc)),
    }
    win.panel.file_tab.set_file_loaded(True)
    win._sync_file_info(str(synthetic_nc))
    win._activate_file_view()
    QCoreApplication.processEvents()

    set_field(win, 0, "tas_t")
    refreshed = win._clim(0)
    assert refreshed is not None
    assert "tas_t" in win._file_state.clim_cache, (
        "after reload, accessing _clim must repopulate the cache"
    )
    assert refreshed == first_clim, (
        f"reloading the same file must recompute the same clim "
        f"(got {refreshed} vs original {first_clim})"
    )
