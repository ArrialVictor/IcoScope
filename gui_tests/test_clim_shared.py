"""Field-shared colour-limit cache: cross-pane comparability."""
from __future__ import annotations

from qtpy.QtCore import QCoreApplication


def _set_field(win, pane_idx: int, field: str) -> None:
    win._select_pane(pane_idx)        # synchronous; no events to drain
    win._on_color_by(field)
    QCoreApplication.processEvents()


def test_same_field_two_panes_share_clim(make_main_window):
    """Two panes both = tas_t → identical clim, even at different times."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")
    _set_field(win, 1, "tas_t")

    # Advance pane 0's time so the per-frame data ranges differ, but the
    # shared cache should still return the same global clim for both.
    win.state.panes[0].time_index = 0
    win.state.panes[1].time_index = 5
    win._refresh_scalars(0)
    win._refresh_scalars(1)

    clim_a = win._clim(0)
    clim_b = win._clim(1)
    assert clim_a is not None and clim_b is not None
    assert clim_a == clim_b, (
        f"shared mode: same field should give same clim across panes; "
        f"got {clim_a} vs {clim_b}")


def test_different_fields_different_clims(make_main_window):
    """Two panes = different fields → independent (field-keyed) clims."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")
    _set_field(win, 1, "tas_daily")

    clim_t = win._clim(0)
    clim_d = win._clim(1)
    assert clim_t is not None and clim_d is not None
    # Different fields → different ranges (data differs by construction).
    assert clim_t != clim_d


def test_symmetric_toggle_field_shared_in_shared_mode(make_main_window):
    """Shared mode: symmetric toggle writes to clim_symmetric; pane.center_zero is unused."""
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_anomaly")
    _set_field(win, 1, "tas_anomaly")

    win._select_pane(0)
    QCoreApplication.processEvents()
    win._on_center_zero(True)
    QCoreApplication.processEvents()

    # Field-shared dict updated (the sole source of truth in shared mode).
    assert win._file_state.clim_symmetric.get("tas_anomaly") is True
    # Both panes render with the same symmetric clim — they read from the
    # shared dict, not from their own pane.center_zero (which is dead
    # state in shared mode).
    c0 = win._clim(0)
    c1 = win._clim(1)
    assert c0 is not None and c1 is not None
    assert c0[0] == -c0[1] and c1[0] == -c1[1]
    assert c0 == c1


def test_toggle_shared_off_carries_back_symmetric_state(make_main_window):
    """Switching shared OFF copies the field-shared value back to pane.center_zero.

    Otherwise a previously-symmetric pane silently flips to autoscale
    when shared mode is disabled (pane.center_zero is stale from
    earlier color_by churn, but _clim now reads it).
    """
    win = make_main_window()
    win._on_pane_layout(2)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_anomaly")
    _set_field(win, 1, "tas_anomaly")

    # Make tas_anomaly symmetric in shared mode.
    win._on_center_zero(True)
    QCoreApplication.processEvents()
    # Briefly switch pane 1 to a different field and back — old code path
    # would clobber pane 1's pane.center_zero during the color_by change.
    _set_field(win, 1, "tas_t")
    _set_field(win, 1, "tas_anomaly")

    # Now toggle shared OFF. The carry-back must restore pane 1's
    # center_zero from the shared dict so the visual doesn't flip.
    win._on_clim_shared_toggled(False)
    QCoreApplication.processEvents()
    assert win.state.panes[0].center_zero is True
    assert win.state.panes[1].center_zero is True, \
        "shared OFF must carry the shared value back to pane.center_zero"


def test_toggle_shared_off_falls_back_to_per_pane(make_main_window):
    """With shared mode off, _clim returns None for non-symmetric panes."""
    win = make_main_window()
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")

    # Default: shared on → cached clim.
    assert win._clim(0) is not None

    # Toggle off → per-frame autoscale.
    win._on_clim_shared_toggled(False)
    QCoreApplication.processEvents()
    assert win._clim(0) is None, (
        "shared off + non-symmetric should return None (PyVista autoscale)")


def test_file_unload_clears_clim_cache(make_main_window):
    """Closing the file drops the cache so a new file doesn't see stale values."""
    win = make_main_window()
    win._on_pane_layout(1)
    QCoreApplication.processEvents()
    _set_field(win, 0, "tas_t")
    win._clim(0)        # forces compute + cache
    assert "tas_t" in win._file_state.clim_cache

    win._on_close_file()
    QCoreApplication.processEvents()
    assert win._file_state.clim_cache == {}
    assert win._file_state.clim_symmetric == {}
