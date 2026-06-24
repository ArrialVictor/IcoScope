"""Auto-rotate spin timer + time-cursor playback for the main window.

Both timers live here; the main window's slots delegate. The spin timer
rotates every visible pane's camera around its own up vector (kept in
lockstep so multi-pane comparison stays aligned); the play timer
advances the master time cursor at a configurable simulated-time-per-
real-time pace (e.g. "500 ms / day"). Each tick computes a small
cursor delta and reuses ``_set_master_cursor`` so multi-pane time
resolution + render happens through the same path as a slider drag.
"""
from datetime import timedelta

import numpy as np
from qtpy.QtCore import QTimer

from .timeline import PLAYBACK_UNIT_SECONDS

# Fixed internal tick rate — hides timer mechanics from the user. 50 ms
# (20 fps) is smooth enough for visual cursor movement without taxing
# the render path. The user-visible "Speed" control is in
# simulated-time-per-real-time units (ms per day / month / year), not
# this tick interval.
PLAYBACK_TICK_MS = 50


class Playback:
    """Owns the spin and play QTimers on behalf of the main window.

    Parameters
    ----------
    window
        The main window. Used for parenting the timers, for iterating
        visible panes during spin, and for accessing the File-tab time
        slider during playback.
    plotter
        The PyVista ``QtInteractor`` whose camera the spin tick rotates
        in single-pane mode (or used as a fallback when the pane
        container isn't yet wired).
    """

    def __init__(self, window, plotter):
        self._window = window
        self._plotter = plotter
        self._spin_timer = QTimer(window)
        self._spin_timer.setInterval(33)
        self._spin_timer.timeout.connect(self._spin_tick)
        self._play_timer: QTimer | None = None

    # ── auto-rotate (camera spin) ─────────────────
    def start_spin(self) -> None:
        """Start the 30 fps camera-rotation timer."""
        self._spin_timer.start()

    def stop_spin(self) -> None:
        """Stop the camera-rotation timer."""
        self._spin_timer.stop()

    def _spin_tick(self) -> None:
        """Advance every visible pane's camera one step around its up vector.

        In multi-pane mode this rotates all visible panes simultaneously
        so the comparison stays geometrically aligned. Each pane uses its
        own camera (cameras aren't truly linked yet — that's the next PR
        in the multi-pane series), but they all receive the same angular
        delta and start from the same configured camera, so they stay
        visually in step.
        """
        container = getattr(self._window, "_pane_container", None)
        plotters = (
            [container.pane(i).plotter for i in range(container.n_visible)]
            if container is not None
            else [self._plotter]
        )
        a = np.radians(0.4)
        ca, sa = np.cos(a), np.sin(a)
        # Direct camera mutations below fire VTK's ModifiedEvent observers
        # which the window uses for camera sync — without the guard, each
        # pane's tick would try to mirror itself onto the others, doing
        # N*(N-1) cross-camera writes per tick. Same final state, just
        # wasteful. Suppress for the duration of the per-pane update.
        prev_sync = getattr(self._window, "_syncing_cameras", False)
        self._window._syncing_cameras = True
        for plotter in plotters:
            vc = plotter.renderer.GetActiveCamera()
            fp = np.array(vc.GetFocalPoint(), dtype=float)
            up = np.array(vc.GetViewUp(), dtype=float)
            up /= np.linalg.norm(up) or 1.0
            rel = np.array(vc.GetPosition(), dtype=float) - fp
            new_rel = rel * ca + np.cross(up, rel) * sa + up * (up @ rel) * (1 - ca)
            new_pos = fp + new_rel
            vc.SetPosition(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
            plotter.render()
        self._window._syncing_cameras = prev_sync

    # ── time-cursor playback ──────────────────────
    def toggle_play(self, on: bool) -> None:
        """Start or stop the cursor-advance timer."""
        if on:
            if self._play_timer is None:
                self._play_timer = QTimer(self._window)
                self._play_timer.setInterval(PLAYBACK_TICK_MS)
                self._play_timer.timeout.connect(self._play_step)
            self._play_timer.start()
        else:
            if self._play_timer is not None:
                self._play_timer.stop()

    def _stop_and_uncheck(self) -> None:
        """End playback + reset the PlaybackBar's play-button visual."""
        if self._play_timer is not None:
            self._play_timer.stop()
        self._window._timeline_strip.playback_bar.set_playing(False)

    def _play_step(self) -> None:
        """Advance the master cursor by one stride.

        Stride = ``(PLAYBACK_TICK_MS / speed_ms) * seconds_per_unit``,
        where ``speed_ms`` is the user-set "ms per day/month/year" value.
        End-of-axis (cursor past the union of all visible panes' axes):
        loop on → wrap to the union's first sample; loop off → stop +
        uncheck the PlaybackBar's play button.
        """
        w = self._window
        state = w._file_state
        strip = w._timeline_strip
        t0, t1 = strip._domain_t0, strip._domain_t1
        if t0 is None or t1 is None:
            # No time-varying field visible — nothing to play.
            self._stop_and_uncheck()
            return

        cursor = state.time_cursor
        if cursor is None:
            # First tick after pressing play with no prior scrub —
            # initialise at the union's first sample.
            w._set_master_cursor(t0)
            return

        seconds_per_unit = PLAYBACK_UNIT_SECONDS.get(
            state.playback_speed_unit, PLAYBACK_UNIT_SECONDS["day"])
        units_per_tick = PLAYBACK_TICK_MS / max(state.playback_speed_value, 1)
        seconds_per_tick = units_per_tick * seconds_per_unit
        new_cursor = cursor + timedelta(seconds=seconds_per_tick)

        if new_cursor > t1:
            # Land exactly on the last frame, then stop so the user sees
            # where the simulation ended. (Loop-and-restart was removed
            # at user request — climate researchers prefer the explicit
            # end-of-data signal.)
            w._set_master_cursor(t1)
            self._stop_and_uncheck()
            return

        w._set_master_cursor(new_cursor)
        w._build_scene()
