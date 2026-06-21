"""Auto-rotate spin timer + time-slider playback for the main window.

Both timers live here; the main window's slots delegate. The spin timer
rotates the camera around the global up vector; the play timer advances
the File-tab time slider, which re-fires the regular ``_on_time_changed``
path so there's no duplicate scalar-refresh logic.
"""
import numpy as np
from qtpy.QtCore import QTimer


class Playback:
    """Owns the spin and play QTimers on behalf of the main window.

    Parameters
    ----------
    window
        The main window. Used for parenting the timers and for accessing
        the File-tab time slider during playback.
    plotter
        The PyVista ``QtInteractor`` whose camera the spin tick rotates.
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
        """Advance the camera one step around the global up vector."""
        vc = self._plotter.renderer.GetActiveCamera()
        fp = np.array(vc.GetFocalPoint(), dtype=float)
        up = np.array(vc.GetViewUp(), dtype=float)
        up /= np.linalg.norm(up) or 1.0
        rel = np.array(vc.GetPosition(), dtype=float) - fp
        a = np.radians(0.4)
        ca, sa = np.cos(a), np.sin(a)
        new_rel = rel * ca + np.cross(up, rel) * sa + up * (up @ rel) * (1 - ca)
        new_pos = fp + new_rel
        vc.SetPosition(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
        self._plotter.render()

    # ── time-slider playback ──────────────────────
    def toggle_play(self, on: bool) -> None:
        """Start or stop the time-slider auto-advance timer."""
        if on:
            if self._play_timer is None:
                self._play_timer = QTimer(self._window)
                self._play_timer.setInterval(
                    self._window.panel.file_tab.display.speed_box.value())
                self._play_timer.timeout.connect(self._play_step)
            self._play_timer.start()
        else:
            if self._play_timer is not None:
                self._play_timer.stop()

    def set_speed(self, ms: int) -> None:
        """Update the play timer's interval (ms per step)."""
        if self._play_timer is not None:
            self._play_timer.setInterval(ms)

    def _play_step(self) -> None:
        """Bump the File-tab time slider one step; loop at the end."""
        w = self._window
        meta = w._file_state.file_fields.get(w._file_state.color_by)
        if not meta or not meta.get("time_varying"):
            self._play_timer.stop()
            w.panel.file_tab.display.play_btn.setChecked(False)
            return
        n = meta["shape"][0]
        new_idx = (w._file_state.time_index + 1) % n
        # Triggers _on_time_changed via the tab's time_changed signal.
        w.panel.file_tab.display.time_slider.setValue(new_idx)
