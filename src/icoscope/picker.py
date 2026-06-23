"""Cell picking + trackpad pinch gestures for the main 3D view.

Holds the analytic ray-sphere intersection picker (more robust than
vtkCellPicker at the silhouette and poles) and the QPinchGesture →
camera roll / dolly translator. The picker reads the *current* mesh
from the window lazily so it keeps working after tab switches that
swap ``window._mesh``.
"""
import numpy as np
import pyvista as pv
from qtpy.QtCore import QEvent, QObject, Qt


class Picker(QObject):
    """Click-to-pick + trackpad-pinch handler for a :class:`QtInteractor`.

    Parameters
    ----------
    window
        The main window. Read for ``_mesh``, ``cells``, the lon/lat
        status widget setter (``_set_lonlat``), and to install the
        pinch-gesture event filter on the right Qt widget.
    plotter
        The PyVista ``QtInteractor`` to attach the picker to.
    """

    def __init__(self, window, plotter):
        super().__init__(window)
        self._window = window
        self._plotter = plotter
        self._cell_locator = None

    def attach(self) -> None:
        """Install the click-picker and the trackpad pinch event filter."""
        self._attach_picker()
        self._attach_trackpad_rotate()

    # ── highlight + clear ─────────────────────────
    def highlight_cell(self, idx: int) -> None:
        """Draw a magenta outline around cell ``idx`` on the sphere."""
        cell = list(self._window.cells[idx])
        pts = np.asarray(self._window.verts)[cell]
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 1.003
        n = len(cell)
        ids = list(range(n)) + [0]   # close the loop
        lines = np.array([len(ids)] + ids, dtype=np.int64)
        poly = pv.PolyData(pts, lines=lines)
        self._plotter.add_mesh(poly, name="highlight", color="#ff2d8a",
                               line_width=3.5, render_lines_as_tubes=False,
                               pickable=False, reset_camera=False)

    def clear_highlight(self) -> None:
        """Remove the highlight outline and clear lon/lat + value status widgets."""
        self._plotter.remove_actor("highlight", reset_camera=False, render=False)
        if hasattr(self._window, "lon_box"):
            self._window._clear_lonlat()
        if hasattr(self._window, "value_label"):
            self._window._clear_cell_value()

    def invalidate_locator(self) -> None:
        """Drop the cached vtkCellLocator (call after the mesh changes)."""
        self._cell_locator = None

    # ── picker ────────────────────────────────────
    def _attach_picker(self) -> None:
        """Wire up left-click → analytic ray-sphere → ``vtkCellLocator``.

        Bypasses vtkCellPicker entirely (which gets flaky at the silhouette
        and especially at the poles where graticule meridians converge).
        We cast a ray from the camera through the click pixel, intersect it
        analytically with the unit sphere (front hit), then ask a
        vtkCellLocator which polydata cell contains that point.
        """
        import vtk
        iren = getattr(self._plotter.iren, "interactor", self._plotter.iren)

        def ensure_locator():
            if self._cell_locator is None:
                loc = vtk.vtkCellLocator()
                loc.SetDataSet(self._window._mesh)
                loc.BuildLocator()
                self._cell_locator = loc
            return self._cell_locator

        def on_pick(point, *args, **kwargs):
            # Empty-sphere state — no cells to pick, and SetDataSet(None) on
            # the locator triggers a "No cells to subdivide" VTK error.
            if self._window._mesh is None:
                return
            x, y = iren.GetEventPosition()
            ren = self._plotter.renderer
            cam = ren.GetActiveCamera()
            cam_pos = np.array(cam.GetPosition(), dtype=float)

            # screen pixel → world ray direction
            ren.SetDisplayPoint(float(x), float(y), 0.0)
            ren.DisplayToWorld()
            p_world = np.array(ren.GetWorldPoint(), dtype=float)
            p_world = p_world[:3] / p_world[3] if p_world[3] != 0 else p_world[:3]
            ray = p_world - cam_pos
            n = np.linalg.norm(ray)
            if n == 0:
                self.clear_highlight()
                self._plotter.render()
                return
            ray /= n

            # intersect with unit sphere: |cam + t*ray|^2 = 1
            b = float(np.dot(cam_pos, ray))
            c = float(np.dot(cam_pos, cam_pos)) - 1.0
            disc = b * b - c
            if disc < 0 or (t := -b - np.sqrt(disc)) <= 0:
                # Ray misses the sphere (or front-hit is behind the camera) —
                # treat as an explicit deselect: clear highlight + lon/lat + value.
                self.clear_highlight()
                self._plotter.render()
                return
            hit = cam_pos + t * ray

            # find the cell containing this surface point
            loc = ensure_locator()
            closest = [0.0, 0.0, 0.0]
            cell_id = vtk.reference(0)
            sub_id = vtk.reference(0)
            dist2 = vtk.reference(0.0)
            gcell = vtk.vtkGenericCell()
            loc.FindClosestPoint(list(hit), closest, gcell, cell_id, sub_id, dist2)
            idx = int(cell_id)
            if idx < 0 or idx >= len(self._window.cells):
                self.clear_highlight()
                self._plotter.render()
                return

            self.highlight_cell(idx)
            hit /= np.linalg.norm(hit) or 1.0
            lat = float(np.degrees(np.arcsin(np.clip(hit[2], -1, 1))))
            lon = float(np.degrees(np.arctan2(hit[1], hit[0])))
            self._window._set_lonlat(lon, lat)
            if hasattr(self._window, "value_label"):
                self._window._set_cell_value(idx, lon=lon, lat=lat)
            self._plotter.render()

        self._plotter.enable_point_picking(
            callback=on_pick, left_clicking=True,
            show_message=False, show_point=False, pickable_window=False,
        )

    # ── trackpad pinch ────────────────────────────
    def _attach_trackpad_rotate(self) -> None:
        """Enable QPinchGesture on the interactor and install the event filter.

        Rotation isn't its own Qt gesture type — it's part of QPinchGesture,
        which exposes ``scaleFactor()`` and ``rotationAngle()``.
        """
        iren_widget = self._plotter.interactor
        iren_widget.grabGesture(Qt.GestureType.PinchGesture)
        iren_widget.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        """Translate trackpad pinch gestures into camera roll / dolly / pan."""
        if event.type() == QEvent.Type.Gesture:
            g = event.gesture(Qt.GestureType.PinchGesture)
            if g is not None:
                cf = g.changeFlags()
                vc = self._plotter.renderer.GetActiveCamera()
                need_render = False
                # rotation → roll the camera around its view axis
                if cf & g.ChangeFlag.RotationAngleChanged:
                    delta = g.rotationAngle() - g.lastRotationAngle()
                    if abs(delta) > 0:
                        vc.Roll(-delta)
                        need_render = True
                # pinch → dolly the camera along its view axis
                if cf & g.ChangeFlag.ScaleFactorChanged:
                    s = g.scaleFactor()  # per-step multiplicative delta
                    if s > 0 and s != 1.0:
                        vc.Dolly(s)
                        self._plotter.renderer.ResetCameraClippingRange()
                        need_render = True
                if need_render:
                    self._plotter.render()
                return True
        return super().eventFilter(obj, event)
