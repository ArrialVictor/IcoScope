"""Qt main window: 3D viewer + control panel + status bar."""
import os
import sys
from datetime import datetime

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy.QtCore import QEvent, Qt, QTimer
from qtpy.QtGui import QIcon, QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from .coastlines import coastline_polydata
from .controls import ControlPanel
from .graticule import graticule_polydata
from .grid import goldberg
from .themes import CMAPS, THEMES

# Color-by options shown when no NetCDF is loaded. Once a file is loaded, the
# dropdown is replaced by "None" + the file's own field names so the
# physicist's data isn't buried among synthetic demo fields.
SYNTHETIC_COLOR_BY = [
    "None", "Latitude", "Cell kind", "Mock temperature", "Realistic temperature",
]


class MainWindow(QMainWindow):
    """Top-level IcoScope window: 3D sphere view + right-side control panel."""

    def __init__(self, verts, cells, centers, initial_n=8, relax=True,
                 zoom_factor=1.0, zoom_lon=0.0, zoom_lat=45.0):
        super().__init__()
        self.setWindowTitle("IcoScope")
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.resize(1320, 840)

        # state
        self.verts = verts
        self.cells = cells
        self.centers = np.asarray(centers)
        self.scalars = None           # what we actually render
        self.color_by = "None"
        self.file_path = None
        self.file_fields = {}         # {name: metadata} loaded from NetCDF
        self.time_index = 0
        self.n = initial_n
        self.max_relax_iters = 200 if relax else 0
        self.zoom_factor = float(zoom_factor)
        self.zoom_lon = float(zoom_lon)
        self.zoom_lat = float(zoom_lat)
        self.transparent_export = False

        self.theme_name = "Dark"
        self.cmap = THEMES[self.theme_name]["cmap"]
        self.coastlines_on = False
        self.graticule_on = False
        self.edges_on = True
        self.colorbar_on = True
        self.center_zero = False
        self.spin_on = False
        self.edge_color_override = None
        self.coast_color_override = None
        self.grat_color_override = None
        self.edge_width = 0.6
        self.coast_width = 1.2
        self.grat_width = 0.6

        # central layout
        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(central)
        h.addWidget(self.plotter.interactor, stretch=1)
        self.panel = ControlPanel(list(THEMES), CMAPS)
        h.addWidget(self.panel)
        self.setCentralWidget(central)

        self._build_menubar()
        self._build_lonlat_widget()
        self.statusBar().showMessage("ready")

        # initial sync of panel state
        self.panel.n_box.blockSignals(True)
        self.panel.n_box.setValue(initial_n)
        self.panel.n_box.blockSignals(False)
        self.panel.relax_iters_box.blockSignals(True)
        self.panel.relax_iters_box.setValue(self.max_relax_iters)
        self.panel.relax_iters_box.blockSignals(False)
        self.panel.set_zoom(self.zoom_factor, self.zoom_lon, self.zoom_lat)
        self.panel.set_theme(self.theme_name)
        self.panel.set_cmap(self.cmap)
        self._sync_color_buttons()
        # color-by initially "None" → disable cmap-related widgets
        self.panel.center_cb.setEnabled(False)
        self.panel.bar_cb.setEnabled(False)
        self.panel.cmap_box.setEnabled(False)

        # wire signals
        self.panel.theme_changed.connect(self._on_theme)
        self.panel.cmap_changed.connect(self._on_cmap)
        self.panel.coastlines_toggled.connect(self._on_coast)
        self.panel.graticule_toggled.connect(self._on_grat)
        self.panel.edges_toggled.connect(self._on_edges)
        self.panel.colorbar_toggled.connect(self._on_colorbar)
        self.panel.color_by_changed.connect(self._on_color_by)
        self.panel.center_zero_toggled.connect(self._on_center_zero)
        self.panel.edge_color_changed.connect(self._on_edge_color)
        self.panel.coast_color_changed.connect(self._on_coast_color)
        self.panel.grat_color_changed.connect(self._on_grat_color)
        self.panel.edge_width_changed.connect(self._on_edge_width)
        self.panel.coast_width_changed.connect(self._on_coast_width)
        self.panel.grat_width_changed.connect(self._on_grat_width)
        self.panel.autorotate_toggled.connect(self._on_spin)
        self.panel.n_changed.connect(self._on_n)
        self.panel.relax_iters_changed.connect(self._on_relax_iters)
        self.panel.zoom_changed.connect(self._on_zoom)
        self.panel.open_file_clicked.connect(self._on_open_file)
        self.panel.close_file_clicked.connect(self._on_close_file)
        self.panel.time_changed.connect(self._on_time_changed)
        self.panel.play_toggled.connect(self._on_play_toggled)
        self.panel.play_speed_changed.connect(self._on_play_speed_changed)
        self.panel.screenshot_clicked.connect(self._on_screenshot)
        self.panel.vector_export_clicked.connect(self._on_vector_export)

        # build scene + interactions
        self._refresh_scalars()
        self._mesh = self._to_polydata()
        self._cell_locator = None
        self._build_scene()
        self.plotter.reset_camera()
        self._attach_picker()
        self._attach_trackpad_rotate()
        self._apply_stylesheet()
        QShortcut(QKeySequence("Escape"), self, activated=self._on_escape)
        self._attach_spin_timer()
        self._update_status()

    # ── colors ─────────────────────────────────────
    def _edge_color(self):
        return self.edge_color_override or THEMES[self.theme_name]["edge"]

    def _coast_color(self):
        return self.coast_color_override or THEMES[self.theme_name]["coast"]

    def _grat_color(self):
        return self.grat_color_override or THEMES[self.theme_name].get(
            "grat", self._coast_color())

    def _sync_color_buttons(self):
        self.panel.set_edge_color(self._color_to_hex(self._edge_color()))
        self.panel.set_coast_color(self._color_to_hex(self._coast_color()))
        self.panel.set_grat_color(self._color_to_hex(self._grat_color()))

    @staticmethod
    def _color_to_hex(c):
        # accept "#rrggbb" or named CSS colors
        from qtpy.QtGui import QColor
        return QColor(c).name()

    # ── geometry ──────────────────────────────────
    def _to_polydata(self):
        faces_flat = []
        for c in self.cells:
            faces_flat.append(len(c))
            faces_flat.extend(c)
        mesh = pv.PolyData(self.verts, faces=np.array(faces_flat, dtype=np.int64))
        if self.scalars is not None:
            mesh.cell_data["scalars"] = np.asarray(self.scalars)
        return mesh

    def _clim(self):
        if self.scalars is None or not self.center_zero:
            return None
        s = np.asarray(self.scalars)
        a = float(np.nanmax(np.abs(s)))
        return [-a, a] if a > 0 else None

    # ── rendering ─────────────────────────────────
    def _build_scene(self):
        theme = THEMES[self.theme_name]
        self.plotter.set_background(theme["bg"])

        self.plotter.add_mesh(
            self._mesh, name="grid",
            scalars="scalars" if self.scalars is not None else None,
            cmap=self.cmap,
            clim=self._clim(),
            show_edges=self.edges_on,
            edge_color=self._edge_color(),
            line_width=self.edge_width,
            smooth_shading=False,
            show_scalar_bar=self.colorbar_on and self.scalars is not None,
            reset_camera=False,
        )

        if self.coastlines_on:
            try:
                cl = coastline_polydata(radius=1.001)
                self.plotter.add_mesh(cl, name="coast",
                                      color=self._coast_color(),
                                      line_width=self.coast_width,
                                      pickable=False, reset_camera=False)
            except Exception as e:
                self.statusBar().showMessage(f"coastlines failed: {e}")
        else:
            self.plotter.remove_actor("coast", reset_camera=False, render=False)

        if self.graticule_on:
            try:
                g = graticule_polydata(radius=1.002, spacing=30)
                self.plotter.add_mesh(g, name="grat",
                                      color=self._grat_color(),
                                      line_width=self.grat_width,
                                      opacity=0.6, pickable=False, reset_camera=False)
            except Exception as e:
                self.statusBar().showMessage(f"graticule failed: {e}")
        else:
            self.plotter.remove_actor("grat", reset_camera=False, render=False)

        self.plotter.render()

    # ── ESC: clear current selection + stop spin ──
    def _on_escape(self):
        self._clear_highlight()
        self._clear_lonlat()
        if self.spin_on:
            self.spin_on = False
            self._spin_timer.stop()
            self.panel.spin_cb.blockSignals(True)
            self.panel.spin_cb.setChecked(False)
            self.panel.spin_cb.blockSignals(False)
        self.plotter.render()

    # ── Qt stylesheet for the panel ───────────────
    def _apply_stylesheet(self):
        self.setStyleSheet("""
        QGroupBox {
            font-weight: 600;
            border: 1px solid #888;
            border-radius: 5px;
            margin-top: 12px;
            padding: 6px 4px 4px 4px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QPushButton {
            padding: 4px 10px;
        }
        QLabel, QCheckBox, QComboBox, QSpinBox {
            padding: 1px 2px;
        }
        QStatusBar {
            border-top: 1px solid #888;
        }
        """)

    # ── highlight picked cell ─────────────────────
    def _highlight_cell(self, idx):
        cell = list(self.cells[idx])
        pts = np.asarray(self.verts)[cell]
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 1.003
        n = len(cell)
        ids = list(range(n)) + [0]   # close the loop
        lines = np.array([len(ids)] + ids, dtype=np.int64)
        poly = pv.PolyData(pts, lines=lines)
        self.plotter.add_mesh(poly, name="highlight", color="#ff2d8a",
                              line_width=3.5, render_lines_as_tubes=False,
                              pickable=False, reset_camera=False)

    def _clear_highlight(self):
        self.plotter.remove_actor("highlight", reset_camera=False, render=False)
        if hasattr(self, "lon_box"):
            self._clear_lonlat()

    # ── picker (click → cell info) ────────────────
    def _attach_picker(self):
        """Left-click → analytic ray-sphere intersection → vtkCellLocator.

        Bypasses vtkCellPicker entirely (which gets flaky at the silhouette
        and especially at the poles where graticule meridians converge).
        We cast a ray from the camera through the click pixel, intersect it
        analytically with the unit sphere (front hit), then ask a
        vtkCellLocator which polydata cell contains that point.
        """
        import vtk
        iren = getattr(self.plotter.iren, "interactor", self.plotter.iren)
        self._cell_locator = None

        def ensure_locator():
            if self._cell_locator is None:
                loc = vtk.vtkCellLocator()
                loc.SetDataSet(self._mesh)
                loc.BuildLocator()
                self._cell_locator = loc
            return self._cell_locator

        def on_pick(point, *args, **kwargs):
            x, y = iren.GetEventPosition()
            ren = self.plotter.renderer
            cam = ren.GetActiveCamera()
            cam_pos = np.array(cam.GetPosition(), dtype=float)

            # screen pixel → world ray direction
            ren.SetDisplayPoint(float(x), float(y), 0.0)
            ren.DisplayToWorld()
            p_world = np.array(ren.GetWorldPoint(), dtype=float)
            if p_world[3] != 0:
                p_world = p_world[:3] / p_world[3]
            else:
                p_world = p_world[:3]
            ray = p_world - cam_pos
            n = np.linalg.norm(ray)
            if n == 0:
                return
            ray /= n

            # intersect with unit sphere: |cam + t*ray|^2 = 1
            b = float(np.dot(cam_pos, ray))
            c = float(np.dot(cam_pos, cam_pos)) - 1.0
            disc = b * b - c
            if disc < 0:
                return                 # ray misses sphere
            t = -b - np.sqrt(disc)     # closer (front) intersection
            if t <= 0:
                return                 # behind camera
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
            if idx < 0 or idx >= len(self.cells):
                return

            self._highlight_cell(idx)
            hit /= np.linalg.norm(hit) or 1.0
            lat = float(np.degrees(np.arcsin(np.clip(hit[2], -1, 1))))
            lon = float(np.degrees(np.arctan2(hit[1], hit[0])))
            self._set_lonlat(lon, lat)
            self.plotter.render()

        self.plotter.enable_point_picking(
            callback=on_pick, left_clicking=True,
            show_message=False, show_point=False, pickable_window=False,
        )

    # ── trackpad rotate gesture (macOS two-finger rotate) ──
    def _attach_trackpad_rotate(self):
        """Map a trackpad pinch gesture's rotation component to a camera roll.

        Rotation isn't its own Qt gesture type — it's part of QPinchGesture,
        which exposes scaleFactor() and rotationAngle().
        """
        iren_widget = self.plotter.interactor
        iren_widget.grabGesture(Qt.GestureType.PinchGesture)
        iren_widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Translate trackpad pinch gestures into camera roll / dolly / pan."""
        if event.type() == QEvent.Type.Gesture:
            g = event.gesture(Qt.GestureType.PinchGesture)
            if g is not None:
                cf = g.changeFlags()
                vc = self.plotter.renderer.GetActiveCamera()
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
                        self.plotter.renderer.ResetCameraClippingRange()
                        need_render = True
                if need_render:
                    self.plotter.render()
                return True
        return super().eventFilter(obj, event)

    # ── auto-rotate ───────────────────────────────
    def _attach_spin_timer(self):
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(33)
        self._spin_timer.timeout.connect(self._spin_tick)

    def _spin_tick(self):
        vc = self.plotter.renderer.GetActiveCamera()
        fp = np.array(vc.GetFocalPoint(), dtype=float)
        up = np.array(vc.GetViewUp(), dtype=float)
        up /= np.linalg.norm(up) or 1.0
        rel = np.array(vc.GetPosition(), dtype=float) - fp
        a = np.radians(0.4)
        ca, sa = np.cos(a), np.sin(a)
        new_rel = rel * ca + np.cross(up, rel) * sa + up * (up @ rel) * (1 - ca)
        new_pos = fp + new_rel
        vc.SetPosition(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
        self.plotter.render()

    # ── status bar ────────────────────────────────
    def _update_status(self):
        if self.file_path:
            msg = f"file: {os.path.basename(self.file_path)}"
        else:
            msg = ""
        self.statusBar().showMessage(msg)

    # ── lon/lat editable widget on the status bar ─
    LON_SENTINEL = -200.0
    LAT_SENTINEL = -100.0

    def _build_lonlat_widget(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 0, 4, 0)
        h.setSpacing(4)
        h.addWidget(QLabel("lon"))
        self.lon_box = QDoubleSpinBox()
        self.lon_box.setRange(self.LON_SENTINEL, 180)
        self.lon_box.setDecimals(2)
        self.lon_box.setSuffix("°")
        self.lon_box.setSpecialValueText("—")
        self.lon_box.setFixedWidth(80)
        self.lon_box.setKeyboardTracking(False)
        self.lon_box.setValue(self.LON_SENTINEL)
        h.addWidget(self.lon_box)
        h.addWidget(QLabel("lat"))
        self.lat_box = QDoubleSpinBox()
        self.lat_box.setRange(self.LAT_SENTINEL, 90)
        self.lat_box.setDecimals(2)
        self.lat_box.setSuffix("°")
        self.lat_box.setSpecialValueText("—")
        self.lat_box.setFixedWidth(80)
        self.lat_box.setKeyboardTracking(False)
        self.lat_box.setValue(self.LAT_SENTINEL)
        h.addWidget(self.lat_box)
        self.statusBar().addPermanentWidget(w)
        # editing either field flies the camera
        self.lon_box.editingFinished.connect(self._fly_to_lonlat)
        self.lat_box.editingFinished.connect(self._fly_to_lonlat)

    def _set_lonlat(self, lon=None, lat=None):
        for box, val, sentinel in (
            (self.lon_box, lon, self.LON_SENTINEL),
            (self.lat_box, lat, self.LAT_SENTINEL),
        ):
            box.blockSignals(True)
            box.setValue(sentinel if val is None else float(val))
            box.blockSignals(False)

    def _clear_lonlat(self):
        self._set_lonlat(None, None)

    def _fly_to_lonlat(self):
        lon_v = self.lon_box.value()
        lat_v = self.lat_box.value()
        if lon_v == self.LON_SENTINEL or lat_v == self.LAT_SENTINEL:
            return
        lon = np.radians(lon_v)
        lat = np.radians(lat_v)
        direction = np.array([np.cos(lat) * np.cos(lon),
                              np.cos(lat) * np.sin(lon),
                              np.sin(lat)])
        cam = self.plotter.camera
        pos = np.asarray(cam.position, dtype=float)
        dist = np.linalg.norm(pos) or 3.0
        cam.position = (direction * dist).tolist()
        cam.focal_point = (0.0, 0.0, 0.0)
        cam.up = (0.0, 0.0, 1.0)
        self.plotter.render()

    # ── menubar ────────────────────────────────────
    def _build_menubar(self):
        mb = self.menuBar()

        help_menu = mb.addMenu("&Help")

        keys_act = QAction("Keyboard && mouse", self)
        keys_act.triggered.connect(self._show_shortcuts)
        help_menu.addAction(keys_act)

        netcdf_act = QAction("NetCDF help", self)
        netcdf_act.triggered.connect(self._show_netcdf_help)
        help_menu.addAction(netcdf_act)

        help_menu.addSeparator()
        about_act = QAction("About IcoScope", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _show_shortcuts(self):
        QMessageBox.information(
            self, "Keyboard & mouse",
            "<b>Mouse</b><br>"
            "Left drag — rotate<br>"
            "Right drag / scroll — zoom<br>"
            "Shift + drag — pan<br>"
            "Left click — pick a cell<br>"
            "<br><b>Keys</b><br>"
            "Esc — clear selection, stop auto-rotate<br>"
            "r — reset camera<br>"
            "f — focus on cursor<br>"
            "w / s — wireframe / surface<br>"
            "q — quit<br>"
            "<br><b>lon / lat fields</b><br>"
            "Type values + Enter to fly the camera there."
        )

    def _show_netcdf_help(self):
        QMessageBox.information(
            self, "NetCDF help",
            "<b>Expected schema (CF convention):</b><br>"
            "<code>lon(cell)</code>, <code>lat(cell)</code><br>"
            "<code>bounds_lon(cell, nvertex)</code>, "
            "<code>bounds_lat(cell, nvertex)</code><br>"
            "<code>&lt;field&gt;(cell)</code> or "
            "<code>&lt;field&gt;(time, cell)</code><br><br>"
            "Common variable-name variants are auto-detected "
            "(<code>lon</code>/<code>longitude</code>/<code>clon</code>, etc.). "
            "Pentagons should pad the last <code>nvertex</code> slot with a "
            "repeated vertex.<br><br>"
            "If your file fails to load, run "
            "<code>icoscope --file &lt;path&gt; --describe</code> to inspect "
            "the schema."
        )

    def _show_about(self):
        QMessageBox.about(
            self, "IcoScope",
            "<b>IcoScope</b><br>"
            "Interactive 3D viewer for icosahedral hex/pent grids on a sphere.<br><br>"
            "Renders DYNAMICO/ICOLMDZ NetCDF output or synthetic Goldberg grids.<br>"
            "Built with PyVista + Qt."
        )

    # ── slots ─────────────────────────────────────
    def _on_theme(self, name):
        self.theme_name = name
        suggested = THEMES[name]["cmap"]
        if suggested != self.cmap:
            self.cmap = suggested
            self.panel.set_cmap(suggested)
        # if user hasn't overridden, refresh swatches
        self._sync_color_buttons()
        self._build_scene()
        self._update_status()

    def _on_cmap(self, name):
        self.cmap = name
        self._build_scene()

    def _on_coast(self, on):
        self.coastlines_on = on
        self._build_scene()

    def _on_grat(self, on):
        self.graticule_on = on
        self._build_scene()

    def _on_edges(self, on):
        self.edges_on = on
        self._build_scene()

    def _refresh_scalars(self):
        """Compute self.scalars from the current `color_by` choice."""
        if self.color_by == "Latitude":
            self.scalars = np.degrees(np.arcsin(self.centers[:, 2]))
        elif self.color_by == "Cell kind":
            self.scalars = np.array([0 if len(c) == 5 else 1 for c in self.cells],
                                    dtype=float)
        elif self.color_by == "Mock temperature":
            # Clean synthetic field: latitude gradient + a faint zonal wave.
            # Same formula as the `tas` field in tools/make_test_nc.py.
            c = self.centers / np.linalg.norm(self.centers, axis=1, keepdims=True)
            lat = np.arcsin(np.clip(c[:, 2], -1, 1))
            lon = np.arctan2(c[:, 1], c[:, 0])
            self.scalars = 250.0 + 50.0 * np.cos(lat) + 5.0 * np.cos(2 * lon)
        elif self.color_by == "Realistic temperature":
            # Earth-like surface-temperature mock: base latitudinal gradient,
            # plus broad Gaussian "hot spots" over major land masses (Sahara,
            # Arabia, Australia, North-American interior) and cold spots
            # (Greenland, Antarctic interior). Visually convincing without
            # any real climate data.
            c = self.centers / np.linalg.norm(self.centers, axis=1, keepdims=True)
            lat = np.arcsin(np.clip(c[:, 2], -1, 1))
            lon = np.arctan2(c[:, 1], c[:, 0])

            def gauss(lat_c, lon_c, amp, sigma_deg):
                dlat = lat - np.radians(lat_c)
                # angular wrap-around in longitude
                dlon = (lon - np.radians(lon_c) + np.pi) % (2 * np.pi) - np.pi
                # great-circle-ish distance (cheap approximation)
                d2 = dlat ** 2 + (dlon * np.cos(lat)) ** 2
                return amp * np.exp(-d2 / (2 * np.radians(sigma_deg) ** 2))

            # base latitudinal profile, ~245 K at poles to ~300 K near equator
            T = 275.0 + 30.0 * np.cos(lat)

            # land hot spots (degrees: lat, lon, amplitude K, sigma deg)
            T += gauss( 23,  10, +14, 14)   # Sahara
            T += gauss( 25,  48, +10, 12)   # Arabia
            T += gauss(-25, 135, + 9, 14)   # Australian outback
            T += gauss( 35, -105, +6, 15)   # SW USA / Mexico
            T += gauss(-20, -60, + 6, 15)   # interior South America

            # cold spots
            T += gauss( 75, -40, -22, 14)   # Greenland
            T += gauss(-82,   0, -32, 22)   # Antarctic interior
            T += gauss( 65,  100, -8, 18)   # Siberian winter (mild proxy)

            self.scalars = T
        elif self.color_by in self.file_fields and self.file_path:
            from .loader import read_field
            self.scalars = read_field(self.file_path, self.color_by,
                                      time_index=self.time_index)
        else:
            self.scalars = None

    def _on_color_by(self, name):
        self.color_by = name
        self.panel.center_cb.setEnabled(name != "None")
        self.panel.bar_cb.setEnabled(name != "None")
        self.panel.cmap_box.setEnabled(name != "None")
        # configure the time slider if this is a time-varying field
        meta = self.file_fields.get(name)
        if meta and meta.get("time_varying"):
            self.panel.set_time_steps(meta["shape"][0])
            self.time_index = 0
        else:
            self.panel.set_time_steps(0)
            self.time_index = 0
        self._refresh_scalars()
        self._mesh = self._to_polydata()
        self._cell_locator = None
        self._build_scene()

    def _on_colorbar(self, on):
        self.colorbar_on = on
        self._build_scene()

    def _on_center_zero(self, on):
        self.center_zero = on
        self._build_scene()

    def _on_edge_color(self, hex_str):
        self.edge_color_override = hex_str
        self._build_scene()

    def _on_coast_color(self, hex_str):
        self.coast_color_override = hex_str
        self._build_scene()

    def _on_grat_color(self, hex_str):
        self.grat_color_override = hex_str
        self._build_scene()

    def _on_edge_width(self, w):
        self.edge_width = float(w)
        self._build_scene()

    def _on_coast_width(self, w):
        self.coast_width = float(w)
        self._build_scene()

    def _on_grat_width(self, w):
        self.grat_width = float(w)
        self._build_scene()

    def _regen_synthetic(self):
        relax = self.max_relax_iters > 0
        v, c, ctr, _ = goldberg(n=self.n, relax=relax,
                                max_iterations=self.max_relax_iters,
                                zoom_factor=self.zoom_factor,
                                zoom_lon=self.zoom_lon,
                                zoom_lat=self.zoom_lat)
        self.verts, self.cells, self.centers = v, c, np.asarray(ctr)
        self._refresh_scalars()
        self._mesh = self._to_polydata()
        self._cell_locator = None
        self._clear_highlight()
        self._build_scene()
        self._update_status()

    def _on_n(self, n):
        if self.file_path:
            return
        self.n = n
        self._regen_synthetic()

    def _on_relax_iters(self, n):
        if self.file_path:
            return
        self.max_relax_iters = n
        self._regen_synthetic()

    def _on_zoom(self, factor, lon, lat):
        if self.file_path:
            return
        self.zoom_factor = float(factor)
        self.zoom_lon = float(lon)
        self.zoom_lat = float(lat)
        self._regen_synthetic()

    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open NetCDF", "", "NetCDF (*.nc *.nc4 *.cdf);;All files (*)"
        )
        if not path:
            return
        try:
            from .loader import load_grid
            verts, cells, centers, fields = load_grid(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            return
        self.file_path = path
        self.verts, self.cells, self.centers = verts, cells, np.asarray(centers)
        self.file_fields = fields
        # File loaded → restrict Color by to None + the file's actual fields.
        items = ["None"] + list(fields.keys())
        self.panel.set_color_by_items(items)
        if fields:
            first = next(iter(fields.keys()))
            self.panel.set_color_by(first)
            self.color_by = first
        else:
            self.color_by = "None"
        self._refresh_scalars()
        self._mesh = self._to_polydata()
        self._cell_locator = None
        self._clear_highlight()
        self.panel.disable_n(True)
        self._sync_file_info(path)
        self._build_scene()
        self._update_status()

    def _sync_file_info(self, path: str):
        """Populate the File tab summary from the currently-loaded file."""
        from .loader import read_global_attrs
        n_time = max(
            (meta["shape"][0] for meta in self.file_fields.values()
             if meta.get("time_varying")),
            default=0,
        )
        try:
            attrs = read_global_attrs(path)
        except Exception:
            attrs = {}
        self.panel.set_file_info(
            path=path,
            n_cells=len(self.cells),
            n_fields=len(self.file_fields),
            n_time_steps=n_time,
            attrs=attrs,
        )

    def _on_close_file(self):
        """Drop the loaded file and return to the synthetic grid."""
        if not self.file_path:
            return
        self.file_path = None
        self.file_fields = {}
        self.panel.set_color_by_items(SYNTHETIC_COLOR_BY)
        self.color_by = "None"
        self.panel.set_time_steps(0)
        self.panel.disable_n(False)
        self.panel.set_file_info()
        self._regen_synthetic()

    def _on_time_changed(self, idx):
        if idx == self.time_index:
            return
        self.time_index = idx
        meta = self.file_fields.get(self.color_by)
        if meta:
            self.panel.set_time_label(idx, meta["shape"][0])
        self._refresh_scalars()
        self._mesh = self._to_polydata()
        self._cell_locator = None
        self._build_scene()

    def _on_play_toggled(self, on):
        if on:
            if not hasattr(self, "_play_timer"):
                self._play_timer = QTimer(self)
                self._play_timer.setInterval(self.panel.speed_box.value())
                self._play_timer.timeout.connect(self._play_step)
            self._play_timer.start()
        else:
            if hasattr(self, "_play_timer"):
                self._play_timer.stop()

    def _on_play_speed_changed(self, ms):
        if hasattr(self, "_play_timer"):
            self._play_timer.setInterval(ms)

    def _play_step(self):
        meta = self.file_fields.get(self.color_by)
        if not meta or not meta.get("time_varying"):
            self._play_timer.stop()
            self.panel.play_btn.setChecked(False)
            return
        n = meta["shape"][0]
        new_idx = (self.time_index + 1) % n
        self.panel.time_slider.setValue(new_idx)   # triggers _on_time_changed

    def _on_spin(self, on):
        if on:
            self._spin_timer.start()
        else:
            self._spin_timer.stop()
        self.spin_on = on

    def _on_screenshot(self):
        from .export_dialog import PngExportDialog
        default = f"icoscope_{datetime.now():%Y%m%d_%H%M%S}.png"
        dlg = PngExportDialog(self, default_filename=default,
                              default_transparent=self.transparent_export)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        path, scale, transparent = dlg.result()
        self.transparent_export = transparent  # remember for next time
        if path:
            # PyVista's `scale=N` screenshot internally resizes the live render
            # window to N× the current size, draws into it, and resizes back.
            # Without intervention the user sees a brief flicker. We:
            #   1. freeze Qt updates on the interactor widget (no paint events
            #      reach the user during the screenshot),
            #   2. snapshot the camera state and restore it afterwards in case
            #      the scaled render perturbs anything,
            #   3. re-enable updates and force one clean render.
            vc = self.plotter.renderer.GetActiveCamera()
            saved = (tuple(vc.GetPosition()),
                     tuple(vc.GetFocalPoint()),
                     tuple(vc.GetViewUp()),
                     vc.GetViewAngle(),
                     vc.GetParallelScale())
            iren_widget = self.plotter.interactor
            iren_widget.setUpdatesEnabled(False)
            try:
                self.plotter.screenshot(path, transparent_background=transparent,
                                        scale=scale)
            finally:
                vc.SetPosition(*saved[0])
                vc.SetFocalPoint(*saved[1])
                vc.SetViewUp(*saved[2])
                vc.SetViewAngle(saved[3])
                vc.SetParallelScale(saved[4])
                iren_widget.setUpdatesEnabled(True)
                self.plotter.render()
            self.statusBar().showMessage(
                f"saved → {path} ({scale}× resolution)", 5000)

    def _on_vector_export(self):
        default = f"icoscope_{datetime.now():%Y%m%d_%H%M%S}.svg"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save SVG", default, "SVG (*.svg)"
        )
        if not path:
            return
        try:
            import vtk
            ex = vtk.vtkGL2PSExporter()
            prefix = os.path.splitext(path)[0]
            ex.SetFilePrefix(prefix)
            ex.SetFileFormatToSVG()
            ex.CompressOff()
            ex.SetSortToBSP()
            ex.SetRenderWindow(self.plotter.render_window)
            ex.Write()
            self.statusBar().showMessage(f"saved → {prefix}.svg", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Vector export failed",
                                 f"{e}\n\nThis needs a VTK build with GL2PS support.")


def run(verts, cells, centers, initial_n=8, relax=True, file_path=None,
        zoom_factor=1.0, zoom_lon=0.0, zoom_lat=45.0):
    """Create the QApplication, show the main window, and start the Qt event loop."""
    app = QApplication.instance() or QApplication(sys.argv)
    # Set the icon on the QApplication BEFORE any window appears — that's the
    # only way macOS picks it up for the dock instead of showing the Python
    # interpreter's icon.
    icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    app.setApplicationName("IcoScope")
    app.setApplicationDisplayName("IcoScope")
    w = MainWindow(verts, cells, centers, initial_n=initial_n, relax=relax,
                   zoom_factor=zoom_factor, zoom_lon=zoom_lon, zoom_lat=zoom_lat)
    if file_path:
        # Load the file's fields into the panel as if user had clicked Open.
        from .loader import load_grid
        _v, _c, _ctr, fields = load_grid(file_path)
        w.file_path = file_path
        w.file_fields = fields
        items = ["None"] + list(fields.keys())
        w.panel.set_color_by_items(items)
        if fields:
            first = next(iter(fields.keys()))
            w.panel.set_color_by(first)
            w.color_by = first
        w._refresh_scalars()
        w._mesh = w._to_polydata()
        w.panel.disable_n(True)
        w._sync_file_info(file_path)
        w._build_scene()
        w._update_status()
    w.show()
    sys.exit(app.exec())
