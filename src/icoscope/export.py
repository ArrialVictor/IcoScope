"""PNG screenshot + SVG vector export handlers.

Plain functions — the slots on :class:`MainWindow` delegate here. The
screenshot path freezes the interactor widget while PyVista resizes the
render window for super-sampling, restores the camera state afterwards,
and forces a clean render.
"""
import os
from datetime import datetime

from qtpy.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from .export_dialog import PngExportDialog


def save_screenshot(window: QMainWindow, plotter, *, transparent: bool) -> bool:
    """Prompt for a PNG path and save a screenshot at the chosen scale.

    Parameters
    ----------
    window
        Parent for the export dialog and status-bar messages.
    plotter
        The PyVista ``QtInteractor`` whose render window to capture.
    transparent
        Initial state of the dialog's "transparent background" checkbox.

    Returns
    -------
    bool
        The final value of the transparent checkbox — the caller should
        store it back so the choice is remembered between exports.
    """
    default = f"icoscope_{datetime.now():%Y%m%d_%H%M%S}.png"
    dlg = PngExportDialog(window, default_filename=default,
                          default_transparent=transparent)
    if dlg.exec() != dlg.DialogCode.Accepted:
        return transparent
    path, scale, transparent = dlg.result()
    if path:
        # PyVista's `scale=N` screenshot internally resizes the live render
        # window to N× the current size, draws into it, and resizes back.
        # Without intervention the user sees a brief flicker. We:
        #   1. freeze Qt updates on the interactor widget (no paint events
        #      reach the user during the screenshot),
        #   2. snapshot the camera state and restore it afterwards in case
        #      the scaled render perturbs anything,
        #   3. re-enable updates and force one clean render.
        vc = plotter.renderer.GetActiveCamera()
        saved = (tuple(vc.GetPosition()),
                 tuple(vc.GetFocalPoint()),
                 tuple(vc.GetViewUp()),
                 vc.GetViewAngle(),
                 vc.GetParallelScale())
        iren_widget = plotter.interactor
        iren_widget.setUpdatesEnabled(False)
        try:
            plotter.screenshot(path, transparent_background=transparent,
                               scale=scale)
        finally:
            vc.SetPosition(*saved[0])
            vc.SetFocalPoint(*saved[1])
            vc.SetViewUp(*saved[2])
            vc.SetViewAngle(saved[3])
            vc.SetParallelScale(saved[4])
            iren_widget.setUpdatesEnabled(True)
            plotter.render()
        window.statusBar().showMessage(
            f"saved → {path} ({scale}× resolution)", 5000)
    return transparent


def save_vector(window: QMainWindow, plotter) -> None:
    """Prompt for an SVG path and export the current scene via vtkGL2PSExporter.

    Parameters
    ----------
    window
        Parent for the file dialog and error message box.
    plotter
        The PyVista ``QtInteractor`` whose render window to export.
    """
    default = f"icoscope_{datetime.now():%Y%m%d_%H%M%S}.svg"
    path, _ = QFileDialog.getSaveFileName(
        window, "Save SVG", default, "SVG (*.svg)"
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
        ex.SetRenderWindow(plotter.render_window)
        ex.Write()
        window.statusBar().showMessage(f"saved → {prefix}.svg", 5000)
    except Exception as e:
        QMessageBox.critical(window, "Vector export failed",
                             f"{e}\n\nThis needs a VTK build with GL2PS support.")
