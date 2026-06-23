"""PNG screenshot + SVG vector export handlers.

The PNG path supports a single composite image (all visible panes laid out
as they appear on screen) plus optional per-pane PNGs and per-pane SVGs.
The composite tile is a pure-numpy step (see :func:`_tile_panes`) so it
can be unit-tested without Qt or VTK.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from PIL import Image
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QMainWindow, QMessageBox

from .export_dialog import ExportDialog


@dataclass
class ExportDefaults:
    """User choices remembered between Export… invocations."""

    transparent: bool = False
    scale: int = 1
    png_composite: bool = True
    png_per_pane: bool = True
    svg_per_pane: bool = False
    flags_initialised: bool = field(default=False, repr=False)


def _tile_panes(
    images: list[np.ndarray],
    rects: list[tuple[int, int, int, int]],
    *,
    scale: int,
    transparent: bool,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Composite per-pane screenshots into one canvas.

    Parameters
    ----------
    images
        One screenshot per pane, ``HxWx3`` or ``HxWx4`` ``uint8``.
    rects
        ``(x, y, w, h)`` in container coordinates (the same coordinate
        system the pane widgets use, **before** multiplying by ``scale``).
    scale
        Resolution multiplier — the output canvas has dimensions
        ``container_size * scale``.
    transparent
        If true, output is RGBA with a transparent background and per-pane
        alpha preserved. If false, output is RGB and the background is
        ``bg_color``.
    bg_color
        Opaque-mode background fill, 0–255 RGB. Ignored when
        ``transparent`` is true.

    Returns
    -------
    numpy.ndarray
        ``HxWx{3,4}`` ``uint8`` canvas with each pane pasted at its
        scaled rectangle. Panes whose screenshot size doesn't exactly
        match the scaled rect are resized with PIL (catches off-by-one
        from VTK's render-window rounding).
    """
    if len(images) != len(rects):
        raise ValueError(
            f"images ({len(images)}) and rects ({len(rects)}) must match"
        )
    if not images:
        raise ValueError("at least one pane image is required")

    total_w = max(x + w for x, y, w, h in rects)
    total_h = max(y + h for x, y, w, h in rects)
    out_w = total_w * scale
    out_h = total_h * scale
    channels = 4 if transparent else 3

    canvas = np.zeros((out_h, out_w, channels), dtype=np.uint8)
    if not transparent:
        canvas[..., 0] = bg_color[0]
        canvas[..., 1] = bg_color[1]
        canvas[..., 2] = bg_color[2]

    for img, (x, y, w, h) in zip(images, rects, strict=True):
        target_w = w * scale
        target_h = h * scale
        tile = _normalise_tile(img, target_w, target_h, channels)
        canvas[y * scale: y * scale + target_h,
               x * scale: x * scale + target_w] = tile
    return canvas


def _normalise_tile(img: np.ndarray, w: int, h: int, channels: int) -> np.ndarray:
    """Resize ``img`` to ``(h, w)`` and coerce to ``channels``.

    VTK occasionally returns a render-window image one pixel off the
    requested dimensions; PIL's high-quality resampler handles the fix
    invisibly. RGB→RGBA pads alpha=255; RGBA→RGB drops the alpha.
    """
    src_h, src_w = img.shape[:2]
    src_channels = img.shape[2] if img.ndim == 3 else 1

    if (src_h, src_w) != (h, w):
        mode = "RGBA" if src_channels == 4 else "RGB"
        pil = Image.fromarray(img, mode=mode)
        pil = pil.resize((w, h), Image.LANCZOS)
        img = np.asarray(pil)
        src_channels = img.shape[2]

    if src_channels == channels:
        return img
    if src_channels == 3 and channels == 4:
        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., :3] = img
        out[..., 3] = 255
        return out
    if src_channels == 4 and channels == 3:
        return img[..., :3].copy()
    raise ValueError(f"cannot convert {src_channels}ch image to {channels}ch")


def _save_array_png(arr: np.ndarray, path: str) -> None:
    """Write an ``HxWx{3,4}`` uint8 array to ``path`` as PNG."""
    mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    Image.fromarray(arr, mode=mode).save(path, format="PNG")


def _pane_screenshot(plotter, *, scale: int, transparent: bool) -> np.ndarray:
    """Capture one pane's render window as a numpy array, restoring camera."""
    vc = plotter.renderer.GetActiveCamera()
    saved = (tuple(vc.GetPosition()),
             tuple(vc.GetFocalPoint()),
             tuple(vc.GetViewUp()),
             vc.GetViewAngle(),
             vc.GetParallelScale())
    iren_widget = plotter.interactor
    iren_widget.setUpdatesEnabled(False)
    try:
        img = plotter.screenshot(filename=None,
                                 transparent_background=transparent,
                                 scale=scale,
                                 return_img=True)
    finally:
        vc.SetPosition(*saved[0])
        vc.SetFocalPoint(*saved[1])
        vc.SetViewUp(*saved[2])
        vc.SetViewAngle(saved[3])
        vc.SetParallelScale(saved[4])
        iren_widget.setUpdatesEnabled(True)
        plotter.render()
    return img


def _save_pane_svg(plotter, prefix: str) -> str:
    """Export one pane to ``<prefix>.svg`` via vtkGL2PSExporter; return final path."""
    import vtk
    ex = vtk.vtkGL2PSExporter()
    ex.SetFilePrefix(prefix)
    ex.SetFileFormatToSVG()
    ex.CompressOff()
    ex.SetSortToBSP()
    ex.SetRenderWindow(plotter.render_window)
    ex.Write()
    return f"{prefix}.svg"


def save_export(window: QMainWindow, pane_container, active_idx: int,
                *, defaults: ExportDefaults) -> ExportDefaults:
    """Run the unified Export dialog and write whatever the user selected.

    Parameters
    ----------
    window
        Parent for the dialog and status-bar messages.
    pane_container
        The :class:`~icoscope.panes.PaneContainer` whose visible panes are
        the export source.
    active_idx
        Currently-selected pane index. In single-pane mode it's the only
        pane; in multi-pane mode the SVG export only meaningfully labels
        per-pane files so this is mainly a hint.
    defaults
        Remembered user choices from the previous invocation. Modified-
        in-place and returned for the caller to persist.

    Returns
    -------
    ExportDefaults
        Updated defaults reflecting this session's choices.
    """
    visible = pane_container.visible_panes()
    n_panes = len(visible)
    default_base = f"icoscope_{datetime.now():%Y%m%d_%H%M%S}"

    # First time we see a multi-pane layout, default the per-pane PNG box
    # ON so the user gets the "composite + individuals" bundle by default.
    if not defaults.flags_initialised:
        defaults.png_per_pane = n_panes > 1
        defaults.flags_initialised = True

    dlg = ExportDialog(window, default_base=default_base, n_panes=n_panes,
                       defaults=defaults)
    if dlg.exec() != dlg.DialogCode.Accepted:
        return defaults
    choice = dlg.result()
    defaults.transparent = choice.transparent
    defaults.scale = choice.scale
    defaults.png_composite = choice.png_composite
    defaults.png_per_pane = choice.png_per_pane
    defaults.svg_per_pane = choice.svg_per_pane

    base = choice.base_path
    if not base:
        return defaults

    written: list[str] = []
    want_png = choice.png_composite or choice.png_per_pane
    pane_imgs: list[np.ndarray] = []
    rects: list[tuple[int, int, int, int]] = []
    if want_png:
        # 2×2 panes are nested in row-splitters, so pane.x()/y() are relative
        # to the inner row, not the container. mapTo() walks the parent chain
        # to give container-local pixels — without this, panes 0/1 land at
        # the same coords as panes 2/3 and get overwritten.
        for pane in visible:
            img = _pane_screenshot(pane.plotter, scale=choice.scale,
                                   transparent=choice.transparent)
            pane_imgs.append(img)
            origin = pane.mapTo(pane_container, QPoint(0, 0))
            rects.append((origin.x(), origin.y(), pane.width(), pane.height()))

    try:
        if choice.png_composite:
            if n_panes == 1:
                composite = pane_imgs[0]
            else:
                composite = _tile_panes(
                    pane_imgs, rects,
                    scale=choice.scale, transparent=choice.transparent,
                )
            path = f"{base}.png"
            _save_array_png(composite, path)
            written.append(path)

        if choice.png_per_pane:
            suffix_fn = (lambda i: ".png") if n_panes == 1 \
                else (lambda i: f"_pane{i + 1}.png")
            for i, img in enumerate(pane_imgs):
                # In single-pane mode, the composite path *is* the per-pane
                # path — skip the second write to avoid overwriting.
                if n_panes == 1 and choice.png_composite:
                    continue
                path = f"{base}{suffix_fn(i)}"
                _save_array_png(img, path)
                written.append(path)

        if choice.svg_per_pane:
            suffix_fn = (lambda i: "") if n_panes == 1 \
                else (lambda i: f"_pane{i + 1}")
            for i, pane in enumerate(visible):
                prefix = f"{base}{suffix_fn(i)}"
                path = _save_pane_svg(pane.plotter, prefix)
                written.append(path)
    except Exception as e:
        QMessageBox.critical(window, "Export failed", str(e))
        return defaults

    if written:
        if len(written) == 1:
            msg = f"saved → {written[0]}"
        else:
            msg = f"saved {len(written)} files → {os.path.dirname(written[0]) or '.'}"
        window.statusBar().showMessage(msg, 5000)
    return defaults
