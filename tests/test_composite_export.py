"""Tile-helper unit tests — pure numpy, no Qt or VTK needed."""
import numpy as np
import pytest

from icoscope.export import _normalise_tile, _tile_panes


def _solid(w, h, rgb, alpha=None):
    if alpha is None:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 0] = rgb[0]
        img[..., 1] = rgb[1]
        img[..., 2] = rgb[2]
        return img
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[..., 0] = rgb[0]
    img[..., 1] = rgb[1]
    img[..., 2] = rgb[2]
    img[..., 3] = alpha
    return img


def test_tile_1x2_opaque_preserves_widths():
    left = _solid(100, 80, (255, 0, 0))
    right = _solid(60, 80, (0, 255, 0))
    out = _tile_panes(
        [left, right],
        [(0, 0, 100, 80), (100, 0, 60, 80)],
        scale=1, transparent=False,
    )
    assert out.shape == (80, 160, 3)
    # left half is red, right slice is green
    assert (out[:, :100, 0] == 255).all()
    assert (out[:, 100:, 1] == 255).all()


def test_tile_2x2_with_scale():
    tiles = [_solid(50, 40, (i * 60, 0, 0)) for i in range(4)]
    rects = [(0, 0, 50, 40), (50, 0, 50, 40),
             (0, 40, 50, 40), (50, 40, 50, 40)]
    out = _tile_panes(tiles, rects, scale=3, transparent=False)
    assert out.shape == (240, 300, 3)
    # Bottom-right quadrant should hold tile #3's red component
    assert (out[120:, 150:, 0] == 180).all()


def test_tile_transparent_keeps_alpha_outside_panes():
    # Panes don't cover the full canvas — leave a gap to verify zero-alpha
    tile = _solid(40, 40, (0, 0, 255), alpha=255)
    out = _tile_panes(
        [tile, tile],
        [(0, 0, 40, 40), (60, 0, 40, 40)],   # 20-px gap between panes
        scale=1, transparent=True,
    )
    assert out.shape == (40, 100, 4)
    # Gap region is fully transparent
    assert (out[:, 40:60, 3] == 0).all()
    # Pane regions are fully opaque blue
    assert (out[:, :40, 2] == 255).all()
    assert (out[:, :40, 3] == 255).all()
    assert (out[:, 60:, 2] == 255).all()


def test_tile_opaque_background_fill():
    tile = _solid(20, 20, (0, 0, 0))
    out = _tile_panes(
        [tile],
        [(10, 10, 20, 20)],
        scale=1, transparent=False, bg_color=(123, 200, 50),
    )
    assert out.shape == (30, 30, 3)
    # Top-left margin uses bg_color
    assert (out[0, 0] == [123, 200, 50]).all()
    # Pane region is the tile colour
    assert (out[20, 20] == [0, 0, 0]).all()


def test_normalise_tile_resizes_off_by_one():
    # VTK can return a render-window image one pixel off the requested size;
    # _normalise_tile resizes invisibly.
    img = _solid(99, 79, (10, 20, 30))
    out = _normalise_tile(img, 100, 80, 3)
    assert out.shape == (80, 100, 3)


def test_normalise_tile_rgb_to_rgba_pads_alpha():
    img = _solid(20, 20, (10, 20, 30))
    out = _normalise_tile(img, 20, 20, 4)
    assert out.shape == (20, 20, 4)
    assert (out[..., 3] == 255).all()


def test_normalise_tile_rgba_to_rgb_drops_alpha():
    img = _solid(20, 20, (10, 20, 30), alpha=128)
    out = _normalise_tile(img, 20, 20, 3)
    assert out.shape == (20, 20, 3)


def test_tile_panes_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        _tile_panes([_solid(10, 10, (0, 0, 0))],
                    [(0, 0, 10, 10), (10, 0, 10, 10)],
                    scale=1, transparent=False)


def test_tile_panes_rejects_empty():
    with pytest.raises(ValueError):
        _tile_panes([], [], scale=1, transparent=False)
