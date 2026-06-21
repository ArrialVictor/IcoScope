"""Sanity checks for the Schmidt conformal stretch (DYNAMICO zoom)."""
import numpy as np
import pytest

from icoscope.grid import goldberg, schmidt_stretch


def _focal_xyz(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    return np.array([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)])


def test_identity_when_factor_is_one():
    """factor=1.0 must be a no-op."""
    pts = goldberg(6).verts
    out = schmidt_stretch(pts, 1.0, 12.3, -45.6)
    assert np.array_equal(pts, out)


def test_focal_point_is_fixed():
    """The focal point and its antipode are fixed points of the map."""
    lon, lat = 30.0, 60.0
    focal = _focal_xyz(lon, lat)
    antipode = -focal
    pts = np.vstack([focal, antipode])
    out = schmidt_stretch(pts, 3.0, lon, lat)
    assert np.allclose(out[0], focal, atol=1e-12)
    assert np.allclose(out[1], antipode, atol=1e-12)


def test_points_stay_on_unit_sphere():
    """All stretched points remain on the unit sphere."""
    pts = goldberg(8).verts
    out = schmidt_stretch(pts, 4.0, 2.0, 48.0)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


@pytest.mark.parametrize("factor", [1.5, 3.0, 5.0])
def test_focal_region_has_smaller_cells(factor: float):
    """With factor>1, cells nearest the focal point are smaller than the mean."""
    lon, lat = 2.0, 48.0          # Paris
    g = goldberg(12, zoom_factor=factor, zoom_lon=lon, zoom_lat=lat)
    focal = _focal_xyz(lon, lat)

    # Per-cell mean radius (proxy for cell size).
    sizes = np.array([
        np.linalg.norm(g.verts[cell] - g.verts[cell].mean(0), axis=1).mean()
        for cell in g.cells
    ])
    # Distance from focal point on the sphere.
    dists = np.arccos(np.clip(g.centers @ focal, -1.0, 1.0))
    near = sizes[dists < 0.3].mean()
    overall = sizes.mean()
    assert near < overall, f"focal cells ({near}) should be smaller than mean ({overall})"


def test_factor_below_one_inverts_concentration():
    """With factor<1, cells at the focal point are *larger* than overall mean."""
    lon, lat = 0.0, 0.0
    g = goldberg(12, zoom_factor=0.5, zoom_lon=lon, zoom_lat=lat)
    focal = _focal_xyz(lon, lat)
    sizes = np.array([
        np.linalg.norm(g.verts[cell] - g.verts[cell].mean(0), axis=1).mean()
        for cell in g.cells
    ])
    dists = np.arccos(np.clip(g.centers @ focal, -1.0, 1.0))
    near = sizes[dists < 0.3].mean()
    overall = sizes.mean()
    assert near > overall


def test_goldberg_zoom_preserves_topology():
    """Zoom must not change cell counts or pentagon count."""
    plain = goldberg(8)
    zoomed = goldberg(8, zoom_factor=3.0, zoom_lon=2.0, zoom_lat=48.0)
    assert len(plain.cells) == len(zoomed.cells)
    assert sum(len(c) == 5 for c in zoomed.cells) == 12


def test_factor_below_one_pushes_near_focal_points_away():
    """factor<1 anti-zoom: a point near the focal point moves outward (away
    in angular distance) — the focal point is still fixed but its neighborhood
    expands.

    Tests :func:`schmidt_stretch` directly, separately from the goldberg path.
    """
    lon, lat = 0.0, 0.0
    focal = _focal_xyz(lon, lat)
    # A point 5° away from the focal point along the lon axis.
    near = _focal_xyz(5.0, 0.0)
    out = schmidt_stretch(np.vstack([focal, near]), factor=0.5,
                          lon_deg=lon, lat_deg=lat)
    # Focal stays put.
    assert np.allclose(out[0], focal, atol=1e-12)
    # The near point moves *further* from the focal point under anti-zoom.
    before_dist = np.arccos(np.clip(focal @ near, -1.0, 1.0))
    after_dist = np.arccos(np.clip(focal @ out[1], -1.0, 1.0))
    assert after_dist > before_dist, (
        f"factor=0.5 should push near-focal points outward "
        f"(before {before_dist}, after {after_dist})"
    )
