"""LMDZ tanh-zoom sanity checks on the synthetic LonLat mesh."""
import numpy as np
import pytest

from icoscope.lonlat import lonlat_mesh


def _cell_area(verts: np.ndarray, cell: list[int]) -> float:
    """Approximate area of a spherical polygon by flat-triangle fan."""
    pts = np.asarray(verts)[cell]
    if len(cell) < 3:
        return 0.0
    a = pts[0]
    area = 0.0
    for i in range(1, len(cell) - 1):
        b = pts[i] - a
        c = pts[i + 1] - a
        area += 0.5 * np.linalg.norm(np.cross(b, c))
    return float(area)


def test_identity_matches_unzoomed():
    """grossismx == grossismy == 1.0 must reproduce the uniform mesh exactly."""
    v0, c0, ctr0 = lonlat_mesh(iim=24, jjm=20)
    v1, c1, ctr1 = lonlat_mesh(iim=24, jjm=20,
                                clon=12.0, clat=-30.0,
                                grossismx=1.0, grossismy=1.0,
                                dzoomx=0.1, dzoomy=0.1,
                                taux=2.5, tauy=2.5)
    assert np.array_equal(v0, v1)
    assert c0 == c1
    assert np.array_equal(ctr0, ctr1)


def test_focal_region_has_smaller_cells():
    """With a non-identity zoom, cells nearest the focal point are smaller."""
    iim, jjm = 60, 40
    clon, clat = 2.0, 48.0
    v, cells, centers = lonlat_mesh(
        iim=iim, jjm=jjm,
        clon=clon, clat=clat,
        grossismx=4.0, grossismy=4.0,
        dzoomx=0.08, dzoomy=0.08,
        taux=3.0, tauy=3.0,
    )
    sizes = np.array([_cell_area(v, c) for c in cells])
    lon_c = np.radians(clon)
    lat_c = np.radians(clat)
    focal = np.array([np.cos(lat_c) * np.cos(lon_c),
                      np.cos(lat_c) * np.sin(lon_c),
                      np.sin(lat_c)])
    dists = np.arccos(np.clip(np.asarray(centers) @ focal, -1.0, 1.0))
    near = sizes[dists < 0.3]
    far = sizes[dists > 2.5]
    # Skip if either region is empty (shouldn't happen with these dims).
    assert near.size > 0 and far.size > 0
    assert near.mean() < far.mean(), (
        f"focal area ({near.mean():.2e}) should be smaller than antipodal "
        f"area ({far.mean():.2e})"
    )


def test_validity_error_is_raised():
    """grossism · dzoom too large → LMDZ's validity check should reject."""
    # grossism * dzoom_frac ≈ 1.0 → very wide zoom that violates 2β - G > 0.
    with pytest.raises(ValueError, match="Decrease dzoomx or grossismx"):
        lonlat_mesh(iim=20, jjm=20,
                    grossismx=20.0, grossismy=1.0,
                    dzoomx=0.49, dzoomy=0.0,
                    taux=3.0, tauy=3.0)


def test_total_area_conserved():
    """Sum of cell areas is the same with or without zoom (sphere is still a sphere)."""
    iim, jjm = 36, 24
    v0, c0, _ = lonlat_mesh(iim=iim, jjm=jjm)
    v1, c1, _ = lonlat_mesh(iim=iim, jjm=jjm,
                             clon=10.0, clat=20.0,
                             grossismx=3.0, grossismy=2.0,
                             dzoomx=0.1, dzoomy=0.1,
                             taux=3.0, tauy=3.0)
    a0 = sum(_cell_area(v0, c) for c in c0)
    a1 = sum(_cell_area(v1, c) for c in c1)
    # Flat-triangle areas don't equal 4π, but they should agree across zoom.
    assert a0 == pytest.approx(a1, rel=0.02)


def test_topology_preserved():
    """Zoom keeps cell count, polar-triangle count, and pole-vertex degree."""
    iim, jjm = 30, 18
    v0, c0, _ = lonlat_mesh(iim=iim, jjm=jjm)
    v1, c1, _ = lonlat_mesh(iim=iim, jjm=jjm,
                             clon=15.0, clat=30.0,
                             grossismx=2.5, grossismy=2.5,
                             dzoomx=0.1, dzoomy=0.1,
                             taux=3.0, tauy=3.0)
    assert len(c0) == len(c1)
    assert sum(len(c) == 3 for c in c0) == sum(len(c) == 3 for c in c1) == 2 * iim
    assert sum(len(c) == 4 for c in c0) == sum(len(c) == 4 for c in c1)
    assert v0.shape == v1.shape


def test_vertices_stay_on_unit_sphere():
    """All vertices and centers of a zoomed mesh sit on the unit sphere."""
    v, cells, centers = lonlat_mesh(
        iim=40, jjm=30,
        clon=2.0, clat=48.0,
        grossismx=4.0, grossismy=4.0,
        dzoomx=0.1, dzoomy=0.1,
        taux=3.0, tauy=3.0,
    )
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-10)
    assert np.allclose(np.linalg.norm(centers, axis=1), 1.0, atol=1e-10)
