"""Topology and invariants of the synthetic LMDZ-style lat-lon mesh."""
from collections import Counter

import numpy as np
import pytest

from icoscope.lonlat import latlon_mesh


@pytest.mark.parametrize(("iim", "jjm"), [(4, 3), (8, 5), (20, 10), (96, 95)])
def test_cell_count_is_iim_times_jjm(iim, jjm):
    """The mesh has exactly ``iim * jjm`` cells."""
    _, cells, _ = latlon_mesh(iim=iim, jjm=jjm)
    assert len(cells) == iim * jjm


def test_vertices_on_unit_sphere():
    """Every vertex and every center sits on the unit sphere."""
    verts, _, centers = latlon_mesh(iim=20, jjm=10)
    assert np.allclose(np.linalg.norm(verts, axis=1), 1.0, atol=1e-10)
    assert np.allclose(np.linalg.norm(centers, axis=1), 1.0, atol=1e-10)


def test_polar_rows_are_triangles_interior_are_quads():
    """First/last latitude bands are triangle fans; interior bands are quads."""
    iim, jjm = 12, 6
    _, cells, _ = latlon_mesh(iim=iim, jjm=jjm)
    n_tri = sum(1 for c in cells if len(c) == 3)
    n_quad = sum(1 for c in cells if len(c) == 4)
    assert n_tri == 2 * iim                      # iim triangles per pole
    assert n_quad == iim * (jjm - 2)
    assert n_tri + n_quad == len(cells)


def test_each_pole_is_shared_by_iim_cells():
    """Both poles deduplicate to one vertex, each shared by exactly ``iim`` cells."""
    iim, jjm = 16, 7
    _, cells, _ = latlon_mesh(iim=iim, jjm=jjm)
    counter = Counter(v for cell in cells for v in cell)
    top_two = [count for _, count in counter.most_common(2)]
    assert top_two == [iim, iim]


def test_periodic_seam_has_no_duplicate_cell():
    """``i = iim`` is the wrap of ``i = 0``; the cell list must not duplicate it."""
    iim, jjm = 6, 4
    _, cells, _ = latlon_mesh(iim=iim, jjm=jjm)
    # Cells are listed j outer, i inner: so cells[0..iim-1] are the north
    # polar band and must all be distinct vertex-index tuples.
    north_band = [tuple(sorted(c)) for c in cells[:iim]]
    assert len(set(north_band)) == iim


def test_polar_cell_centers_coincide_with_pole_vertex():
    """Polar-row centers sit AT the pole (matching LMDZ's scalar placement)."""
    iim, jjm = 10, 5
    verts, cells, centers = latlon_mesh(iim=iim, jjm=jjm)
    # The north pole is shared by the first iim cells; its xyz is (0, 0, 1).
    north_pole = np.array([0.0, 0.0, 1.0])
    south_pole = np.array([0.0, 0.0, -1.0])
    for k in range(iim):
        assert np.allclose(centers[k], north_pole, atol=1e-12)
    for k in range(len(cells) - iim, len(cells)):
        assert np.allclose(centers[k], south_pole, atol=1e-12)


def test_default_lmdz_low_res():
    """Defaults are LMDZ's commonly cited low-res ``iim=96, jjm=95``."""
    verts, cells, centers = latlon_mesh()
    assert len(cells) == 96 * 95
    assert len(centers) == 96 * 95
    assert verts.shape[1] == 3
