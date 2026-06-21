"""Topology, cell-count, and relaxation invariants of the Goldberg grid."""
import numpy as np
import pytest

from icoscope.grid import GoldbergGrid, goldberg, icosahedron, relax_mesh, subdivide


@pytest.mark.parametrize("n", [1, 2, 4, 8, 12, 20])
def test_cell_count_and_pentagons(n: int):
    """Goldberg(n) → 10n²+2 cells, always 12 pentagons, rest hexagons."""
    g = goldberg(n)
    assert isinstance(g, GoldbergGrid)
    assert len(g.cells) == 10 * n * n + 2
    pent = sum(1 for c in g.cells if len(c) == 5)
    hexa = sum(1 for c in g.cells if len(c) == 6)
    assert pent == 12
    assert hexa == len(g.cells) - 12


def test_vertices_on_unit_sphere():
    """Every vertex and every center sits on the unit sphere."""
    g = goldberg(6)
    norms_v = np.linalg.norm(g.verts, axis=1)
    norms_c = np.linalg.norm(g.centers, axis=1)
    assert np.allclose(norms_v, 1.0, atol=1e-10)
    assert np.allclose(norms_c, 1.0, atol=1e-10)


def test_relaxation_reduces_size_spread():
    """Relaxation should lower the cell-size coefficient of variation."""
    raw = goldberg(12, relax=False)
    relaxed = goldberg(12, relax=True, max_iterations=200)

    def size_cv(g: GoldbergGrid) -> float:
        rs = []
        for cell in g.cells:
            pts = g.verts[cell]
            mu = pts.mean(0)
            rs.append(np.linalg.norm(pts - mu, axis=1).mean())
        rs = np.array(rs)
        return rs.std() / rs.mean()

    assert size_cv(relaxed) < size_cv(raw)


def test_relaxation_pins_icosahedron_vertices():
    """The 12 original icosahedron vertices must not move during relaxation."""
    iv, ifc = icosahedron()
    sv, sf = subdivide(iv, ifc, 8)
    before = sv[:12].copy()
    after, _ = relax_mesh(sv, sf, max_iterations=50)
    assert np.allclose(before, after[:12], atol=1e-12)


def test_relaxation_converges_early():
    """Early-stop should fire before the cap at small n."""
    g = goldberg(6, relax=True, max_iterations=500, tol=1e-4)
    assert 0 < g.iters < 500


def test_relax_mesh_hits_iteration_cap():
    """With tol=0 (impossible to converge) relaxation must hit max_iterations."""
    iv, ifc = icosahedron()
    sv, sf = subdivide(iv, ifc, 4)
    _, iters = relax_mesh(sv, sf, max_iterations=3, tol=0.0)
    assert iters == 3


def test_no_relax_returns_iters_zero():
    g = goldberg(8, relax=False)
    assert g.iters == 0
