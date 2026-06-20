"""Build a Goldberg polyhedron: dual of a Class-I subdivided icosahedron.

Produces 12 pentagons (at original icosahedron vertices) and 10*n^2 - 20 hexagons
for subdivision frequency n. Not identical to ICOLMDZ's optimized grid, but the
same topology — a useful stand-in for visualization.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

PHI = (1 + np.sqrt(5)) / 2


class GoldbergGrid(NamedTuple):
    """The output of :func:`goldberg`.

    Attributes
    ----------
        verts:   (V, 3) array of dual-cell corner positions on the unit sphere.
        cells:   list of length C, where ``cells[i]`` is a list of vertex
                 indices forming cell *i* (length 5 for pentagons, 6 for
                 hexagons). Cells are in CCW order viewed from outside.
        centers: (C, 3) array of cell-center positions on the unit sphere.
        iters:   relaxation iterations actually used (0 when ``relax=False``).
    """

    verts: np.ndarray
    cells: list[list[int]]
    centers: np.ndarray
    iters: int


def icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(verts, faces)`` for a unit-sphere regular icosahedron."""
    v = np.array([
        [-1,  PHI, 0], [ 1,  PHI, 0], [-1, -PHI, 0], [ 1, -PHI, 0],
        [0, -1,  PHI], [0,  1,  PHI], [0, -1, -PHI], [0,  1, -PHI],
        [ PHI, 0, -1], [ PHI, 0,  1], [-PHI, 0, -1], [-PHI, 0,  1],
    ], dtype=float)
    v /= np.linalg.norm(v[0])
    f = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ])
    return v, f


def subdivide(verts: np.ndarray, faces: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Class-I subdivision: each triangle → n^2 smaller triangles, projected on sphere."""
    out_verts = [tuple(v) for v in verts]
    index = {tuple(np.round(v, 9)): i for i, v in enumerate(verts)}
    out_faces = []

    def get(p):
        p = p / np.linalg.norm(p)
        key = tuple(np.round(p, 9))
        if key not in index:
            index[key] = len(out_verts)
            out_verts.append(tuple(p))
        return index[key]

    for tri in faces:
        A, B, C = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        g = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                g[(i, j)] = get((i * A + j * B + k * C) / n)
        for i in range(n):
            for j in range(n - i):
                out_faces.append([g[(i, j)], g[(i, j + 1)], g[(i + 1, j)]])
                if j < n - i - 1:
                    out_faces.append([g[(i + 1, j)], g[(i, j + 1)], g[(i + 1, j + 1)]])

    return np.array(out_verts), np.array(out_faces)


def dual(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Goldberg dual: one cell per vertex, cell vertices are incident-triangle centroids."""
    incident = [[] for _ in verts]
    for fi, tri in enumerate(faces):
        for vi in tri:
            incident[vi].append(fi)

    centroids = verts[faces].mean(axis=1)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    # vectorised tangent frames (u, w) per vertex
    tmps = np.where(np.abs(verts[:, 0:1]) < 0.9,
                    np.array([[1.0, 0.0, 0.0]]),
                    np.array([[0.0, 1.0, 0.0]]))
    us = np.cross(verts, tmps)
    us /= np.linalg.norm(us, axis=1, keepdims=True)
    ws = np.cross(verts, us)

    cells = []
    for vi, tris in enumerate(incident):
        ds = centroids[tris] - verts[vi]
        angles = np.arctan2(ds @ ws[vi], ds @ us[vi])
        cells.append([tris[k] for k in np.argsort(angles)])

    return centroids, cells


def relax_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    max_iterations: int = 200,
    step: float = 1.0,
    tol: float = 1e-4,
) -> tuple[np.ndarray, int]:
    """Spring-relax the triangle mesh on the unit sphere.

    The 12 icosahedron vertices (indices 0–11) stay pinned by symmetry; all
    other vertices slide tangentially until edge lengths equalise.

    Iterates until the edge-length coefficient-of-variation (std/mean) changes
    by less than `tol` between steps, or until `max_iterations` is reached
    (DYNAMICO-style convergence + safety cap).

    Returns (verts, iters_used).
    """
    verts = np.asarray(verts, dtype=float).copy()
    e_set = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        e_set.add((min(a, b), max(a, b)))
        e_set.add((min(b, c), max(b, c)))
        e_set.add((min(a, c), max(a, c)))
    edges = np.array(sorted(e_set), dtype=np.int64)

    valence = np.bincount(edges.ravel(), minlength=len(verts)).astype(float)
    valence = np.maximum(valence, 1)[:, None]

    prev_cv = None
    used = 0
    for it in range(max_iterations):
        ev = verts[edges[:, 1]] - verts[edges[:, 0]]
        L = np.linalg.norm(ev, axis=1)
        target = L.mean()
        cv = L.std() / target  # coefficient of variation — uniformity metric

        if prev_cv is not None:
            rel = abs(prev_cv - cv) / max(cv, 1e-12)
            if rel < tol:
                used = it
                break
        prev_cv = cv

        f_edge = (ev / L[:, None]) * (L - target)[:, None]
        # accumulate per-vertex force via bincount on each axis
        # (faster than np.add.at by ~5×)
        e0, e1 = edges[:, 0], edges[:, 1]
        N = len(verts)
        forces = np.empty_like(verts)
        for k in range(3):
            forces[:, k] = (np.bincount(e0, weights=f_edge[:, k], minlength=N)
                            - np.bincount(e1, weights=f_edge[:, k], minlength=N))
        forces /= valence

        normals = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        forces -= (forces * normals).sum(axis=1, keepdims=True) * normals
        forces[:12] = 0

        verts += step * forces
        verts /= np.linalg.norm(verts, axis=1, keepdims=True)
        used = it + 1

    return verts, used


def schmidt_stretch(
    points: np.ndarray,
    factor: float,
    lon_deg: float,
    lat_deg: float,
) -> np.ndarray:
    """Apply DYNAMICO's Schmidt conformal stretch to points on the unit sphere.

    Mirrors the implementation in DYNAMICO's ``spherical_geom.f90``
    (subroutine ``schmidt_transform``), which cites Guo & Drake, *JCP* 2005,
    eq. (12). The map rotates the focal point ``(lon_deg, lat_deg)`` to the
    north pole, applies a Möbius stretch in ``sin(lat)``, and rotates back.

    Parameters
    ----------
    points : ndarray
        ``(N, 3)`` array of positions on the unit sphere.
    factor : float
        Schmidt stretching factor. ``1.0`` is the identity; ``> 1`` concentrates
        cells at the focal point and coarsens the antipode; ``< 1`` does the
        opposite. Linear refinement at the focal point ≈ ``factor``; areal
        refinement ≈ ``factor ** 2``.
    lon_deg, lat_deg : float
        Focal-point coordinates in degrees.

    Returns
    -------
    ndarray
        ``(N, 3)`` stretched positions, re-normalized to the unit sphere.
    """
    if abs(factor - 1.0) < 1e-12:
        return points

    # Rotate the focal point to the north pole. Apply lon by Rz(-lon), then
    # lat by Ry(lat - π/2). The inverse rotates back.
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    cz, sz = np.cos(-lon), np.sin(-lon)
    cy, sy = np.cos(lat - np.pi / 2), np.sin(lat - np.pi / 2)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot = ry @ rz                  # focal → north pole
    rot_inv = rot.T                # north pole → focal

    p_rot = points @ rot.T

    # Möbius stretch on sin(lat'). cc = factor ** 2 matches DYNAMICO's
    # `schmidt_factor = schmidt_factor**2` convention internally.
    cc = factor * factor
    mu = p_rot[:, 2]                       # sin(lat') because we're on unit sphere
    mu_p = ((cc - 1.0) + mu * (cc + 1.0)) / ((cc + 1.0) + mu * (cc - 1.0))
    # Lift back to (x, y, z): preserve azimuth, replace sin(lat) with mu_p.
    horiz = p_rot[:, :2]
    horiz_norm = np.linalg.norm(horiz, axis=1, keepdims=True)
    safe = horiz_norm > 1e-12
    new_horiz_scale = np.where(safe, np.sqrt(1.0 - mu_p * mu_p)[:, None] / horiz_norm, 0.0)
    out_rot = np.empty_like(p_rot)
    out_rot[:, :2] = horiz * new_horiz_scale
    out_rot[:, 2] = mu_p

    out = out_rot @ rot_inv.T
    # Defensive re-normalization against floating-point drift.
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def goldberg(
    n: int = 4,
    relax: bool = False,
    max_iterations: int = 200,
    tol: float = 1e-4,
    zoom_factor: float = 1.0,
    zoom_lon: float = 0.0,
    zoom_lat: float = 45.0,
) -> GoldbergGrid:
    """Build a Goldberg polyhedron of frequency *n*.

    With ``relax=True``, run spring-relaxation (with early-stop on convergence
    of the edge-length CV) on the subdivided triangle mesh before taking the
    dual. Yields cells closer to the DYNAMICO-style optimized grid.

    With ``zoom_factor != 1.0``, apply DYNAMICO's Schmidt conformal stretch
    (see :func:`schmidt_stretch`) to the subdivided mesh before taking the
    dual, concentrating cells near ``(zoom_lon, zoom_lat)``.

    Parameters
    ----------
    n : int
        Subdivision frequency (DYNAMICO ``nbp``). Total cells = ``10·n² + 2``.
    relax : bool
        Run spring-relaxation on the triangle mesh before the dual.
    max_iterations, tol : int, float
        Relaxation cap and convergence tolerance.
    zoom_factor : float
        Schmidt stretching factor; ``1.0`` is the identity (uniform mesh).
    zoom_lon, zoom_lat : float
        Schmidt focal point in degrees.

    Returns
    -------
    GoldbergGrid
        Named tuple ``(verts, cells, centers, iters)``.
    """
    iv, ifc = icosahedron()
    sv, sf = subdivide(iv, ifc, n)
    iters = 0
    if relax:
        sv, iters = relax_mesh(sv, sf, max_iterations=max_iterations, tol=tol)
    if abs(zoom_factor - 1.0) >= 1e-12:
        sv = schmidt_stretch(sv, zoom_factor, zoom_lon, zoom_lat)
    verts, cells = dual(sv, sf)
    return GoldbergGrid(verts=verts, cells=cells, centers=sv, iters=iters)
