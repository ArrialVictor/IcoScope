"""Build LMDZ's classical regular lat-lon ("dyn3d") mesh on the unit sphere.

The dyn3d core stores scalars on the ``(rlonv, rlatu)`` corner of the Arakawa
C-grid. With ``iim`` distinct longitudes and ``jjm`` latitude bands, the mesh
has exactly ``iim * jjm`` cells:

- Interior bands (``1 < j < jjp1``) are quadrilaterals bounded by
  ``rlonu(i-1..i)`` × ``rlatv(j-1..j)``.
- The first and last bands collapse into triangle fans of ``iim`` triangles
  meeting at a single pole vertex (``rlatu(1) = +π/2``, ``rlatu(jjp1) = -π/2``).

The output tuple ``(verts, cells, centers)`` matches :func:`icoscope.grid.goldberg`
so the rest of the rendering pipeline works unchanged.
"""
from __future__ import annotations

import numpy as np


def _lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Spherical → Cartesian on the unit sphere. ``lon``, ``lat`` in radians."""
    return np.stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ], axis=-1)


def _dyn3d_coord_arrays(iim: int, jjm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rlonu, rlonv, rlatu, rlatv)`` for the unzoomed dyn3d mesh.

    ``rlonv(i) = -π + (i-1) · 2π / iim`` and ``rlonu(i) = rlonv(i) + π/iim``.
    Index conventions follow the Fortran 1-based layout in LMDZ; the arrays
    returned here are 0-indexed Python arrays of the same length (``iip1`` for
    the longitudes, ``jjp1`` for ``rlatu``, ``jjm`` for ``rlatv``).
    """
    iip1 = iim + 1
    jjp1 = jjm + 1
    i = np.arange(iip1, dtype=float)
    rlonv = -np.pi + i * (2.0 * np.pi / iim)
    rlonu = rlonv + np.pi / iim
    # rlatu spans pole-to-pole with rlatu[0] = +π/2 and rlatu[-1] = -π/2.
    j = np.arange(jjp1, dtype=float)
    rlatu = np.pi / 2.0 - j * np.pi / jjm
    # rlatv interleaves: rlatv[j] is the latitude between rlatu[j] and rlatu[j+1].
    rlatv = 0.5 * (rlatu[:-1] + rlatu[1:])
    return rlonu, rlonv, rlatu, rlatv


def _build_mesh_from_arrays(
    rlonu: np.ndarray,
    rlonv: np.ndarray,
    rlatu: np.ndarray,
    rlatv: np.ndarray,
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Assemble ``(verts, cells, centers)`` from the four 1-D coord arrays.

    Shared between the synthetic generator and the NetCDF dyn3d loader path.
    The cell-ordering convention is ``j`` outer (latitude band, north→south),
    ``i`` inner (longitude index 0..iim-1).
    """
    iip1 = len(rlonu)
    jjp1 = len(rlatu)
    iim = iip1 - 1
    jjm = jjp1 - 1

    # Interior corner vertices: each is at (rlonu[i], rlatv[j]).
    # i indexes the longitude *edges*; we only need 0..iim distinct longitudes
    # (the iip1-th wraps to the 0th). j indexes interior latitude edges.
    # Layout: corner_idx[j, i] is the vertex index of the corner at the
    # intersection of the j-th interior latitude edge and the i-th longitude
    # edge. The two poles are appended at the end.
    n_corner_lats = jjm - 1   # interior latitude edges between bands
    n_corner_lons = iim       # distinct longitudes; i = iim wraps to i = 0

    lon_edges = rlonu[:iim]                # shape (iim,)
    # rlatv has jjm entries; the interior latitude *edges* are at rlatv[0]..rlatv[jjm-2]
    # because rlatv[j] sits between rlatu[j] and rlatu[j+1]. The northernmost
    # interior edge (between band j=0 and j=1) is rlatv[0]; the southernmost
    # interior edge (between band j=jjm-1 and j=jjm) is rlatv[jjm-2].
    # Total interior latitude edges = jjm - 1 = n_corner_lats.
    interior_lats = rlatv[: jjm - 1] if jjm >= 2 else rlatv[:0]

    if n_corner_lats > 0:
        LON, LAT = np.meshgrid(lon_edges, interior_lats)   # shape (n_corner_lats, iim)
        interior_xyz = _lonlat_to_xyz(LON.ravel(), LAT.ravel())
    else:
        interior_xyz = np.zeros((0, 3), dtype=float)

    north_xyz = np.array([[0.0, 0.0, 1.0]])
    south_xyz = np.array([[0.0, 0.0, -1.0]])

    verts = np.vstack([interior_xyz, north_xyz, south_xyz])
    north_idx = len(interior_xyz)
    south_idx = north_idx + 1

    def corner(j_edge: int, i_lon: int) -> int:
        """Index of the corner vertex at interior edge j_edge, longitude i_lon."""
        return j_edge * n_corner_lons + (i_lon % n_corner_lons)

    cells: list[list[int]] = []
    centers_list: list[np.ndarray] = []

    # Iterate j from 0 (north polar band) to jjm-1 (south polar band).
    for j in range(jjm):
        is_north_pole_band = (j == 0)
        is_south_pole_band = (j == jjm - 1)
        for i in range(iim):
            if is_north_pole_band and jjm >= 2:
                # Triangle: north pole + two interior corners on edge j_edge=0.
                a = north_idx
                b = corner(0, i + 1)   # east corner first to keep CCW from outside
                c = corner(0, i)
                cells.append([a, b, c])
            elif is_south_pole_band and jjm >= 2:
                # Triangle: south pole + two interior corners on edge j_edge=jjm-2.
                a = south_idx
                b = corner(jjm - 2, i)
                c = corner(jjm - 2, i + 1)
                cells.append([a, b, c])
            elif jjm == 1:
                # Degenerate single-band case: every cell is a "bowtie"
                # spanning pole-to-pole. We still need iim cells; render each
                # as a triangle from north→south→back at one longitude pair.
                # (Not a realistic configuration but keeps the contract.)
                a = north_idx
                b = south_idx
                cells.append([a, b, a])
            else:
                # Quad: CCW from outside the sphere → NE, NW, SW, SE order
                # (north edge is j_edge = j-1, south edge is j_edge = j).
                ne = corner(j - 1, i + 1)
                nw = corner(j - 1, i)
                sw = corner(j, i)
                se = corner(j, i + 1)
                cells.append([ne, nw, sw, se])

            # Center: scalar lives at (rlonv[i], rlatu[j]) → projected to xyz.
            # For polar rows, rlatu[j] is exactly ±π/2, so the projection IS
            # the pole vertex.
            if is_north_pole_band:
                centers_list.append(north_xyz[0])
            elif is_south_pole_band:
                centers_list.append(south_xyz[0])
            else:
                c_xyz = _lonlat_to_xyz(np.array([rlonv[i]]),
                                       np.array([rlatu[j]]))[0]
                centers_list.append(c_xyz)

    centers = np.asarray(centers_list)
    return verts, cells, centers


def latlon_mesh(
    iim: int = 96,
    jjm: int = 95,
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Build a synthetic LMDZ-style regular lat-lon mesh on the unit sphere.

    Parameters
    ----------
    iim : int
        Number of distinct longitudes (default ``96``, LMDZ low-res).
    jjm : int
        Number of latitude bands (default ``95``, LMDZ low-res).

    Returns
    -------
    tuple
        ``(verts, cells, centers)``: ``verts`` is ``(V, 3)`` deduplicated unit
        sphere positions, ``cells`` is a list of vertex-index lists (3 entries
        for the two polar rows, 4 for interior bands), and ``centers`` is
        ``(iim * jjm, 3)`` cell centers. Polar cell centers coincide with the
        pole vertex (matching LMDZ's convention that polar scalars sit AT the
        pole).
    """
    if iim < 2:
        raise ValueError(f"iim must be >= 2 (got {iim})")
    if jjm < 1:
        raise ValueError(f"jjm must be >= 1 (got {jjm})")
    rlonu, rlonv, rlatu, rlatv = _dyn3d_coord_arrays(iim, jjm)
    return _build_mesh_from_arrays(rlonu, rlonv, rlatu, rlatv)
