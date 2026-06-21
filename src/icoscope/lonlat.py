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


def build_mesh_from_arrays(
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
                # West corner first so [pole, west, east] winds CCW viewed
                # from outside the sphere (right-hand rule: outward normal).
                a = north_idx
                b = corner(0, i)
                c = corner(0, i + 1)
                cells.append([a, b, c])
            elif is_south_pole_band and jjm >= 2:
                # Triangle: south pole + two interior corners on edge j_edge=jjm-2.
                # Order mirrors the north pole: from outside the south pole
                # (below the sphere) CCW reads the longitudes east-then-west.
                a = south_idx
                b = corner(jjm - 2, i + 1)
                c = corner(jjm - 2, i)
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


_NMAX_DEFAULT = 10000   # auxiliary-grid resolution; LMDZ uses 30000.


def _fhyp_x(xtild: np.ndarray, tau: float, dzoom: float) -> np.ndarray:
    """Evaluate LMDZ's ``fhyp`` for longitude on ``xtild ∈ [0, π]``.

    Mirrors the inner block of ``fxhyp_m.f90`` for ``x >= 0``. The result is
    extended by symmetry (``fhyp(-x) = fhyp(x)``) by the caller. The function
    is +1 well outside the zoom window and saturates to -1 inside it (in
    units of the auxiliary coordinate), with a smooth tanh transition whose
    sharpness is controlled by ``tau``.

    Singularities at ``x = 0`` and ``x = π`` are handled with the same
    200·fb-vs-fa branching that LMDZ uses to avoid 0/0.
    """
    pi_d = np.pi
    fa = tau * (0.5 * dzoom - xtild)
    fb = xtild * (pi_d - xtild)
    out = np.empty_like(xtild)
    # default tanh branch
    with np.errstate(divide="ignore", invalid="ignore"):
        tanh_arg = np.where(fb != 0.0, fa / fb, 0.0)
        out[:] = np.tanh(tanh_arg)
    out = np.where(200.0 * fb < -fa, -1.0, out)
    out = np.where(200.0 * fb < fa, 1.0, out)
    out = np.where(xtild == 0.0, 1.0, out)
    out = np.where(xtild == pi_d, -1.0, out)
    return out


def _fhyp_y(yt: np.ndarray, y0: float, tau: float, dzoom: float) -> np.ndarray:
    """Evaluate LMDZ's ``fhyp`` for latitude on ``yt ∈ [-π/2, π/2]``.

    Mirrors ``fyhyp_m.f90``: the zoom is built around ``y0 = clat`` directly
    in the natural coordinate, with the Heaviside-based denominator that
    keeps the formula well-defined on either side of ``y0``.
    """
    pi = np.pi
    pis2 = 0.5 * pi
    heavyy0m = 1.0 if -y0 > 0.0 else (0.5 if -y0 == 0.0 else 0.0)
    heavyy0 = 1.0 if y0 > 0.0 else (0.5 if y0 == 0.0 else 0.0)

    fa = np.zeros_like(yt)
    fb = np.zeros_like(yt)

    left = yt < y0
    right = yt > y0
    # left of y0
    fa[left] = tau * (yt[left] - y0 + 0.5 * dzoom)
    fb[left] = (yt[left] - 2.0 * y0 * heavyy0m + pis2) * (y0 - yt[left])
    # right of y0
    fa[right] = tau * (y0 - yt[right] + 0.5 * dzoom)
    fb[right] = (2.0 * y0 * heavyy0 - yt[right] + pis2) * (yt[right] - y0)

    out = np.empty_like(yt)
    with np.errstate(divide="ignore", invalid="ignore"):
        tanh_arg = np.where(fb != 0.0, fa / fb, 0.0)
        out[:] = np.tanh(tanh_arg)
    out = np.where(200.0 * fb < -fa, -1.0, out)
    out = np.where(200.0 * fb < fa, 1.0, out)

    y0min = 2.0 * y0 * heavyy0m - pis2
    y0max = 2.0 * y0 * heavyy0 + pis2
    out = np.where(yt == y0, 1.0, out)
    out = np.where((yt == y0min) | (yt == y0max), -1.0, out)
    return out


def _tanh_coord_1d(
    n: int,
    half_domain: float,
    center: float,
    grossism: float,
    dzoom_frac: float,
    tau: float,
    *,
    is_latitude: bool,
    nmax: int = _NMAX_DEFAULT,
    error_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """LMDZ tanh-zoom coord generator for one dimension.

    Parameters
    ----------
    n : int
        Number of cells along this dimension (``iim`` for longitude, ``jjm``
        for latitude).
    half_domain : float
        ``π`` for longitude (full range ``2π``) or ``π/2`` for latitude
        (full range ``π``).
    center : float
        Zoom focal point in radians (``clon`` or ``clat``).
    grossism : float
        Refinement factor ``G ≥ 1``. The cell-size ratio at the focal
        point is ~``1/G``.
    dzoom_frac : float
        Half-width of the zoom window, as a fraction of the full domain
        (LMDZ's ``dzoomx`` / ``dzoomy``; typical 0.05 … 0.3).
    tau : float
        Transition sharpness (LMDZ's ``taux`` / ``tauy``; typical 1 … 6).
    is_latitude : bool
        Selects between the longitude (fxhyp) and latitude (fyhyp) variants.
    nmax : int
        Auxiliary-grid half-resolution. LMDZ uses 30000; 10000 is enough
        for visualization (cell positions accurate to ~1e-7 rad).
    error_name : str
        Suffix to embed in the validity-error message (``"x"`` or ``"y"``).

    Returns
    -------
    tuple of np.ndarray
        ``(edges, centers)`` — arrays of length ``n + 1`` and ``n``
        respectively, in radians, on the natural domain (then shifted by
        ``center`` and wrapped). For longitude, edges == ``rlonu``,
        centers == ``rlonv``. For latitude, ``rlatu`` (the *band* centers)
        plays the role of ``edges`` (length ``jjm+1``, pole-to-pole) and
        ``rlatv`` (the *band* boundaries) plays the role of ``centers``
        (length ``jjm``, interior-only). The caller passes them to
        ``build_mesh_from_arrays`` in that role-swapped order — see
        ``latlon_mesh``.
    """
    pi_d = float(half_domain)        # π or π/2
    full = 2.0 * pi_d                 # 2π or π
    dzoom = dzoom_frac * full

    # 1. Auxiliary grid xtild ∈ [-pi_d, +pi_d], 2*nmax intervals.
    xtild = np.linspace(-pi_d, pi_d, 2 * nmax + 1)

    # 2. Evaluate fhyp on the auxiliary grid.
    if is_latitude:
        fhyp = _fhyp_y(xtild, center, tau, dzoom)
    else:
        # Symmetry: fhyp(-x) = fhyp(|x|). Build on [0, pi_d] then mirror.
        fhyp_pos = _fhyp_x(np.abs(xtild), tau, dzoom)
        fhyp = fhyp_pos

    # 3. Midpoint integral for ffdx / ffdy (trapezoid-equivalent given
    #    midpoint evaluation on uniform spacing).
    xmid = 0.5 * (xtild[:-1] + xtild[1:])
    fxm = (
        _fhyp_y(xmid, center, tau, dzoom)
        if is_latitude
        else _fhyp_x(np.abs(xmid), tau, dzoom)
    )
    dx = xtild[1] - xtild[0]
    # LMDZ's lat path integrates over the full range; lon path uses [0, pi_d]
    # and doubles (symmetry). Trapezoid over the full grid is equivalent and
    # avoids edge-case branching.
    ffdx = float(np.sum(fxm) * dx)

    # 4. Solve for beta. We integrate over the FULL [-pi_d, +pi_d] range, so
    #    the formula uses `full` (the total domain length), not pi_d. LMDZ's
    #    longitude path integrates over [0, π] only and uses pi (the half
    #    domain), but it's an equivalent factor-of-2 cancellation.
    denom = ffdx - full
    if abs(denom) < 1e-30:
        raise ValueError(
            f"Bad choice of grossism{error_name}, tau{error_name}, "
            f"dzoom{error_name}. Decrease dzoom{error_name} or "
            f"grossism{error_name}."
        )
    beta = (grossism * ffdx - full) / denom

    if 2.0 * beta - grossism <= 0.0:
        raise ValueError(
            f"Bad choice of grossism{error_name}, tau{error_name}, "
            f"dzoom{error_name}. Decrease dzoom{error_name} or "
            f"grossism{error_name}."
        )

    # 5. Density Xprimt at the auxiliary nodes.
    Xprimt = beta + (grossism - beta) * fhyp
    # And at midpoints (used for the cumulative integral).
    xxpr = beta + (grossism - beta) * fxm

    # 6. Cumulative integral Xf on the auxiliary grid.
    Xf = np.empty_like(xtild)
    Xf[0] = -pi_d
    Xf[1:] = -pi_d + np.cumsum(xxpr * dx)
    # Pin the last node exactly (cancels accumulated rounding).
    Xf[-1] = pi_d

    # 7. For each target index i, solve Xf(λ) = i*full/n - pi_d.
    #    We have Xf monotone (since Xprimt > 0); use bracketed Newton on a
    #    piecewise-linear interpolant, with bisection fallback. Vectorized
    #    inversion via np.interp is enough for visualization precision.

    def _invert(target_indices: np.ndarray) -> np.ndarray:
        targets = -pi_d + target_indices * full / n  # uniform image space
        # The "image space" of Xf is exactly [-pi_d, +pi_d]; values outside
        # that range correspond to the periodic continuation of rlonu past
        # the wrap. Bring the target into [-pi_d, +pi_d] for the inversion,
        # then put back the period offset on the output.
        period_offset = np.zeros_like(targets)
        if not is_latitude:
            # Longitude is periodic; lat is not (it bounds at the poles).
            shifted = ((targets + pi_d) % full) - pi_d
            period_offset = targets - shifted
            targets = shifted
        out = np.interp(targets, Xf, xtild)
        # Newton refinement using the density: f(λ) ≈ target + delta;
        # δλ = (target - Xf(λ)) / Xprimt(λ). Use linear interp for both.
        for _ in range(3):
            Xf_at = np.interp(out, xtild, Xf)
            Xp_at = np.interp(out, xtild, Xprimt)
            out = out + (targets - Xf_at) / np.maximum(Xp_at, 1e-30)
            out = np.clip(out, -pi_d, pi_d)
        return out + period_offset

    # Longitude convention from LMDZ:
    #   rlonv(i) = invert at offset 0     (cell centers)
    #   rlonu(i) = invert at offset 0.5   (cell edges, shifted half-cell east)
    # Both arrays have length n+1; the last entry is the periodic wrap of
    # the first and build_mesh_from_arrays uses only [:n] for the longitude
    # arrays. For latitude, all n+1 entries of the "edges role" are used
    # (pole-to-pole), and the "centers role" is internally truncated to n.
    iv = np.arange(n + 1, dtype=float)               # 0 .. n
    centers_arr = _invert(iv)                        # length n + 1
    edges_arr = _invert(iv + 0.5)                    # length n + 1

    # 8. Apply the center offset for longitude (the zoom was built on the
    #    natural [-π, +π] domain, centered at 0; LMDZ then shifts by clon).
    #    For latitude the center is already baked into fhyp_y so we don't
    #    shift here. Don't wrap: build_mesh_from_arrays / _lonlat_to_xyz
    #    are 2π-periodic and the arrays must stay monotone.
    if not is_latitude:
        centers_arr = centers_arr + center
        edges_arr = edges_arr + center

    return edges_arr, centers_arr


def latlon_mesh(
    iim: int = 96,
    jjm: int = 95,
    clon: float = 0.0,
    clat: float = 0.0,
    grossismx: float = 1.0,
    grossismy: float = 1.0,
    dzoomx: float = 0.0,
    dzoomy: float = 0.0,
    taux: float = 3.0,
    tauy: float = 3.0,
) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Build a synthetic LMDZ-style regular lat-lon mesh on the unit sphere.

    Parameters
    ----------
    iim : int
        Number of distinct longitudes (default ``96``, LMDZ low-res).
    jjm : int
        Number of latitude bands (default ``95``, LMDZ low-res).
    clon, clat : float
        Zoom focal point in degrees. Ignored when ``grossismx == 1`` and
        ``grossismy == 1``.
    grossismx, grossismy : float
        LMDZ's longitudinal/latitudinal refinement factor (``≥ 1``). At
        ``1.0`` (default) the mesh is uniform; values above 1 concentrate
        cells near ``(clon, clat)``.
    dzoomx, dzoomy : float
        Half-width of the zoom window, as a fraction of the full domain
        (LMDZ convention: ``dzoomx ∈ [0, 0.5]`` for the 2π longitude range,
        ``dzoomy ∈ [0, 0.5]`` for the π latitude range).
    taux, tauy : float
        Transition sharpness of the tanh profile. Typical 1 … 6.

    Returns
    -------
    tuple
        ``(verts, cells, centers)``: ``verts`` is ``(V, 3)`` deduplicated unit
        sphere positions, ``cells`` is a list of vertex-index lists (3 entries
        for the two polar rows, 4 for interior bands), and ``centers`` is
        ``(iim * jjm, 3)`` cell centers. Polar cell centers coincide with the
        pole vertex (matching LMDZ's convention that polar scalars sit AT the
        pole).

    Raises
    ------
    ValueError
        If the chosen ``(grossism, tau, dzoom)`` triple violates LMDZ's
        ``2β - grossism > 0`` validity condition (typically when
        ``grossism · dzoom`` is too large).
    """
    if iim < 2:
        raise ValueError(f"iim must be >= 2 (got {iim})")
    if jjm < 1:
        raise ValueError(f"jjm must be >= 1 (got {jjm})")

    uniform = (
        abs(grossismx - 1.0) < 1e-12
        and abs(grossismy - 1.0) < 1e-12
    )
    if uniform:
        rlonu, rlonv, rlatu, rlatv = _dyn3d_coord_arrays(iim, jjm)
        return build_mesh_from_arrays(rlonu, rlonv, rlatu, rlatv)

    # Longitude: half-domain π, full 2π. _tanh_coord_1d returns
    # (edges_arr, centers_arr) both of length iim+1 — matching rlonu/rlonv.
    rlonu, rlonv = _tanh_coord_1d(
        iim, half_domain=np.pi, center=np.radians(clon),
        grossism=grossismx, dzoom_frac=dzoomx, tau=taux,
        is_latitude=False, error_name="x",
    )
    # Latitude: half-domain π/2, full π. The helper returns
    # (edges_arr, centers_arr) both of length jjm+1. In LMDZ:
    #   rlatu has length jjm+1 (band centers, pole-to-pole)
    #   rlatv has length jjm   (interior band boundaries)
    # The helper's "centers" role plays rlatu (offset 0, includes poles)
    # and "edges" role plays rlatv (offset 0.5, interior boundaries).
    # Truncate rlatv to length jjm. Then flip so latitudes run north→south.
    lat_edges_full, lat_centers_full = _tanh_coord_1d(
        jjm, half_domain=np.pi / 2.0, center=np.radians(clat),
        grossism=grossismy, dzoom_frac=dzoomy, tau=tauy,
        is_latitude=True, error_name="y",
    )
    rlatu = lat_centers_full[::-1]               # length jjm+1, +π/2 → -π/2
    rlatv = lat_edges_full[:jjm][::-1]           # length jjm
    # Pin the poles exactly to avoid floating-point drift at ±π/2.
    rlatu[0] = np.pi / 2.0
    rlatu[-1] = -np.pi / 2.0

    return build_mesh_from_arrays(rlonu, rlonv, rlatu, rlatv)
