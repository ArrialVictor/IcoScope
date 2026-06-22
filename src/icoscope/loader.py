"""Load a NetCDF grid (icosahedral DYNAMICO/ICOLMDZ, or regular lat-lon).

Three grid families are auto-detected, tried in order:

1. LMDZ dyn3d regular lat-lon (Arakawa C-grid) — sniffed by the presence of
   all four coord variables ``rlonu``, ``rlatu``, ``rlonv``, ``rlatv``. Cell
   polygons are reconstructed from the coord arrays. Data variables shaped
   ``(rlatu, rlonv)`` or ``(time, rlatu, rlonv)`` are flattened to the
   cell-flat layout via :func:`_flatten_dyn3d_field`.

2. CF-convention "bounds" layout (DYNAMICO / ICOLMDZ icosahedral)::

       lon(cell)                                ; bounds = "bounds_lon"
       lat(cell)                                ; bounds = "bounds_lat"
       bounds_lon(cell, nvertex)
       bounds_lat(cell, nvertex)

   Variable names are auto-detected (``lon``/``longitude``/``cell_lon`` etc.).
   Pentagons are detected by duplicate consecutive vertices in the bounds array.

3. XIOS-interpolated regular lat-lon (the standard ICOLMDZ analysis-output
   format) — 1-D ``lon`` and ``lat`` cell-center coords with no CF bounds and
   no dyn3d staggered arrays. Latitude endpoints must be ``±90°``. Data
   variables shaped ``(lat, lon)``, ``(time, lat, lon)``, ``(presnivs, lat,
   lon)``, and ``(time, presnivs, lat, lon)`` are all surfaced and flattened
   via :func:`_flatten_xios_field`.

Beyond the grid, :func:`load_grid` returns metadata for every cell-shaped
data variable (including ``n_levels`` ≥ 1 when the variable has a vertical
``presnivs`` dim) so the GUI can populate a "Color by" dropdown and a level
slider. :func:`read_field` fetches a single field's values on demand,
slicing optional ``time`` and ``presnivs`` dims first and handling the
dyn3d / XIOS 2-D flatten transparently. :func:`read_levels` returns the
``presnivs`` axis values (Pa) for slider labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

FieldMeta = dict[str, Any]
# Expected keys: units (str), long_name (str), shape (tuple), time_varying (bool),
# n_levels (int — 0 if no vertical dim, otherwise the size of the `presnivs`
# dim). dyn3d / XIOS files additionally set "kind" so callers can branch.

LEVEL_DIM = "presnivs"
TIME_DIMS = ("time", "time_counter")

CENTER_LON_NAMES = ("lon", "longitude", "cell_lon", "clon")
CENTER_LAT_NAMES = ("lat", "latitude", "cell_lat", "clat")
BOUNDS_LON_NAMES = ("bounds_lon", "lon_bounds", "clon_bnds", "lon_bnds")
BOUNDS_LAT_NAMES = ("bounds_lat", "lat_bounds", "clat_bnds", "lat_bnds")

EXCLUDED = set(CENTER_LON_NAMES + CENTER_LAT_NAMES
               + BOUNDS_LON_NAMES + BOUNDS_LAT_NAMES
               + ("time", "nvertex", "cell"))

# LMDZ dyn3d (classical regular lat-lon) signature. A file is treated as dyn3d
# when ALL four of these 1-D coord arrays are present.
DYN3D_COORDS = ("rlonu", "rlatu", "rlonv", "rlatv")


def _find(ds, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"none of {candidates} found in NetCDF variables")


def _lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    return np.stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ], axis=-1)


def load_grid(
    path: str | Path,
) -> tuple[np.ndarray, list[list[int]], np.ndarray, dict[str, FieldMeta]]:
    """Read the mesh + per-field metadata from a NetCDF file.

    Auto-routes between the CF-bounds icosahedral path and the LMDZ dyn3d
    path based on the presence of the dyn3d coord arrays.

    Parameters
    ----------
    path : str or Path
        Path to a CF-convention icosahedral NetCDF file, or to an LMDZ
        dyn3d file with the four ``rlonu/rlatu/rlonv/rlatv`` coord arrays.

    Returns
    -------
    verts : ndarray
        ``(V, 3)`` deduplicated vertex positions on the unit sphere.
    cells : list of list of int
        Vertex-index lists; length 5 or 6 for icosahedral cells, 3 (polar)
        or 4 (interior) for dyn3d cells.
    centers : ndarray
        ``(C, 3)`` cell-center positions on the unit sphere.
    fields : dict
        ``{name: FieldMeta}`` for every cell-shaped data variable. Field
        values themselves are fetched on demand via :func:`read_field` so
        the underlying Dataset handle is closed by the time this returns.
    """
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        if all(name in ds.variables for name in DYN3D_COORDS):
            return _load_dyn3d_grid(ds)
        if not _has_cf_bounds(ds):
            return _load_xios_latlon_grid(ds)
        center_lon_name = _find(ds, CENTER_LON_NAMES)
        clon = np.asarray(ds.variables[center_lon_name][:])
        clat = np.asarray(ds.variables[_find(ds, CENTER_LAT_NAMES)][:])
        blon = np.asarray(ds.variables[_find(ds, BOUNDS_LON_NAMES)][:])
        blat = np.asarray(ds.variables[_find(ds, BOUNDS_LAT_NAMES)][:])

        cell_dim = ds.variables[center_lon_name].dimensions[0]

        fields = {}
        for name, var in ds.variables.items():
            if name in EXCLUDED:
                continue
            if cell_dim not in var.dimensions:
                continue
            time_varying = len(var.dimensions) > 1 and var.dimensions[0] != cell_dim
            fields[name] = {
                "units": getattr(var, "units", ""),
                "long_name": getattr(var, "long_name", name),
                "shape": tuple(var.shape),
                "time_varying": time_varying,
                "n_levels": 0,   # icosahedral path: vertical fields not yet supported
            }

    n_cells, n_max = blon.shape
    centers = _lonlat_to_xyz(clon, clat)

    flat_xyz = _lonlat_to_xyz(blon.reshape(-1), blat.reshape(-1))
    keys = np.round(flat_xyz, 8)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    unique_first = np.array([np.where(inverse == u)[0][0]
                             for u in np.unique(inverse)])
    verts = flat_xyz[unique_first]
    remap = -np.ones(int(inverse.max()) + 1, dtype=np.int64)
    remap[np.unique(inverse)] = np.arange(len(unique_first))
    inv2d = remap[inverse].reshape(n_cells, n_max)

    cells = []
    for row in inv2d:
        cell = [int(row[0])]
        for v in row[1:]:
            if int(v) != cell[-1]:
                cell.append(int(v))
        if cell[0] == cell[-1] and len(cell) > 1:
            cell.pop()
        cells.append(cell)

    return verts, cells, centers, fields


def _load_dyn3d_grid(
    ds,
) -> tuple[np.ndarray, list[list[int]], np.ndarray, dict[str, FieldMeta]]:
    """Build mesh and collect field metadata from an LMDZ dyn3d NetCDF.

    Reads the four ``rlonu/rlonv/rlatu/rlatv`` coord arrays and reconstructs
    cell polygons via :func:`icoscope.lonlat.build_mesh_from_arrays`. Then
    scans all data variables for ones shaped ``(rlatu, rlonv)`` or
    ``(time, rlatu, rlonv)`` and exposes them in the returned ``fields``
    dict so the GUI's "Color by" combo can offer them. Values are fetched on
    demand via :func:`read_field`, which flattens the 2-D ``(j, i)`` layout
    to the cell-flat layout (``j`` outer, ``i`` inner) used by the renderer.
    """
    from .lonlat import build_mesh_from_arrays

    rlonu = np.asarray(ds.variables["rlonu"][:], dtype=float)
    rlonv = np.asarray(ds.variables["rlonv"][:], dtype=float)
    rlatu = np.asarray(ds.variables["rlatu"][:], dtype=float)
    rlatv = np.asarray(ds.variables["rlatv"][:], dtype=float)
    verts, cells, centers = build_mesh_from_arrays(rlonu, rlonv, rlatu, rlatv)

    fields: dict[str, FieldMeta] = {}
    # A dyn3d data variable lives on the scalar grid (rlatu × rlonv), with an
    # optional leading time dim and optional vertical (presnivs) dim. Drop the
    # four coord arrays themselves and anything purely descriptive.
    coord_dims = ("rlatu", "rlonv")
    coord_var_names = set(DYN3D_COORDS) | {
        "time", "time_counter", "nav_lat", "nav_lon",
        "controle", "tab_cntrl", "presnivs", "sigs", "sig",
    }
    n_levels_avail = len(ds.dimensions[LEVEL_DIM]) if LEVEL_DIM in ds.dimensions else 0
    for name, var in ds.variables.items():
        if name in coord_var_names:
            continue
        meta = _classify_var(var.dimensions, coord_dims, n_levels_avail)
        if meta is None:
            continue
        time_varying, n_levels = meta
        fields[name] = {
            "units": getattr(var, "units", ""),
            "long_name": getattr(var, "long_name", name),
            "shape": tuple(var.shape),
            "time_varying": time_varying,
            "n_levels": n_levels,
            "kind": "dyn3d",
        }

    return verts, cells, centers, fields


def _classify_var(
    dims: tuple[str, ...],
    horizontal: tuple[str, str],
    n_levels_avail: int,
) -> tuple[bool, int] | None:
    """Return ``(time_varying, n_levels)`` for a variable, or ``None`` to skip.

    Accepts variables whose dims end in ``horizontal`` (e.g. ``(rlatu, rlonv)``
    for dyn3d, ``(lat, lon)`` for XIOS) and whose leading dims are some prefix
    of ``[time, presnivs]``. Other layouts are skipped.
    """
    if dims[-2:] != horizontal:
        return None
    leading = dims[:-2]
    if len(leading) == 0:
        return False, 0
    if len(leading) == 1 and leading[0] in TIME_DIMS:
        return True, 0
    if len(leading) == 1 and leading[0] == LEVEL_DIM:
        return False, n_levels_avail
    if len(leading) == 2 and leading[0] in TIME_DIMS and leading[1] == LEVEL_DIM:
        return True, n_levels_avail
    return None


def _flatten_dyn3d_field(arr: np.ndarray) -> np.ndarray:
    """Flatten a dyn3d 2-D field ``(jjp1, iim_or_iip1)`` to the cell-flat layout.

    Cell ordering is ``j`` outer, ``i`` inner (matches
    :func:`icoscope.lonlat.build_mesh_from_arrays`), so the result is a
    1-D array of length ``jjm * iim`` (the mesh's cell count).

    Row mapping:
    - mesh row 0 (north polar band) ← field row 0 (north pole)
    - mesh rows 1..jjm-2 (interior) ← field rows 1..jjm-2 (rlatu values used
      by the mesh's interior cell centers)
    - mesh row jjm-1 (south polar band) ← field row jjp1-1 (south pole)

    Field row ``jjm-1`` is skipped: the synthetic mesh has ``jjm = jjp1 - 1``
    cell rows whereas LMDZ stores ``jjp1`` scalar rows, and the mesh's
    south-polar band sits at the south pole (field row ``jjp1-1``) rather
    than at the southernmost interior latitude (field row ``jjm-1``). The
    trailing periodic-duplicate column at ``i = iim`` (when present) is
    dropped.
    """
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D field, got shape {arr.shape}")
    jjp1, ni = arr.shape
    iim = ni - 1 if ni >= 2 else ni     # drop the periodic duplicate column
    jjm = jjp1 - 1
    out = np.empty((jjm, iim), dtype=arr.dtype)
    out[: jjm - 1, :] = arr[: jjm - 1, :iim]   # north pole + interior rows used
    out[jjm - 1, :] = arr[jjp1 - 1, :iim]       # south pole (skips field row jjm-1)
    return out.reshape(-1)


def _has_cf_bounds(ds) -> bool:
    """True if the dataset exposes both a ``bounds_lon`` and ``bounds_lat`` variable."""
    return (any(b in ds.variables for b in BOUNDS_LON_NAMES)
            and any(b in ds.variables for b in BOUNDS_LAT_NAMES))


def _is_xios_latlon(ds) -> bool:
    """True if the dataset looks like XIOS-interpolated regular lat-lon.

    Sniffed by the presence of 1-D ``lon`` and ``lat`` coord variables that
    sit on their own same-named dimension, with no CF bounds and no dyn3d
    staggered arrays. The dyn3d and bounds checks are assumed to have been
    done by the caller — this function is the third-and-final detector.
    """
    if "lon" not in ds.variables or "lat" not in ds.variables:
        return False
    lon = ds.variables["lon"]
    lat = ds.variables["lat"]
    return (lon.ndim == 1 and lat.ndim == 1
            and lon.dimensions == ("lon",) and lat.dimensions == ("lat",))


def _load_xios_latlon_grid(
    ds,
) -> tuple[np.ndarray, list[list[int]], np.ndarray, dict[str, FieldMeta]]:
    """Build mesh and collect field metadata from an XIOS regular lat-lon file.

    Reads 1-D ``lon`` and ``lat`` cell-center coords and reconstructs cell
    polygons via :func:`icoscope.lonlat.build_mesh_from_centers`. Surfaces
    data variables shaped ``(lat, lon)``, ``(time, lat, lon)``, ``(presnivs,
    lat, lon)``, and ``(time, presnivs, lat, lon)``. Vertical layouts get
    ``n_levels`` set to the size of the ``presnivs`` dim so the GUI can show
    a level slider.
    """
    from .lonlat import build_mesh_from_centers

    if not _is_xios_latlon(ds):
        raise ValueError(
            "no recognised grid layout: missing dyn3d coord arrays, CF bounds, "
            "or 1-D lon/lat XIOS coords"
        )
    lon = np.asarray(ds.variables["lon"][:], dtype=float)
    lat = np.asarray(ds.variables["lat"][:], dtype=float)
    verts, cells, centers, _south_first = build_mesh_from_centers(lon, lat)

    fields: dict[str, FieldMeta] = {}
    coord_dims = ("lat", "lon")
    skip_names = {"lon", "lat", "presnivs", "time", "time_counter",
                  "time_centered", "time_counter_bounds", "time_centered_bounds",
                  "nav_lon", "nav_lat"}
    n_levels_avail = len(ds.dimensions[LEVEL_DIM]) if LEVEL_DIM in ds.dimensions else 0
    for name, var in ds.variables.items():
        if name in skip_names:
            continue
        meta = _classify_var(var.dimensions, coord_dims, n_levels_avail)
        if meta is None:
            continue
        time_varying, n_levels = meta
        fields[name] = {
            "units": getattr(var, "units", ""),
            "long_name": getattr(var, "long_name", name),
            "shape": tuple(var.shape),
            "time_varying": time_varying,
            "n_levels": n_levels,
            "kind": "xios",
        }

    return verts, cells, centers, fields


def _flatten_xios_field(arr: np.ndarray, south_first: bool) -> np.ndarray:
    """Flatten an XIOS 2-D ``(lat, lon)`` field to the cell-flat layout.

    The mesh built by :func:`icoscope.lonlat.build_mesh_from_centers` iterates
    ``j`` north→south (outer) and ``i`` unchanged (inner), matching
    :func:`icoscope.lonlat.build_mesh_from_arrays`. When the source file
    stores lat south→north (``south_first=True``), the lat axis is flipped
    here so the field aligns with the mesh.

    Row mapping (same skip pattern as :func:`_flatten_dyn3d_field`):
    - mesh row 0 (north polar band) ← field row 0 (north pole scalar)
    - mesh rows 1..jjm-2 (interior) ← field rows 1..jjm-2
    - mesh row jjm-1 (south polar band) ← field row nlat-1 (south pole scalar)

    Field row ``jjm-1 = nlat-2`` is skipped: the mesh forces the south polar
    band's center to the pole, so it must be colored by the south-pole scalar
    rather than the southernmost interior latitude.
    """
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D field, got shape {arr.shape}")
    if south_first:
        arr = arr[::-1, :]
    nlat, nlon = arr.shape
    jjm = nlat - 1
    out = np.empty((jjm, nlon), dtype=arr.dtype)
    out[: jjm - 1, :] = arr[: jjm - 1, :]
    out[jjm - 1, :] = arr[nlat - 1, :]
    return out.reshape(-1)


def read_global_attrs(path: str | Path) -> dict[str, str]:
    """Return the NetCDF file's global attributes as a ``{name: str}`` dict.

    Only string-valued attributes are kept; non-string values are coerced via
    ``str()``. Missing or unreadable files raise; absent attributes simply
    don't appear in the returned dict.
    """
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        return {name: str(ds.getncattr(name)) for name in ds.ncattrs()}


def read_field(
    path: str | Path, name: str, time_index: int = 0, level_index: int = 0,
) -> np.ndarray:
    """Read a field's values. Returns a 1-D array of length ``n_cells``.

    Slices the leading ``time`` dim (if present) at ``time_index``, then the
    ``presnivs`` dim (if present) at ``level_index``, then dispatches to the
    dyn3d / XIOS flatten as appropriate. The icosahedral CF-bounds path
    returns the time-sliced 1-D ``(cell,)`` array directly.

    The leading-dim indices are pushed down to the netCDF4 lazy slicer so
    only the needed slab is read from disk — about 40× faster than reading
    the full variable then slicing in Python, for a typical 79-level x 28-step
    XIOS file. ``level_index`` is ignored for fields without a vertical dim.
    """
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        if name not in ds.variables:
            raise KeyError(f"field '{name}' not in {path}")
        var = ds.variables[name]
        dims = var.dimensions

        # Build a slice tuple that picks just (time_index, level_index, :, :)
        # so netCDF4 reads only the needed slab — full var[:] would pull the
        # entire 4-D array.
        idx: list[int | slice] = []
        consumed = 0
        if dims and dims[0] in TIME_DIMS:
            idx.append(time_index)
            consumed += 1
        if len(dims) > consumed and dims[consumed] == LEVEL_DIM:
            idx.append(level_index)
            consumed += 1
        idx.extend([slice(None)] * (len(dims) - consumed))
        data = np.asarray(var[tuple(idx)])

        is_dyn3d = all(c in ds.variables for c in DYN3D_COORDS)
        is_xios = not is_dyn3d and not _has_cf_bounds(ds) and _is_xios_latlon(ds)
        xios_south_first = bool(is_xios and ds.variables["lat"][0] < ds.variables["lat"][-1])

    if is_dyn3d and data.ndim == 2:
        return _flatten_dyn3d_field(data)
    if is_xios and data.ndim == 2:
        return _flatten_xios_field(data, xios_south_first)
    return data


def read_levels(path: str | Path) -> np.ndarray | None:
    """Return the ``presnivs`` axis values (Pa) if present, else ``None``.

    Used by the GUI to label the vertical-level slider. Returns ``None`` for
    files with no vertical dim (the slider stays hidden).
    """
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        if LEVEL_DIM not in ds.variables:
            return None
        return np.asarray(ds.variables[LEVEL_DIM][:], dtype=float)


def describe(path: str | Path) -> None:
    """Print the file's dimensions and variable shapes (for unknown layouts)."""
    from netCDF4 import Dataset

    print(f"\n=== {Path(path).name} ===")
    with Dataset(path) as ds:
        print("Dimensions:")
        for k, v in ds.dimensions.items():
            print(f"  {k:20s} = {len(v)}")
        print("Variables:")
        for k, v in ds.variables.items():
            print(f"  {k:20s} {v.dtype} {v.shape}")
