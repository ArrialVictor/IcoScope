"""Load a DYNAMICO / ICOLMDZ grid from NetCDF.

Targets the CF-convention "bounds" pattern:

    lon(cell)                                ; bounds = "bounds_lon"
    lat(cell)                                ; bounds = "bounds_lat"
    bounds_lon(cell, nvertex)
    bounds_lat(cell, nvertex)

Variable names are auto-detected (lon vs longitude vs cell_lon, etc.).
Pentagons are detected by duplicate consecutive vertices in the bounds array.

Beyond the grid, `load_grid()` also returns metadata for every cell-shaped
data variable so the GUI can populate a "Color by" dropdown. `read_field()`
fetches a single field's values on demand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

FieldMeta = dict[str, Any]  # {"units": str, "long_name": str, "shape": tuple, "time_varying": bool}

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

    Args:
        path: path to a CF-convention icosahedral NetCDF file.

    Returns
    -------
        ``(verts, cells, centers, fields)`` where ``verts`` is ``(V, 3)``
        deduplicated vertex positions on the unit sphere, ``cells`` is a list
        of vertex-index lists (length 5 or 6), ``centers`` is ``(C, 3)`` cell
        centers, and ``fields`` is a dict of per-field metadata. Field values
        themselves are fetched on demand via :func:`read_field` so the
        underlying Dataset handle is closed by the time this returns.
    """
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        if all(name in ds.variables for name in DYN3D_COORDS):
            return _load_dyn3d_grid(ds)
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
    cell polygons via :func:`icoscope.lonlat._build_mesh_from_arrays`. Then
    scans all data variables for ones shaped ``(rlatu, rlonv)`` or
    ``(time, rlatu, rlonv)`` and exposes them in the returned ``fields``
    dict so the GUI's "Color by" combo can offer them. Values are fetched on
    demand via :func:`read_field`, which flattens the 2-D ``(j, i)`` layout
    to the cell-flat layout (``j`` outer, ``i`` inner) used by the renderer.
    """
    from .lonlat import _build_mesh_from_arrays

    rlonu = np.asarray(ds.variables["rlonu"][:], dtype=float)
    rlonv = np.asarray(ds.variables["rlonv"][:], dtype=float)
    rlatu = np.asarray(ds.variables["rlatu"][:], dtype=float)
    rlatv = np.asarray(ds.variables["rlatv"][:], dtype=float)
    verts, cells, centers = _build_mesh_from_arrays(rlonu, rlonv, rlatu, rlatv)

    fields: dict[str, FieldMeta] = {}
    # A dyn3d data variable lives on the scalar grid (rlatu × rlonv), with an
    # optional leading time dim. Drop the four coord arrays themselves and
    # anything purely descriptive (controls, time, presnivs, …).
    coord_dims = ("rlatu", "rlonv")
    coord_var_names = set(DYN3D_COORDS) | {
        "time", "time_counter", "nav_lat", "nav_lon",
        "controle", "tab_cntrl", "presnivs", "sigs", "sig",
    }
    for name, var in ds.variables.items():
        if name in coord_var_names:
            continue
        dims = var.dimensions
        # Must end in (rlatu, rlonv). Anything else (e.g. (presnivs, rlatu, rlonv)
        # vertical profiles, or 1-D timeseries) is skipped — needs a different
        # UI than a flat 2-D scalar per cell.
        if len(dims) < 2 or dims[-2:] != coord_dims:
            continue
        time_varying = len(dims) >= 3 and dims[0] in ("time", "time_counter")
        if not time_varying and len(dims) > 2:
            # 3-D without time on the outside — vertical level, skip for now.
            continue
        fields[name] = {
            "units": getattr(var, "units", ""),
            "long_name": getattr(var, "long_name", name),
            "shape": tuple(var.shape),
            "time_varying": time_varying,
            "kind": "dyn3d",
        }

    return verts, cells, centers, fields


def _flatten_dyn3d_field(arr: np.ndarray) -> np.ndarray:
    """Flatten a dyn3d 2-D field ``(jjp1, iim_or_iip1)`` to the cell-flat layout.

    Cell ordering is ``j`` outer, ``i`` inner (matches
    :func:`icoscope.lonlat._build_mesh_from_arrays`), so the result is a
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


def read_global_attrs(path: str | Path) -> dict[str, str]:
    """Return the NetCDF file's global attributes as a ``{name: str}`` dict.

    Only string-valued attributes are kept; non-string values are coerced via
    ``str()``. Missing or unreadable files raise; absent attributes simply
    don't appear in the returned dict.
    """
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        return {name: str(ds.getncattr(name)) for name in ds.ncattrs()}


def read_field(path: str | Path, name: str, time_index: int = 0) -> np.ndarray:
    """Read a field's values. Returns a 1-D array of length ``n_cells``.

    If the variable is time-dependent (its first dim is the time dim),
    returns the slice at ``time_index``. For LMDZ dyn3d files (sniffed via
    the presence of all four ``rlonu/rlatu/rlonv/rlatv`` coord arrays), the
    resulting 2-D ``(rlatu, rlonv)`` slice is flattened to the cell-flat
    layout that matches :func:`icoscope.lonlat._build_mesh_from_arrays`.
    """
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        if name not in ds.variables:
            raise KeyError(f"field '{name}' not in {path}")
        var = ds.variables[name]
        data = np.asarray(var[:])
        is_dyn3d = all(c in ds.variables for c in DYN3D_COORDS)

    # Time slice first if the leading dim looks like time.
    if data.ndim > 1 and is_dyn3d:
        # dyn3d: shape (time?, rlatu, rlonv) → slice the time dim if present.
        if data.ndim == 3:
            data = data[time_index]
        return _flatten_dyn3d_field(data)
    if data.ndim > 1:
        data = data[time_index]
    return data


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
