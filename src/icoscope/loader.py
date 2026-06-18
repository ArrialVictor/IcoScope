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

    Returns:
        ``(verts, cells, centers, fields)`` where ``verts`` is ``(V, 3)``
        deduplicated vertex positions on the unit sphere, ``cells`` is a list
        of vertex-index lists (length 5 or 6), ``centers`` is ``(C, 3)`` cell
        centers, and ``fields`` is a dict of per-field metadata. Field values
        themselves are fetched on demand via :func:`read_field` so the
        underlying Dataset handle is closed by the time this returns.
    """
    from netCDF4 import Dataset

    with Dataset(path) as ds:
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


def read_field(path: str | Path, name: str, time_index: int = 0) -> np.ndarray:
    """Read a field's values. Returns a 1-D array of length ``n_cells``.

    If the variable is time-dependent (its first dim is not the cell dim),
    returns the slice at ``time_index``.
    """
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        if name not in ds.variables:
            raise KeyError(f"field '{name}' not in {path}")
        data = np.asarray(ds.variables[name][:])
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
