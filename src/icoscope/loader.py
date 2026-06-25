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
# time_dim_name (str | None — the leading time dim's name, e.g. "time_counter",
# or None for static fields), n_levels (int — 0 if no vertical dim, otherwise
# the size of the `presnivs` dim). dyn3d / XIOS files additionally set "kind"
# so callers can branch.

LEVEL_DIM = "presnivs"
# Time dim is named ``time`` in dyn3d, ``time_counter`` in XIOS, ``Time`` in
# WRF/MPAS-style files, plus a few rarer variants seen in CF outputs. read_field
# slices the leading dim only if its name is in this set, and load_grid mirrors
# the same set when deciding ``time_varying`` — keeping the two sides aligned.
TIME_DIMS = ("time", "time_counter", "Time", "t", "time_instant", "time_centered")

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
        n_levels_avail = (
            len(ds.dimensions[LEVEL_DIM]) if LEVEL_DIM in ds.dimensions else 0
        )

        fields = {}
        for name, var in ds.variables.items():
            if name in EXCLUDED:
                continue
            meta = _classify_var(var.dimensions, (cell_dim,), n_levels_avail)
            if meta is None:
                continue
            time_dim_name, n_levels = meta
            fields[name] = {
                "units": getattr(var, "units", ""),
                "long_name": getattr(var, "long_name", name),
                "shape": tuple(var.shape),
                "time_varying": time_dim_name is not None,
                "time_dim_name": time_dim_name,
                "n_levels": n_levels,
            }

    n_cells, n_max = blon.shape
    centers = _lonlat_to_xyz(clon, clat)

    flat_xyz = _lonlat_to_xyz(blon.reshape(-1), blat.reshape(-1))
    keys = np.round(flat_xyz, 8)
    # np.unique with return_index gives the first-occurrence index of each
    # unique vertex directly; an earlier draft did a Python list-comp of
    # np.where(inverse == u) which was O(n_unique × n_flat) — minutes on
    # a 643k-cell ICOLMDZ grid (~1.2M unique × ~3.9M flat).
    _, unique_idx, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True)
    verts = flat_xyz[unique_idx]
    inv2d = inverse.reshape(n_cells, n_max)

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
        time_dim_name, n_levels = meta
        fields[name] = {
            "units": getattr(var, "units", ""),
            "long_name": getattr(var, "long_name", name),
            "shape": tuple(var.shape),
            "time_varying": time_dim_name is not None,
            "time_dim_name": time_dim_name,
            "n_levels": n_levels,
            "kind": "dyn3d",
        }

    return verts, cells, centers, fields


def _classify_var(
    dims: tuple[str, ...],
    horizontal: tuple[str, ...],
    n_levels_avail: int,
) -> tuple[str | None, int] | None:
    """Return ``(time_dim_name, n_levels)`` for a variable, or ``None`` to skip.

    ``time_dim_name`` is the *actual name* of the variable's leading time
    dim (e.g. ``"time_counter"`` or ``"Time"``) when one is present, else
    ``None``. The name is used downstream to look up the variable's
    datetime axis values via :func:`read_times`. ``n_levels`` is the size
    of the ``presnivs`` dim when present, else 0.

    Accepts variables whose dims end in ``horizontal`` (e.g. ``(rlatu,
    rlonv)`` for dyn3d, ``(lat, lon)`` for XIOS, or just ``(cell,)`` for
    the icosahedral CF-bounds path) and whose leading dims are some
    prefix of ``[time, presnivs]``. Other layouts are skipped.
    """
    h_len = len(horizontal)
    if dims[-h_len:] != horizontal:
        return None
    leading = dims[:-h_len]
    if len(leading) == 0:
        return None, 0
    if len(leading) == 1 and leading[0] in TIME_DIMS:
        return leading[0], 0
    if len(leading) == 1 and leading[0] == LEVEL_DIM:
        return None, n_levels_avail
    if len(leading) == 2 and leading[0] in TIME_DIMS and leading[1] == LEVEL_DIM:
        return leading[0], n_levels_avail
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
        time_dim_name, n_levels = meta
        fields[name] = {
            "units": getattr(var, "units", ""),
            "long_name": getattr(var, "long_name", name),
            "shape": tuple(var.shape),
            "time_varying": time_dim_name is not None,
            "time_dim_name": time_dim_name,
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


class FileContext:
    """Hold a NetCDF Dataset open + cached sniffer results across reads.

    Building this object opens the file and runs the grid-family detectors
    once. Subsequent :func:`read_field` calls passed this context skip both
    the per-call ``Dataset(path)`` open and the sniffer work — measured 87 ms
    → 10 ms per slider tick on a 4 GB ICOLMDZ file, ~9× faster than the
    open-per-call path.

    The Dataset handle stays open for the context's lifetime. Call
    :meth:`close` (or use as a context manager) when the file is unloaded.
    """

    def __init__(self, path: str | Path):
        from netCDF4 import Dataset
        self.path = path
        self.ds = Dataset(path)
        self.is_dyn3d = all(c in self.ds.variables for c in DYN3D_COORDS)
        self.is_xios = (
            not self.is_dyn3d
            and not _has_cf_bounds(self.ds)
            and _is_xios_latlon(self.ds)
        )
        self.xios_south_first = bool(
            self.is_xios
            and self.ds.variables["lat"][0] < self.ds.variables["lat"][-1]
        )
        # Lazy datetime-array cache keyed by time-axis name. Populated on
        # first get_times(axis) call so files with multiple time axes only
        # pay the parse cost for axes the user actually views.
        self._times: dict[str, np.ndarray | None] = {}

    def get_times(self, axis_name: str) -> np.ndarray | None:
        """Return the parsed datetimes for ``axis_name``, or ``None``.

        Returns ``None`` if the coord variable is missing or unparseable.
        Caches per axis — first call parses, subsequent calls hit the cache.
        Returned array contains ``cftime.datetime`` instances (the calendar
        comes from the coord's ``calendar`` attribute; ``num2date`` picks the
        right subclass).
        """
        if axis_name in self._times:
            return self._times[axis_name]
        result = _parse_times(self.ds, axis_name)
        self._times[axis_name] = result
        return result

    def close(self) -> None:
        """Close the underlying Dataset. Safe to call more than once."""
        if self.ds is not None:
            self.ds.close()
            self.ds = None

    def __enter__(self) -> FileContext:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_field(
    path: str | Path,
    name: str,
    time_index: int = 0,
    level_index: int = 0,
    *,
    context: FileContext | None = None,
) -> np.ndarray:
    """Read a field's values. Returns a 1-D array of length ``n_cells``.

    Slices the leading ``time`` dim (if present) at ``time_index``, then the
    ``presnivs`` dim (if present) at ``level_index``, then dispatches to the
    dyn3d / XIOS flatten as appropriate. The icosahedral CF-bounds path
    returns the time-sliced 1-D ``(cell,)`` array directly.

    The leading-dim indices are pushed down to the netCDF4 lazy slicer so
    only the needed slab is read from disk. If ``context`` is provided, its
    held-open Dataset and cached sniffer state are reused, skipping per-call
    file-open overhead — pass a context whenever scrubbing a slider.
    ``level_index`` is ignored for fields without a vertical dim.
    """
    from netCDF4 import Dataset

    if context is not None:
        return _read_field_using(context.ds, name, time_index, level_index,
                                 context.is_dyn3d, context.is_xios,
                                 context.xios_south_first)
    with Dataset(path) as ds:
        is_dyn3d = all(c in ds.variables for c in DYN3D_COORDS)
        is_xios = not is_dyn3d and not _has_cf_bounds(ds) and _is_xios_latlon(ds)
        south = bool(is_xios and ds.variables["lat"][0] < ds.variables["lat"][-1])
        return _read_field_using(ds, name, time_index, level_index,
                                 is_dyn3d, is_xios, south)


def _read_field_using(
    ds, name: str, time_index: int, level_index: int,
    is_dyn3d: bool, is_xios: bool, xios_south_first: bool,
) -> np.ndarray:
    """Slice + flatten a field given an already-open Dataset and sniffer flags."""
    if name not in ds.variables:
        raise KeyError(f"field '{name}' not in {ds.filepath()}")
    var = ds.variables[name]
    dims = var.dimensions

    # Build a slice tuple that picks just (time_index, level_index, :, :) so
    # netCDF4 reads only the needed slab — full var[:] would pull the entire
    # 4-D array.
    idx: list[int | slice] = []
    consumed = 0
    if dims and dims[0] in TIME_DIMS:
        idx.append(time_index)
        consumed += 1
    if len(dims) > consumed and dims[consumed] == LEVEL_DIM:
        idx.append(level_index)
        consumed += 1
    idx.extend([slice(None)] * (len(dims) - consumed))
    # netCDF4 returns a MaskedArray whenever _FillValue is set; np.asarray
    # would silently drop the mask, letting the sentinel (e.g. 9.97e36) leak
    # into the rendered scalars and collapse the colormap range. Replace
    # masked values with NaN so the nan-aware downstream code (np.nanmax in
    # _clim, PyVista's cell_data) skips them.
    raw = var[tuple(idx)]
    if np.ma.isMaskedArray(raw):
        data = np.ma.filled(raw.astype(float, copy=False), np.nan)
    else:
        data = np.asarray(raw)

    if is_dyn3d and data.ndim == 2:
        return _flatten_dyn3d_field(data)
    if is_xios and data.ndim == 2:
        return _flatten_xios_field(data, xios_south_first)
    return data


def iter_field_slabs(ds, name: str):
    """Yield one slab per timestep (or one whole-array slab if static).

    Each slab is a numpy array of every cell and every level for a single
    timestep, with masked values replaced by NaN. Dimensionality is
    preserved — 3-D for level-having fields, 2-D for level-less,
    1-D for static — because callers using this helper (currently only
    :meth:`MainWindow._compute_field_clim`) only need aggregate
    statistics (``nanmin`` / ``nanmax``) and don't care about cell
    ordering or per-level indexing.

    Reading per-timestep instead of per-(time, level) collapses the
    nested loop that drove ``_compute_field_clim``'s wall clock — one
    netCDF slab per t covers every level + every cell in a single I/O
    + a single numpy reduction.
    """
    if name not in ds.variables:
        raise KeyError(f"field '{name}' not in {ds.filepath()}")
    var = ds.variables[name]
    dims = var.dimensions
    has_time = bool(dims) and dims[0] in TIME_DIMS
    n_time = var.shape[0] if has_time else 1
    for t in range(n_time):
        raw = var[t] if has_time else var[...]
        if np.ma.isMaskedArray(raw):
            yield np.ma.filled(raw.astype(float, copy=False), np.nan)
        else:
            yield np.asarray(raw)


def _parse_times(ds, axis_name: str) -> np.ndarray | None:
    """Read + parse a time-axis coord variable into datetimes.

    Used by :class:`FileContext.get_times` and :func:`read_times`. Returns
    ``None`` when the coord variable is absent (e.g. subsetted file) or
    when parsing fails (missing ``units``, unrecognised calendar, etc.) —
    callers fall back to integer-index labels in that case.
    """
    if axis_name not in ds.variables:
        return None
    var = ds.variables[axis_name]
    units = getattr(var, "units", None)
    if not units or "since" not in units:
        return None
    calendar = getattr(var, "calendar", "standard")
    try:
        from netCDF4 import num2date
        return np.asarray(num2date(var[:], units=units, calendar=calendar))
    except Exception:
        return None


def read_times(path: str | Path, axis_name: str) -> np.ndarray | None:
    """Return the parsed datetimes for time axis ``axis_name``, or ``None``.

    Uses CF time conventions (``units: "<unit> since <epoch>"`` +
    ``calendar:``). Falls back to ``None`` when the coord variable is
    missing (subsetted file) or parsing fails; callers can use integer
    indices as a label fallback in that case.

    Library-style API. The GUI uses :meth:`FileContext.get_times` instead,
    which caches the result per file open.
    """
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        return _parse_times(ds, axis_name)


def read_levels(path: str | Path) -> np.ndarray | None:
    """Return the ``presnivs`` axis values (Pa) if present, else ``None``.

    Used by the GUI to label the vertical-level slider. If only the dimension
    exists (no coord variable — possible when a file has been subsetted with
    ``ncks``/``cdo`` and the coord var was dropped), returns ``np.arange(N)``
    so the slider still works, labelled by integer index instead of pressure.
    Returns ``None`` only when there is no vertical dim at all.
    """
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        if LEVEL_DIM in ds.variables:
            return np.asarray(ds.variables[LEVEL_DIM][:], dtype=float)
        if LEVEL_DIM in ds.dimensions:
            return np.arange(len(ds.dimensions[LEVEL_DIM]), dtype=float)
        return None


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
