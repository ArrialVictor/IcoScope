"""XIOS regular lat-lon loader: 1-D lon/lat centers, no bounds."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from icoscope.loader import (
    _flatten_xios_field,
    _is_xios_latlon,
    load_grid,
    read_field,
)
from icoscope.lonlat import build_mesh_from_centers


def _make_xios_nc(
    path: Path,
    *,
    nlon: int = 8,
    nlat: int = 6,
    south_first: bool = True,
    with_time: bool = False,
    with_3d: bool = False,
) -> None:
    """Write a minimal XIOS-style lat-lon NetCDF with a 2-D scalar field.

    The field's value pattern is ``i + 100 * j_north_first`` (where
    ``j_north_first`` indexes from the north pole) so the flatten ordering can
    be verified unambiguously. Polar rows are overridden to constants.
    """
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    lat = np.linspace(-90.0, 90.0, nlat) if south_first else np.linspace(90.0, -90.0, nlat)

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("lon", nlon)
        ds.createDimension("lat", nlat)
        v = ds.createVariable("lon", "f8", ("lon",))
        v[:] = lon
        v.axis = "X"
        v.units = "degrees_east"
        v = ds.createVariable("lat", "f8", ("lat",))
        v[:] = lat
        v.axis = "Y"
        v.units = "degrees_north"

        # 2-D field, north-first orientation. Value pattern lets the flatten
        # be verified row by row.
        i_idx, j_north = np.meshgrid(np.arange(nlon), np.arange(nlat))
        field_n = (i_idx + 100.0 * j_north).astype("f8")
        field_n[0, :] = 999.0   # north pole row
        field_n[-1, :] = -999.0  # south pole row
        field_storage = field_n[::-1, :] if south_first else field_n

        v = ds.createVariable("slp", "f8", ("lat", "lon"))
        v[:] = field_storage
        v.units = "Pa"
        v.long_name = "Sea Level Pressure"

        if with_time:
            ds.createDimension("time_counter", 3)
            ds.createVariable("time_counter", "f8", ("time_counter",))[:] = [0, 1, 2]
            t_field = np.stack([field_storage + 1000.0 * t for t in range(3)])
            v = ds.createVariable("temp", "f8", ("time_counter", "lat", "lon"))
            v[:] = t_field
            v.units = "K"
            v.long_name = "Air temperature"

        if with_3d:
            ds.createDimension("presnivs", 4)
            ds.createVariable("presnivs", "f8", ("presnivs",))[:] = [
                100000.0, 50000.0, 10000.0, 1000.0,
            ]
            v = ds.createVariable("temp3d", "f8", ("presnivs", "lat", "lon"))
            v[:] = np.zeros((4, nlat, nlon))


def test_build_mesh_from_centers_shapes():
    lon = np.linspace(0.0, 360.0, 8, endpoint=False)
    lat = np.linspace(-90.0, 90.0, 6)
    verts, cells, centers, south_first = build_mesh_from_centers(lon, lat)
    assert south_first is True
    # jjm = nlat - 1 = 5 bands, iim = 8 cells per band → 40 cells
    assert len(cells) == 5 * 8
    assert centers.shape == (40, 3)
    # Cells in polar bands are triangles (3 verts), interior quads (4 verts)
    cells_by_size = {len(c) for c in cells}
    assert cells_by_size == {3, 4}


def test_build_mesh_from_centers_north_first():
    lon = np.linspace(0.0, 360.0, 8, endpoint=False)
    lat = np.linspace(90.0, -90.0, 6)
    verts, cells, centers, south_first = build_mesh_from_centers(lon, lat)
    assert south_first is False
    assert len(cells) == 5 * 8


def test_build_mesh_from_centers_requires_poles():
    lon = np.linspace(0.0, 360.0, 8, endpoint=False)
    lat = np.linspace(-89.5, 89.5, 6)
    with pytest.raises(ValueError, match="±90"):
        build_mesh_from_centers(lon, lat)


def test_build_mesh_from_centers_centers_on_unit_sphere():
    lon = np.linspace(0.0, 360.0, 8, endpoint=False)
    lat = np.linspace(-90.0, 90.0, 6)
    _, _, centers, _ = build_mesh_from_centers(lon, lat)
    norms = np.linalg.norm(centers, axis=1)
    assert np.allclose(norms, 1.0)


def test_flatten_xios_field_north_first_pole_mapping():
    # 4 rows, 3 cols, north-first
    arr = np.array([
        [10, 11, 12],   # north pole
        [20, 21, 22],
        [30, 31, 32],   # this row should be SKIPPED (jjm-1 = nlat-2)
        [40, 41, 42],   # south pole — fills mesh row jjm-1
    ], dtype=float)
    out = _flatten_xios_field(arr, south_first=False)
    # jjm = 3 rows
    assert out.shape == (3 * 3,)
    expected = np.array([
        10, 11, 12,     # mesh row 0 = north pole
        20, 21, 22,     # mesh row 1 = interior
        40, 41, 42,     # mesh row 2 = south pole (skipped field row 30,31,32)
    ], dtype=float)
    np.testing.assert_array_equal(out, expected)


def test_flatten_xios_field_south_first_pole_mapping():
    # Same data but south-first storage: pole rows swap
    arr = np.array([
        [40, 41, 42],   # south pole (stored first)
        [30, 31, 32],   # skipped after flip
        [20, 21, 22],
        [10, 11, 12],   # north pole (stored last)
    ], dtype=float)
    out = _flatten_xios_field(arr, south_first=True)
    expected = np.array([
        10, 11, 12,
        20, 21, 22,
        40, 41, 42,
    ], dtype=float)
    np.testing.assert_array_equal(out, expected)


def test_is_xios_latlon_sniffer():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path)
        with Dataset(path) as ds:
            assert _is_xios_latlon(ds) is True


def test_load_grid_xios_path():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path, with_time=True)
        verts, cells, centers, fields = load_grid(path)
        assert len(cells) > 0
        assert "slp" in fields
        assert "temp" in fields
        assert fields["slp"]["time_varying"] is False
        assert fields["temp"]["time_varying"] is True
        assert fields["slp"]["kind"] == "xios"


def test_load_grid_xios_skips_3d_fields():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path, with_3d=True)
        _, _, _, fields = load_grid(path)
        assert "temp3d" not in fields
        assert "slp" in fields


def test_read_field_xios_static():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path, nlon=8, nlat=6, south_first=True)
        values = read_field(path, "slp")
        # 5 bands × 8 longitudes = 40 cells
        assert values.shape == (40,)
        # First 8 values (mesh row 0 = north band) should all be 999 (north pole)
        assert np.all(values[:8] == 999.0)
        # Last 8 (mesh row jjm-1 = south band) should all be -999
        assert np.all(values[-8:] == -999.0)


def test_read_field_xios_time_slice():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path, with_time=True)
        v0 = read_field(path, "temp", time_index=0)
        v2 = read_field(path, "temp", time_index=2)
        # Each time step adds 1000 to the field — north pole row at t=0 is 999,
        # at t=2 is 2999. Last band (south pole) is -999 → 1001.
        assert np.all(v0[:8] == 999.0)
        assert np.all(v2[:8] == 2999.0)
        assert np.all(v0[-8:] == -999.0)
        assert np.all(v2[-8:] == 1001.0)
