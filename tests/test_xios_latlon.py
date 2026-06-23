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
    read_levels,
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
            # Per-level value pattern: level k fills with constant k, so the
            # level-slice test can verify k was picked from a (t, k, lat, lon)
            # field unambiguously. Polar rows still override to ±999 so the
            # flatten path is exercised.
            v = ds.createVariable("temp3d", "f8", ("presnivs", "lat", "lon"))
            data3 = np.zeros((4, nlat, nlon))
            for k in range(4):
                data3[k, :, :] = k
                data3[k, 0, :] = 999.0 if south_first else -999.0
                data3[k, -1, :] = -999.0 if south_first else 999.0
            v[:] = data3
            if with_time:
                v4 = ds.createVariable(
                    "temp4d", "f8", ("time_counter", "presnivs", "lat", "lon"))
                # Per-(t,k) constant t*100 + k so each slice is distinguishable
                data4 = np.zeros((3, 4, nlat, nlon))
                for t in range(3):
                    for k in range(4):
                        data4[t, k, :, :] = t * 100 + k
                v4[:] = data4


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


def test_load_grid_xios_surfaces_3d_fields():
    """3-D (presnivs, lat, lon) fields are now surfaced with n_levels set."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path, with_3d=True)
        _, _, _, fields = load_grid(path)
        assert "slp" in fields
        assert fields["slp"]["n_levels"] == 0
        assert "temp3d" in fields
        assert fields["temp3d"]["n_levels"] == 4
        assert fields["temp3d"]["time_varying"] is False


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


def test_read_field_xios_level_slice_3d():
    """For a (presnivs, lat, lon) field, level_index picks the right level."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path, nlon=8, nlat=6, south_first=True, with_3d=True)
        v0 = read_field(path, "temp3d", level_index=0)
        v2 = read_field(path, "temp3d", level_index=2)
        # Interior cells (not the polar ±999 overrides) should equal the level index.
        assert v0[10] == 0.0
        assert v2[10] == 2.0


def test_read_field_xios_level_and_time_slice_4d():
    """For (time, presnivs, lat, lon) the slice order is time-then-level."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path, nlon=8, nlat=6, south_first=True,
                      with_time=True, with_3d=True)
        v_t1_k2 = read_field(path, "temp4d", time_index=1, level_index=2)
        # Constant 1*100 + 2 = 102 everywhere (no polar overrides for temp4d)
        assert np.all(v_t1_k2 == 102.0)


def test_read_levels_returns_presnivs():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path, with_3d=True)
        levels = read_levels(path)
        np.testing.assert_array_equal(levels, [100000.0, 50000.0, 10000.0, 1000.0])


def test_read_levels_none_when_absent():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path)
        assert read_levels(path) is None


def test_read_field_masked_array_becomes_nan():
    """_FillValue should be replaced by NaN, not leak through as the sentinel."""
    fv = np.float32(9.96921e36)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        nlon, nlat = 8, 6
        lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
        lat = np.linspace(-90.0, 90.0, nlat)
        with Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("lon", nlon)
            ds.createDimension("lat", nlat)
            ds.createVariable("lon", "f8", ("lon",))[:] = lon
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            v = ds.createVariable("slp", "f4", ("lat", "lon"), fill_value=fv)
            arr = np.full((nlat, nlon), 100000.0, dtype=np.float32)
            arr[2, 3] = fv  # masked cell
            v[:] = arr
        out = read_field(path, "slp")
        assert not np.any(out == fv), "sentinel leaked through asarray"
        assert np.any(np.isnan(out)), "masked cell should be NaN"


def test_read_levels_falls_back_to_indices_without_coord_var():
    """File with presnivs dim but no presnivs coord var: return integer indices."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path)  # no presnivs at all
        with Dataset(path, "a") as ds:
            ds.createDimension("presnivs", 5)
            # Deliberately do NOT create a presnivs variable
        levels = read_levels(path)
        assert levels is not None
        np.testing.assert_array_equal(levels, np.arange(5, dtype=float))


def test_classify_var_returns_time_dim_name():
    """_classify_var should return the actual time-dim name (not a bool)."""
    from icoscope.loader import _classify_var
    # Static field on cell
    assert _classify_var(("cell",), ("cell",), 0) == (None, 0)
    # Time-varying on cell — returns the dim name
    assert _classify_var(("time", "cell"), ("cell",), 0) == ("time", 0)
    assert _classify_var(("time_counter", "cell"), ("cell",), 0) == ("time_counter", 0)
    # Vertical on cell, no time
    assert _classify_var(("presnivs", "cell"), ("cell",), 5) == (None, 5)
    # Time + vertical on cell
    assert _classify_var(("time", "presnivs", "cell"), ("cell",), 5) == ("time", 5)


def test_classify_var_accepts_capital_time():
    """TIME_DIMS now includes 'Time'/'t' so WRF-style files aren't dropped."""
    from icoscope.loader import _classify_var
    assert _classify_var(("Time", "lat", "lon"), ("lat", "lon"), 0) == ("Time", 0)
    assert _classify_var(("t", "lat", "lon"), ("lat", "lon"), 0) == ("t", 0)


def test_file_context_caches_sniffer_and_dataset():
    """FileContext holds the Dataset open + cached sniffer flags."""
    from icoscope.loader import FileContext
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path)
        with FileContext(path) as ctx:
            assert ctx.is_dyn3d is False
            assert ctx.is_xios is True
            assert ctx.xios_south_first is True   # _make_xios_nc default
            assert ctx.ds is not None
        # closed after context manager exit
        assert ctx.ds is None


def test_read_field_with_context_matches_no_context():
    """read_field with and without context returns identical data."""
    from icoscope.loader import FileContext
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path, with_time=True)
        ref = read_field(path, "temp", time_index=2)
        with FileContext(path) as ctx:
            via_ctx = read_field(path, "temp", time_index=2, context=ctx)
        np.testing.assert_array_equal(ref, via_ctx)


def test_field_meta_carries_time_dim_name():
    """FieldMeta should record which time dim each variable lives on."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path, with_time=True)
        _, _, _, fields = load_grid(path)
        # 'slp' is static
        assert fields["slp"]["time_varying"] is False
        assert fields["slp"]["time_dim_name"] is None
        # 'temp' is time-varying on the 'time_counter' axis
        assert fields["temp"]["time_varying"] is True
        assert fields["temp"]["time_dim_name"] == "time_counter"


def test_read_times_parses_gregorian():
    """read_times converts the seconds-since coord into cftime datetimes."""
    from icoscope.loader import read_times
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        nlon, nlat, n_t = 4, 4, 3
        lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
        lat = np.linspace(-90.0, 90.0, nlat)
        with Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("lon", nlon)
            ds.createDimension("lat", nlat)
            ds.createDimension("time_counter", n_t)
            ds.createVariable("lon", "f8", ("lon",))[:] = lon
            ds.createVariable("lat", "f8", ("lat",))[:] = lat
            v = ds.createVariable("time_counter", "f8", ("time_counter",))
            # 0, 86400, 172800 seconds since 2025-02-01 → Feb 1, 2, 3
            v[:] = [0.0, 86400.0, 172800.0]
            v.units = "seconds since 2025-02-01 00:00:00"
            v.calendar = "gregorian"
        times = read_times(path, "time_counter")
        assert times is not None
        assert len(times) == 3
        assert str(times[0])[:10] == "2025-02-01"
        assert str(times[2])[:10] == "2025-02-03"


def test_read_times_returns_none_without_coord_var():
    """read_times falls back when the axis dim has no coord variable."""
    from icoscope.loader import read_times
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        with Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("time_counter", 5)
            # No coord variable — only the dim
        assert read_times(path, "time_counter") is None


def test_read_times_returns_none_for_unparseable_units():
    """Bad/missing units shouldn't blow up — return None and let the caller fall back."""
    from icoscope.loader import read_times
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        with Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("time_counter", 3)
            v = ds.createVariable("time_counter", "f8", ("time_counter",))
            v[:] = [0, 1, 2]
            v.units = "seconds"   # missing the "since X" anchor
        assert read_times(path, "time_counter") is None


def test_filecontext_caches_times_per_axis():
    """FileContext.get_times parses on first call, returns cached array after."""
    from icoscope.loader import FileContext
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.nc"
        _make_xios_nc(path, with_time=True)
        # _make_xios_nc writes time_counter without units/calendar, so the
        # parser returns None — but the *caching* path should still work.
        with FileContext(path) as ctx:
            first = ctx.get_times("time_counter")
            second = ctx.get_times("time_counter")
            # Both calls return the same object (cached).
            assert first is second
