"""Dyn3d field reading: 2-D (rlatu, rlonv) → cell-flat layout."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from icoscope.loader import _flatten_dyn3d_field, load_grid, read_field
from icoscope.lonlat import latlon_mesh


def _make_dyn3d_nc(path: Path, iim: int, jjm: int, with_time: bool = False) -> None:
    """Write a minimal dyn3d NetCDF: coord arrays + a scalar field.

    The field's value pattern (i + 100*j for the 2-D case, +1000*t for time)
    lets tests verify the cell-flatten ordering without ambiguity.
    """
    iip1 = iim + 1
    jjp1 = jjm + 1
    rlonu = np.linspace(-np.pi, np.pi, iip1, endpoint=True)
    rlonv = rlonu + (np.pi / iim)
    rlatu = np.linspace(np.pi / 2, -np.pi / 2, jjp1)
    rlatv = (rlatu[:-1] + rlatu[1:]) / 2  # jjm entries

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("rlonu", iip1)
        ds.createDimension("rlonv", iip1)
        ds.createDimension("rlatu", jjp1)
        ds.createDimension("rlatv", jjm)
        ds.createVariable("rlonu", "f8", ("rlonu",))[:] = rlonu
        ds.createVariable("rlonv", "f8", ("rlonv",))[:] = rlonv
        ds.createVariable("rlatu", "f8", ("rlatu",))[:] = rlatu
        ds.createVariable("rlatv", "f8", ("rlatv",))[:] = rlatv

        # 2-D scalar field (rlatu, rlonv) — value pattern: i + 100*j
        i_idx, j_idx = np.meshgrid(np.arange(iip1), np.arange(jjp1))
        field2d = (i_idx + 100 * j_idx).astype("f8")
        # Polar rows: all values along that row must be identical (LMDZ
        # convention — they all sit at the same pole). Override to a constant.
        field2d[0, :] = 99.0
        field2d[-1, :] = -99.0
        v = ds.createVariable("temp", "f8", ("rlatu", "rlonv"))
        v[:] = field2d
        v.units = "K"
        v.long_name = "temperature"

        if with_time:
            ds.createDimension("time", 3)
            ds.createVariable("time", "f8", ("time",))[:] = [0, 1, 2]
            t_field = np.stack([field2d, field2d + 1000.0, field2d + 2000.0])
            tv = ds.createVariable("pres", "f8", ("time", "rlatu", "rlonv"))
            tv[:] = t_field
            tv.units = "Pa"
            tv.long_name = "pressure"


def test_load_dyn3d_exposes_fields():
    iim, jjm = 8, 6
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        path = Path(f.name)
    try:
        _make_dyn3d_nc(path, iim=iim, jjm=jjm, with_time=True)
        verts, cells, centers, fields = load_grid(path)
        assert len(cells) == iim * jjm
        assert set(fields) == {"temp", "pres"}
        assert fields["temp"]["units"] == "K"
        assert fields["temp"]["long_name"] == "temperature"
        assert fields["temp"]["kind"] == "dyn3d"
        assert fields["temp"]["time_varying"] is False
        assert fields["pres"]["time_varying"] is True
        assert fields["pres"]["shape"] == (3, jjm + 1, iim + 1)
    finally:
        path.unlink(missing_ok=True)


def test_flatten_dyn3d_field_drops_periodic_duplicate_and_skipped_row():
    """The trailing iip1-th column wraps and is dropped.

    Also: the mesh's south-polar band maps to the south-pole field row, not
    the southernmost interior row, so field row jjm-1 is skipped entirely.
    """
    jjp1, iip1 = 4, 5            # jjm=3, iim=4
    i_idx, j_idx = np.meshgrid(np.arange(iip1), np.arange(jjp1))
    arr = (i_idx + 100 * j_idx).astype(float)
    flat = _flatten_dyn3d_field(arr)
    assert flat.shape == (3 * 4,)
    # cell at (j=0, i=0) → flat[0]; (j=1, i=2) → flat[1*4+2]=flat[6]
    assert flat[0] == 0
    assert flat[6] == 102                          # mesh row 1 ← field row 1
    # mesh row 2 (south polar) ← field row jjp1-1 = 3 (south pole), NOT field row 2
    assert flat[3 * 4 - 1] == 3 + 100 * 3          # i=3, j=3
    # The dropped iip1-th column had values 4, 104, 204, 304 — not present.
    assert 4 not in flat
    assert 304 not in flat
    # Skipped field row (j=2): values 200, 201, 202, 203 — not present.
    assert 200 not in flat
    assert 203 not in flat


def test_read_field_returns_cell_flat_ordering():
    iim, jjm = 8, 6
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        path = Path(f.name)
    try:
        _make_dyn3d_nc(path, iim=iim, jjm=jjm, with_time=False)
        values = read_field(path, "temp")
        assert values.shape == (iim * jjm,)
        # Polar rows of the source had all values = ±99. The mesh's first iim
        # cells are the north polar band; last iim are the south polar band.
        np.testing.assert_array_equal(values[:iim], 99.0)
        np.testing.assert_array_equal(values[-iim:], -99.0)
        # Interior cell (j=2, i=3) → flat index 2*iim + 3 = 19; value = 3 + 100*2.
        assert values[2 * iim + 3] == 3 + 100 * 2
    finally:
        path.unlink(missing_ok=True)


def test_read_field_time_index():
    iim, jjm = 6, 4
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        path = Path(f.name)
    try:
        _make_dyn3d_nc(path, iim=iim, jjm=jjm, with_time=True)
        v0 = read_field(path, "pres", time_index=0)
        v1 = read_field(path, "pres", time_index=1)
        v2 = read_field(path, "pres", time_index=2)
        # Each time step adds 1000 to every interior cell.
        # Interior cell (j=1, i=2) — flat 1*iim+2 = 8 — has value 2 + 100*1 = 102 at t=0.
        assert v0[8] == 102
        assert v1[8] == 1102
        assert v2[8] == 2102
    finally:
        path.unlink(missing_ok=True)


def test_synthetic_lonlat_mesh_matches_field_count():
    """End-to-end sanity: the synthetic generator and the loader's cell count
    agree, so a field generated against the synthetic mesh shape matches the
    loader-rebuilt mesh shape."""
    iim, jjm = 12, 8
    verts_s, cells_s, _ = latlon_mesh(iim=iim, jjm=jjm)
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        path = Path(f.name)
    try:
        _make_dyn3d_nc(path, iim=iim, jjm=jjm)
        verts_l, cells_l, _, _ = load_grid(path)
        assert len(cells_s) == len(cells_l)
        values = read_field(path, "temp")
        assert values.shape == (len(cells_l),)
    finally:
        path.unlink(missing_ok=True)


def test_flatten_field_rejects_3d():
    with pytest.raises(ValueError, match="expected 2-D"):
        _flatten_dyn3d_field(np.zeros((2, 3, 4)))


def _make_dyn3d_nc_with_extras(path: Path, iim: int, jjm: int) -> None:
    """Like _make_dyn3d_nc but also adds variables that should be SKIPPED:
    a vertical-profile field on (presnivs, rlatu, rlonv), a 1-D timeseries,
    a scalar attribute, and one of the coord vars (which must also be skipped).
    """
    iip1, jjp1 = iim + 1, jjm + 1
    rlonu = np.linspace(-np.pi, np.pi, iip1)
    rlonv = rlonu + (np.pi / iim)
    rlatu = np.linspace(np.pi / 2, -np.pi / 2, jjp1)
    rlatv = (rlatu[:-1] + rlatu[1:]) / 2

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("rlonu", iip1)
        ds.createDimension("rlonv", iip1)
        ds.createDimension("rlatu", jjp1)
        ds.createDimension("rlatv", jjm)
        ds.createDimension("time", 2)
        ds.createDimension("presnivs", 5)
        ds.createVariable("rlonu", "f8", ("rlonu",))[:] = rlonu
        ds.createVariable("rlonv", "f8", ("rlonv",))[:] = rlonv
        ds.createVariable("rlatu", "f8", ("rlatu",))[:] = rlatu
        ds.createVariable("rlatv", "f8", ("rlatv",))[:] = rlatv
        ds.createVariable("time", "f8", ("time",))[:] = [0, 1]
        ds.createVariable("presnivs", "f8", ("presnivs",))[:] = np.arange(5)

        # KEEP: 2-D field on (rlatu, rlonv)
        ds.createVariable("tas", "f8", ("rlatu", "rlonv"))[:] = np.zeros((jjp1, iip1))
        # KEEP: time-varying 3-D field on (time, rlatu, rlonv)
        ds.createVariable("ps", "f8", ("time", "rlatu", "rlonv"))[:] = np.zeros((2, jjp1, iip1))
        # SKIP: vertical profile on (presnivs, rlatu, rlonv)
        ds.createVariable("u", "f8", ("presnivs", "rlatu", "rlonv"))[:] = np.zeros((5, jjp1, iip1))
        # SKIP: 4-D with both time and presnivs
        ds.createVariable("temp4d", "f8", ("time", "presnivs", "rlatu", "rlonv"))[:] = (
            np.zeros((2, 5, jjp1, iip1))
        )
        # SKIP: 1-D timeseries (no rlatu/rlonv)
        ds.createVariable("global_mean", "f8", ("time",))[:] = [273.0, 274.0]
        # SKIP: a control array (matches the descriptive-vars filter)
        ds.createDimension("ncontrol", 100)
        ds.createVariable("controle", "f8", ("ncontrol",))[:] = np.zeros(100)


def test_load_dyn3d_surfaces_vertical_and_skips_descriptive_vars():
    """Vertical-profile fields are now surfaced (with n_levels); descriptive vars stay filtered."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        path = Path(f.name)
    try:
        _make_dyn3d_nc_with_extras(path, iim=6, jjm=4)
        _, cells, _, fields = load_grid(path)
        assert len(cells) == 6 * 4
        # Both flat and vertical fields surface; descriptive ones stay out.
        assert set(fields) == {"tas", "ps", "u", "temp4d"}
        assert fields["tas"]["time_varying"] is False
        assert fields["tas"]["n_levels"] == 0
        assert fields["ps"]["time_varying"] is True
        assert fields["ps"]["n_levels"] == 0
        assert fields["u"]["time_varying"] is False
        assert fields["u"]["n_levels"] == 5
        assert fields["temp4d"]["time_varying"] is True
        assert fields["temp4d"]["n_levels"] == 5
        for skipped in ("global_mean", "controle", "presnivs",
                        "time", "rlonu", "rlatu", "rlonv", "rlatv"):
            assert skipped not in fields, f"{skipped!r} should have been skipped"
    finally:
        path.unlink(missing_ok=True)


def test_load_dyn3d_missing_coord_falls_through_to_icosa():
    """If any of the four dyn3d coords is missing, the file is NOT recognized."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        path = Path(f.name)
    try:
        with Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("rlonu", 4)
            ds.createDimension("rlatu", 3)
            ds.createDimension("rlonv", 4)
            # Missing rlatv → not a dyn3d sniffer hit
            ds.createVariable("rlonu", "f8", ("rlonu",))[:] = np.zeros(4)
            ds.createVariable("rlatu", "f8", ("rlatu",))[:] = np.zeros(3)
            ds.createVariable("rlonv", "f8", ("rlonv",))[:] = np.zeros(4)
        # Should now fail: dyn3d sniffer misses, no CF bounds, no XIOS coords either
        with pytest.raises((KeyError, ValueError), match="lon|latitude|grid layout"):
            load_grid(path)
    finally:
        path.unlink(missing_ok=True)
