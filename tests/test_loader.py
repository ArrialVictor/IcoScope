"""NetCDF loader: round-trip a synthetic file and verify the parsed mesh + fields."""
import os
import subprocess
import sys

import numpy as np
import pytest
from netCDF4 import Dataset

from icoscope.loader import (
    describe,
    iter_field_slabs,
    load_grid,
    read_field,
    read_global_attrs,
)


@pytest.fixture(scope="module")
def test_nc(tmp_path_factory) -> str:
    """Run the dev script to produce a synthetic NetCDF in a tmp dir."""
    out = tmp_path_factory.mktemp("nc") / "test_grid.nc"
    repo_root = os.path.dirname(os.path.dirname(__file__))
    subprocess.run(
        [sys.executable, os.path.join(repo_root, "tools", "make_test_nc.py"),
         "-o", str(out)],
        check=True, capture_output=True,
    )
    return str(out)


def test_load_grid_returns_mesh_and_fields(test_nc):
    verts, cells, centers, fields = load_grid(test_nc)
    # mesh
    assert len(cells) == 4002
    assert sum(1 for c in cells if len(c) == 5) == 12
    assert sum(1 for c in cells if len(c) == 6) == 3990
    # centers on the unit sphere
    assert np.allclose(np.linalg.norm(centers, axis=1), 1.0, atol=1e-5)
    # field metadata
    assert {"tas", "tas_anomaly", "precip", "tas_t", "vort_t"} <= set(fields)
    assert fields["tas"]["units"] == "K"
    assert fields["tas"]["time_varying"] is False
    assert fields["tas_t"]["time_varying"] is True


def test_read_field_returns_per_cell_array(test_nc):
    tas = read_field(test_nc, "tas")
    assert tas.shape == (4002,)
    # temperature in a plausible range
    assert 200 < tas.min() < tas.max() < 350


def test_read_field_time_slice(test_nc):
    a = read_field(test_nc, "tas_t", time_index=0)
    b = read_field(test_nc, "tas_t", time_index=3)   # quarter year — peak seasonal contrast
    assert a.shape == (4002,) and b.shape == (4002,)
    # different snapshots of the same field should not be identical
    assert not np.allclose(a, b)


def test_load_grid_raises_keyerror_when_lon_missing(tmp_path):
    """A valid NetCDF without lon/longitude should raise a KeyError naming the
    candidate variables it tried."""
    path = tmp_path / "no_lon.nc"
    with Dataset(path, "w", format="NETCDF4") as ds:
        # Create a valid file but without any of the lon-candidate variables.
        ds.createDimension("cell", 4)
        ds.createDimension("nvertex", 6)
        ds.createVariable("lat", "f8", ("cell",))[:] = [0, 30, -30, 60]
        ds.createVariable("bounds_lon", "f8", ("cell", "nvertex"))[:] = np.zeros((4, 6))
        ds.createVariable("bounds_lat", "f8", ("cell", "nvertex"))[:] = np.zeros((4, 6))
    with pytest.raises(KeyError, match="lon|longitude"):
        load_grid(str(path))


def test_read_global_attrs_returns_str_dict(tmp_path):
    """Global attributes are returned as ``str`` regardless of their original type."""
    path = tmp_path / "with_attrs.nc"
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.title = "synthetic test"
        ds.version = 1                  # int
        ds.pi = 3.14                    # float
        ds.history = "created in a test"
    attrs = read_global_attrs(str(path))
    assert attrs["title"] == "synthetic test"
    assert attrs["history"] == "created in a test"
    # Non-strings are coerced via str().
    assert attrs["version"] == "1"
    assert attrs["pi"] == "3.14"
    # All values are strings.
    for v in attrs.values():
        assert isinstance(v, str)


def test_iter_field_slabs_static_yields_single_slab(test_nc):
    """A static (no-time) field yields exactly one whole-array slab."""
    with Dataset(test_nc) as ds:
        slabs = list(iter_field_slabs(ds, "tas"))
    assert len(slabs) == 1
    assert slabs[0].shape == (4002,)


def test_iter_field_slabs_time_varying_yields_per_timestep(test_nc):
    """A time-varying field yields one slab per timestep, each covering all cells."""
    with Dataset(test_nc) as ds:
        n_time = ds.dimensions["time"].size
        slabs = list(iter_field_slabs(ds, "tas_t"))
    assert len(slabs) == n_time
    for slab in slabs:
        # tas_t is (time, cell) → each slab is 1-D over cells.
        assert slab.shape == (4002,)


def test_iter_field_slabs_min_max_matches_per_step_reads(test_nc):
    """Aggregate min/max over slabs equals the same over per-step read_field reads.

    Pins down the contract that ``_compute_field_clim`` relies on: the
    fast path (one read per timestep) and the slow path (one read per
    (time, level)) must produce the same global range. Without this
    guarantee the clim-vectorization could silently shift the colour
    range on real files.
    """
    with Dataset(test_nc) as ds:
        n_time = ds.dimensions["time"].size
        fast_lo, fast_hi = np.inf, -np.inf
        for slab in iter_field_slabs(ds, "tas_t"):
            fast_lo = min(fast_lo, float(np.nanmin(slab)))
            fast_hi = max(fast_hi, float(np.nanmax(slab)))

    slow_lo, slow_hi = np.inf, -np.inf
    for t in range(n_time):
        arr = read_field(test_nc, "tas_t", time_index=t)
        slow_lo = min(slow_lo, float(np.nanmin(arr)))
        slow_hi = max(slow_hi, float(np.nanmax(arr)))

    assert fast_lo == slow_lo
    assert fast_hi == slow_hi


def test_iter_field_slabs_raises_keyerror_on_unknown_field(test_nc):
    with (
        Dataset(test_nc) as ds,
        pytest.raises(KeyError, match="no_such_field"),
    ):
        list(iter_field_slabs(ds, "no_such_field"))


def test_describe_prints_dimensions_and_variables(test_nc, capsys):
    """``describe`` should print every dim and var so users can adapt the loader."""
    describe(test_nc)
    out = capsys.readouterr().out
    assert "Dimensions:" in out
    assert "Variables:" in out
    # Some specific names from the synthetic fixture must appear.
    assert "lon" in out
    assert "tas" in out
