"""NetCDF loader: round-trip a synthetic file and verify the parsed mesh + fields."""
import os
import subprocess
import sys

import numpy as np
import pytest

from icoscope.loader import load_grid, read_field


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


def test_load_grid_raises_on_missing_variables(tmp_path):
    """A non-NetCDF or schema-incompatible file should raise."""
    bad = tmp_path / "bad.nc"
    bad.write_bytes(b"not a netcdf")
    # netCDF4 may raise OSError / KeyError / ValueError depending on what's wrong;
    # the only contract is that load_grid does not silently succeed.
    with pytest.raises((OSError, KeyError, ValueError)):
        load_grid(str(bad))
