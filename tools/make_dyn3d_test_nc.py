"""Generate a synthetic LMDZ dyn3d-style NetCDF — development utility.

Writes a file with the four rlonu/rlonv/rlatu/rlatv coord arrays and a
couple of physically-plausible scalar fields on the (rlatu, rlonv) grid,
plus one time-varying field. Useful for sanity-checking the dyn3d loader
path (cell reconstruction + field reading) without needing a real LMDZ
output file.

Usage (from the repo root, with icoscope installed in editable mode):
    python tools/make_dyn3d_test_nc.py                # writes test_dyn3d.nc
    python tools/make_dyn3d_test_nc.py -o foo.nc      # custom output path
    python tools/make_dyn3d_test_nc.py --iim 144 --jjm 143
"""
import argparse
import os

import numpy as np
from netCDF4 import Dataset


def main():
    ap = argparse.ArgumentParser(
        prog="icoscope-mktest-dyn3d",
        description="Generate a synthetic LMDZ dyn3d-style test NetCDF.",
    )
    ap.add_argument("-o", "--output", default="test_dyn3d.nc", metavar="PATH",
                    help="output file path (default: ./test_dyn3d.nc)")
    ap.add_argument("--iim", type=int, default=96,
                    help="number of distinct longitudes (default 96)")
    ap.add_argument("--jjm", type=int, default=95,
                    help="number of latitude bands (default 95)")
    ap.add_argument("--n-time", type=int, default=12, dest="n_time",
                    help="number of time steps for the time-varying field (default 12)")
    args = ap.parse_args()

    iim, jjm = args.iim, args.jjm
    iip1, jjp1 = iim + 1, jjm + 1
    outpath = os.path.abspath(args.output)

    print(f"Building dyn3d coord arrays (iim={iim}, jjm={jjm})…")
    rlonu = np.linspace(-np.pi, np.pi, iip1, endpoint=True)
    # Cell centers in longitude: shifted by half a step.
    rlonv = rlonu + (np.pi / iim)
    rlatu = np.linspace(np.pi / 2, -np.pi / 2, jjp1)
    rlatv = (rlatu[:-1] + rlatu[1:]) / 2

    print("Generating synthetic fields…")
    lat2d = rlatu[:, None] * np.ones((1, iip1))
    lon2d = np.ones((jjp1, 1)) * rlonv[None, :]

    # Surface-temperature mock (Kelvin): latitudinal gradient + a faint
    # zonal wave + a Saharan hot spot.
    tas = 275.0 + 30.0 * np.cos(lat2d) + 5.0 * np.cos(2 * lon2d)
    sahara = 14.0 * np.exp(-(((lat2d - np.radians(23)) ** 2
                              + (lon2d - np.radians(10)) ** 2) / (2 * (np.radians(14)) ** 2)))
    tas = tas + sahara
    # Polar rows: enforce identical values across the row (LMDZ convention).
    tas[0, :] = tas[0, 0]
    tas[-1, :] = tas[-1, 0]

    # Surface pressure mock (Pa) — gentle anti-correlation with temperature.
    ps = 101325.0 - 100.0 * (tas - 280.0)
    ps[0, :] = ps[0, 0]
    ps[-1, :] = ps[-1, 0]

    # Time-varying mock: zonal wind (m/s) with a wave that rotates over time.
    n_time = args.n_time
    times = np.arange(n_time, dtype=float)
    u = np.empty((n_time, jjp1, iip1))
    for t in range(n_time):
        phase = 2 * np.pi * t / n_time
        u[t] = 20.0 * np.sin(lat2d) * np.cos(2 * lon2d + phase)
        u[t, 0, :] = u[t, 0, 0]
        u[t, -1, :] = u[t, -1, 0]

    print(f"Writing {outpath}…")
    if os.path.exists(outpath):
        os.remove(outpath)
    with Dataset(outpath, "w", format="NETCDF4") as ds:
        ds.title = "Synthetic LMDZ dyn3d test grid"
        ds.source = "tools/make_dyn3d_test_nc.py"

        ds.createDimension("rlonu", iip1)
        ds.createDimension("rlonv", iip1)
        ds.createDimension("rlatu", jjp1)
        ds.createDimension("rlatv", jjm)
        ds.createDimension("time", n_time)

        ds.createVariable("rlonu", "f8", ("rlonu",))[:] = rlonu
        ds.createVariable("rlonv", "f8", ("rlonv",))[:] = rlonv
        ds.createVariable("rlatu", "f8", ("rlatu",))[:] = rlatu
        ds.createVariable("rlatv", "f8", ("rlatv",))[:] = rlatv
        ds.createVariable("time", "f8", ("time",))[:] = times

        tas_v = ds.createVariable("tas", "f8", ("rlatu", "rlonv"))
        tas_v[:] = tas
        tas_v.units = "K"
        tas_v.long_name = "near-surface air temperature"

        ps_v = ds.createVariable("ps", "f8", ("rlatu", "rlonv"))
        ps_v[:] = ps
        ps_v.units = "Pa"
        ps_v.long_name = "surface pressure"

        u_v = ds.createVariable("u_surf", "f8", ("time", "rlatu", "rlonv"))
        u_v[:] = u
        u_v.units = "m s-1"
        u_v.long_name = "near-surface zonal wind"

    print(f"  {iim*jjm} cells, fields: tas, ps, u_surf ({n_time} time steps)")
    print(f"\nTry: icoscope --file {outpath}")


if __name__ == "__main__":
    main()
