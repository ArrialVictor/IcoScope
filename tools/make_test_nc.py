"""Generate a synthetic ICOLMDZ-like NetCDF — development utility.

Used during development to test the loader and time-slider paths without
needing a real ICOLMDZ output file.

Usage (from the repo root, with icoscope installed in editable mode):
    python tools/make_test_nc.py            # writes test_grid.nc here
    python tools/make_test_nc.py -o foo.nc  # custom output path
"""
import argparse
import os
from datetime import datetime, timezone

import numpy as np
from netCDF4 import Dataset

from icoscope.grid import goldberg

N = 20                   # Goldberg frequency → 4002 cells
NVERTEX = 6


def xyz_to_lonlat(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z, -1, 1)))
    return lon, lat


def main():
    ap = argparse.ArgumentParser(prog="icoscope-mktest",
                                 description="Generate a synthetic ICOLMDZ-like test NetCDF.")
    ap.add_argument("-o", "--output", default="test_grid.nc", metavar="PATH",
                    help="output file path (default: ./test_grid.nc)")
    args = ap.parse_args()
    OUTPATH = os.path.abspath(args.output)

    print(f"Generating Goldberg grid n={N} (relaxed)…")
    verts, cells, centers, iters = goldberg(N, relax=True, max_iterations=200)
    n_cells = len(cells)
    print(f"  {n_cells} cells (relax converged in {iters} iters)")

    centers = np.asarray(centers)
    clon, clat = xyz_to_lonlat(centers)

    # cell vertices → (cell, nvertex); pentagons pad by repeating last vertex
    blon = np.zeros((n_cells, NVERTEX))
    blat = np.zeros((n_cells, NVERTEX))
    for i, cell in enumerate(cells):
        vlon, vlat = xyz_to_lonlat(verts[cell])
        n_v = len(cell)
        blon[i, :n_v] = vlon
        blat[i, :n_v] = vlat
        if n_v < NVERTEX:
            blon[i, n_v:] = vlon[-1]
            blat[i, n_v:] = vlat[-1]

    # synthetic fields
    print("Generating synthetic fields…")
    lat_rad = np.radians(clat)
    lon_rad = np.radians(clon)

    # tas: surface air temperature, latitude gradient + zonal wave
    tas = 250.0 + 50.0 * np.cos(lat_rad) + 5.0 * np.cos(2 * lon_rad)

    # tas_anomaly: deviation from global mean (diverging field)
    tas_anomaly = tas - tas.mean()

    # precip: log-normal, equator-weighted
    rng = np.random.default_rng(42)
    precip = rng.lognormal(mean=0.0, sigma=1.5, size=n_cells)
    precip *= (1.0 + np.cos(lat_rad)) / 2

    # tas_t: seasonal cycle with strong NH/SH contrast that flips over the year.
    # `season` runs +1 (mid-summer NH) → -1 (mid-summer SH).
    # NH gets warmer when season > 0; SH gets warmer when season < 0.
    n_months = 12
    seasonal = np.zeros((n_months, n_cells))
    for m in range(n_months):
        season = np.sin(2 * np.pi * m / n_months)
        seasonal[m] = (280.0
                       + 25.0 * np.cos(lat_rad)              # warm equator
                       + 30.0 * season * np.sin(lat_rad)     # hemispheric flip
                       + 5.0  * np.cos(2 * lon_rad))         # zonal contrast

    # vort_t: a planetary-scale wave that rotates around the globe over the year.
    # Drastic moving pattern — great for spotting the time slider working.
    vort = np.zeros((n_months, n_cells))
    for m in range(n_months):
        # the wave's longitudinal phase advances; pattern peaks at mid-latitudes
        phase = 2 * np.pi * m / n_months
        vort[m] = (np.cos(2 * lat_rad)
                   * np.cos(3 * lon_rad + 4 * phase) * 1e-4)

    # tas_daily: a daily-cadence temperature field on a DISTINCT time axis
    # (``time_counter`` instead of ``time``). Lets the multi-pane cursor logic
    # be exercised — daily vs monthly mismatch is the realistic ICOLMDZ-ISO
    # scenario (histday vs histmth output files merged at runtime).
    n_days = 60
    daily = np.zeros((n_days, n_cells))
    for d in range(n_days):
        day_phase = 2 * np.pi * d / n_days
        daily[d] = (290.0
                    + 20.0 * np.cos(lat_rad)
                    + 5.0  * np.cos(day_phase) * np.sin(lat_rad))

    # write NetCDF
    print(f"Writing {OUTPATH}…")
    with Dataset(OUTPATH, "w", format="NETCDF4") as nc:
        nc.title = "Synthetic ICOLMDZ-like test grid"
        nc.history = f"Created {datetime.now(timezone.utc).isoformat()} by make_test_nc.py"
        nc.Conventions = "CF-1.8"
        nc.source = "IcoScope synthetic generator (tools/make_test_nc.py)"

        nc.createDimension("cell", n_cells)
        nc.createDimension("nvertex", NVERTEX)
        nc.createDimension("time", n_months)
        nc.createDimension("time_counter", n_days)

        lon = nc.createVariable("lon", "f4", ("cell",))
        lon[:] = clon
        lon.units = "degrees_east"
        lon.standard_name = "longitude"
        lon.bounds = "bounds_lon"

        lat = nc.createVariable("lat", "f4", ("cell",))
        lat[:] = clat
        lat.units = "degrees_north"
        lat.standard_name = "latitude"
        lat.bounds = "bounds_lat"

        nc.createVariable("bounds_lon", "f4", ("cell", "nvertex"))[:] = blon
        nc.createVariable("bounds_lat", "f4", ("cell", "nvertex"))[:] = blat

        t = nc.createVariable("time", "f4", ("time",))
        t[:] = np.arange(n_months) * 30.0
        t.units = "days since 2020-01-01"
        t.standard_name = "time"
        t.calendar = "noleap"

        # Second, daily-cadence time axis — covers the first 60 days of 2020,
        # so the monthly axis (one sample every 30 days for a year) only
        # overlaps it for the first ~two samples. Out-of-cursor-range banner
        # exercise lives here.
        tc = nc.createVariable("time_counter", "f4", ("time_counter",))
        tc[:] = np.arange(n_days, dtype="f4")
        tc.units = "days since 2020-01-01"
        tc.standard_name = "time"
        tc.calendar = "noleap"

        def add_field(name, data, units, long_name, standard_name=None, dims=("cell",)):
            v = nc.createVariable(name, "f4", dims)
            v[:] = data
            v.units = units
            v.long_name = long_name
            if standard_name:
                v.standard_name = standard_name
            v.coordinates = "lon lat"

        add_field("tas", tas, "K", "Near-surface air temperature", "air_temperature")
        add_field("tas_anomaly", tas_anomaly, "K", "Near-surface air temperature anomaly")
        add_field("precip", precip, "mm/day", "Precipitation", "precipitation_flux")
        add_field("tas_t", seasonal, "K", "Near-surface air temperature (monthly)",
                  dims=("time", "cell"))
        add_field("vort_t", vort, "s-1", "Mid-tropospheric vorticity (monthly)",
                  standard_name="atmosphere_relative_vorticity",
                  dims=("time", "cell"))
        add_field("tas_daily", daily, "K",
                  "Near-surface air temperature (daily)",
                  dims=("time_counter", "cell"))

    print(f"  wrote {OUTPATH}")
    print("  fields: tas (sequential), tas_anomaly (diverging), "
          "precip (log-distributed),")
    print("          tas_t (seasonal cycle, hemispheric flip), "
          "vort_t (rotating wave),")
    print("          tas_daily (daily cadence on time_counter axis)")


if __name__ == "__main__":
    main()
