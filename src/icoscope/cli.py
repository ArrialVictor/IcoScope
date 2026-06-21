"""IcoScope CLI entry point."""
import argparse
import sys
import time

from .grid import goldberg
from .lonlat import latlon_mesh


def _step(msg: str) -> float:
    """Print a progress line to stderr (so it shows up live) and return now()."""
    sys.stderr.write(f"  {msg}…\n")
    sys.stderr.flush()
    return time.time()


def main() -> None:
    """Parse command-line arguments, build (or load) the grid, and launch the viewer."""
    ap = argparse.ArgumentParser(
        prog="icoscope",
        description="Interactive 3D viewer for icosahedral hex/pent grids on a sphere."
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--generate", "-n", type=int, default=40, metavar="N",
                     help="generate a Goldberg grid of frequency N "
                          "(default 40 — matches ICOLMDZ's typical low-res nbp=40)")
    src.add_argument("--file", "-f", type=str, metavar="PATH",
                     help="load grid from NetCDF file")
    ap.add_argument("--grid", choices=("ico", "lonlat"), default="ico",
                    help="initial synthetic mesh family (default 'ico'). "
                         "'lonlat' builds an LMDZ dyn3d-style regular lat-lon "
                         "mesh; -n and --zoom-* are ignored in that mode.")
    ap.add_argument("--iim", type=int, default=96, metavar="N",
                    help="LonLat: number of distinct longitudes (default 96, "
                         "LMDZ low-res). Only used with --grid lonlat.")
    ap.add_argument("--jjm", type=int, default=95, metavar="N",
                    help="LonLat: number of latitude bands (default 95, "
                         "LMDZ low-res). Only used with --grid lonlat.")
    ap.add_argument("--no-relax", action="store_true",
                    help="don't run spring-relaxation on the synthetic grid")
    ap.add_argument("--zoom-factor", type=float, default=1.0, metavar="F",
                    help="Schmidt stretching factor for the synthetic grid "
                         "(default 1.0 = uniform; >1 concentrates cells at the focal point)")
    ap.add_argument("--zoom-lon", type=float, default=0.0, metavar="DEG",
                    help="Schmidt focal-point longitude in degrees (default 0.0)")
    ap.add_argument("--zoom-lat", type=float, default=45.0, metavar="DEG",
                    help="Schmidt focal-point latitude in degrees (default 45.0)")
    ap.add_argument("--lmdz-clon", type=float, default=0.0, metavar="DEG",
                    help="LMDZ tanh zoom: focal-point longitude in degrees "
                         "(LonLat tab only, default 0.0).")
    ap.add_argument("--lmdz-clat", type=float, default=0.0, metavar="DEG",
                    help="LMDZ tanh zoom: focal-point latitude in degrees "
                         "(LonLat tab only, default 0.0).")
    ap.add_argument("--lmdz-grossismx", type=float, default=1.0, metavar="G",
                    help="LMDZ tanh zoom: longitudinal refinement factor "
                         "(default 1.0 = uniform).")
    ap.add_argument("--lmdz-grossismy", type=float, default=1.0, metavar="G",
                    help="LMDZ tanh zoom: latitudinal refinement factor "
                         "(default 1.0 = uniform).")
    ap.add_argument("--lmdz-dzoomx", type=float, default=0.0, metavar="F",
                    help="LMDZ tanh zoom: longitudinal half-width as a "
                         "fraction of 2π (default 0.0).")
    ap.add_argument("--lmdz-dzoomy", type=float, default=0.0, metavar="F",
                    help="LMDZ tanh zoom: latitudinal half-width as a "
                         "fraction of π (default 0.0).")
    ap.add_argument("--lmdz-taux", type=float, default=3.0, metavar="T",
                    help="LMDZ tanh zoom: longitudinal transition sharpness "
                         "(default 3.0).")
    ap.add_argument("--lmdz-tauy", type=float, default=3.0, metavar="T",
                    help="LMDZ tanh zoom: latitudinal transition sharpness "
                         "(default 3.0).")
    ap.add_argument("--describe", action="store_true",
                    help="(with --file) print variables and exit")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="suppress startup progress messages")
    args = ap.parse_args()

    log = (lambda _msg: time.time()) if args.quiet else _step

    sys.stderr.write("IcoScope starting (first launch can take a few seconds)\n")
    sys.stderr.flush()

    t0 = log("loading data")
    if args.file:
        from .loader import describe, load_grid
        if args.describe:
            describe(args.file)
            return
        verts, cells, centers, _fields = load_grid(args.file)
        if not args.quiet:
            sys.stderr.write(
                f"  loaded {len(cells)} cells from {args.file} "
                f"({time.time()-t0:.1f}s)\n"
            )
    elif args.grid == "lonlat":
        t1 = log(f"building synthetic lat-lon grid iim={args.iim}, jjm={args.jjm}")
        verts, cells, centers = latlon_mesh(
            iim=args.iim, jjm=args.jjm,
            clon=args.lmdz_clon, clat=args.lmdz_clat,
            grossismx=args.lmdz_grossismx, grossismy=args.lmdz_grossismy,
            dzoomx=args.lmdz_dzoomx, dzoomy=args.lmdz_dzoomy,
            taux=args.lmdz_taux, tauy=args.lmdz_tauy,
        )
        if not args.quiet:
            sys.stderr.write(
                f"  {len(cells)} cells "
                f"({sum(len(c)==3 for c in cells)} polar triangles, "
                f"{sum(len(c)==4 for c in cells)} quads) "
                f"in {time.time()-t1:.1f}s\n"
            )
    else:
        zoom_note = (f", Schmidt zoom factor={args.zoom_factor} "
                     f"at ({args.zoom_lon}, {args.zoom_lat})"
                     if abs(args.zoom_factor - 1.0) >= 1e-12 else "")
        t1 = log(f"building synthetic grid n={args.generate}"
                 f"{' (with relaxation)' if not args.no_relax else ''}"
                 f"{zoom_note}")
        verts, cells, centers, _iters = goldberg(
            n=args.generate,
            relax=not args.no_relax,
            zoom_factor=args.zoom_factor,
            zoom_lon=args.zoom_lon,
            zoom_lat=args.zoom_lat,
        )
        if not args.quiet:
            sys.stderr.write(
                f"  {len(cells)} cells "
                f"({sum(len(c)==5 for c in cells)} pentagons, "
                f"{sum(len(c)==6 for c in cells)} hexagons) "
                f"in {time.time()-t1:.1f}s\n"
            )

    log("initializing Qt + VTK (first launch is slow, subsequent are fast)")
    from .app import run
    log("opening window")
    run(verts, cells, centers, initial_n=args.generate, relax=not args.no_relax,
        file_path=args.file,
        zoom_factor=args.zoom_factor, zoom_lon=args.zoom_lon, zoom_lat=args.zoom_lat,
        initial_grid=args.grid, iim=args.iim, jjm=args.jjm,
        lmdz_clon=args.lmdz_clon, lmdz_clat=args.lmdz_clat,
        lmdz_grossismx=args.lmdz_grossismx, lmdz_grossismy=args.lmdz_grossismy,
        lmdz_dzoomx=args.lmdz_dzoomx, lmdz_dzoomy=args.lmdz_dzoomy,
        lmdz_taux=args.lmdz_taux, lmdz_tauy=args.lmdz_tauy)


if __name__ == "__main__":
    main()
