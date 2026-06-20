"""IcoScope CLI entry point."""
import argparse
import sys
import time

from .grid import goldberg


def _step(msg: str) -> float:
    """Print a progress line to stderr (so it shows up live) and return now()."""
    sys.stderr.write(f"  {msg}…\n")
    sys.stderr.flush()
    return time.time()


def main():
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
    ap.add_argument("--no-relax", action="store_true",
                    help="don't run spring-relaxation on the synthetic grid")
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
    else:
        t1 = log(f"building synthetic grid n={args.generate}"
                 f"{' (with relaxation)' if not args.no_relax else ''}")
        verts, cells, centers, _iters = goldberg(n=args.generate,
                                                 relax=not args.no_relax)
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
        file_path=args.file)


if __name__ == "__main__":
    main()
