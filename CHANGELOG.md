# Changelog

All notable changes are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.2] — 2026-06-21 (alpha)

### Added
- **Side panel restructured into three independent tabs**: `Ico` (synthetic
  Goldberg mesh + Schmidt zoom), `LonLat` (regular lat-lon mesh + LMDZ tanh
  zoom), and `File` (loaded NetCDF). Each tab is a complete, independent
  visualization configuration — switching tabs swaps the rendered mesh and
  preserves per-tab display settings (color-by, colormap, overlays,
  animation, export). Theme moved to a `View → Theme` menu.
- **Schmidt zoom on the Ico tab**: faithful Python port of DYNAMICO's
  `schmidt_transform` (Guo & Drake 2005). Activate/Deactivate toggle with
  live spinbox re-apply; CLI flags `--zoom-factor / --zoom-lon / --zoom-lat`.
- **LonLat mesh generator** (`lonlat.py::latlon_mesh`): LMDZ dyn3d's
  classical Arakawa C-grid with polar triangle fans + interior quads.
  CLI flags `--grid lonlat --iim N --jjm M`.
- **LMDZ tanh zoom on the LonLat tab**: port of `fxhyp_m.f90` / `fyhyp_m.f90`
  with the 8-parameter family (`clon, clat, grossismx/y, dzoomx/y, taux/y`).
  Activate/Deactivate toggle, live re-apply, snap-back on invalid
  combinations, validity check disables the toggle when off and the
  combination is bad.
- **dyn3d NetCDF support**: loader auto-detects files with the four
  `rlonu/rlatu/rlonv/rlatv` coord arrays, reconstructs cell polygons, and
  surfaces `(rlatu, rlonv)` data variables in the "Color by" combo.
- **Mesh caching**: switching tabs is now instant — each tab's mesh is
  cached keyed by its parameters and reused on tab switch.
- **Adaptive scrollable tabs**: tab pages scroll vertically when the
  window is shorter than the active tab's content; tab height adapts to
  the active tab.
- **Per-tab Color by selector** with synthetic field options for Ico /
  LonLat (Latitude, Cell kind, Mock & Realistic temperature) and the
  loaded file's actual variables for the File tab.
- **File tab summary**: filename, full path (tooltip), `N cells · M fields
  · K time steps` line, NetCDF `title` and `source` global attributes.
- **Empty-state sphere**: the File tab shows a plain Icosphere when no
  NetCDF is loaded, instead of a stale mesh from another tab.
- **`tools/make_dyn3d_test_nc.py`**: companion to `make_test_nc.py` for
  generating synthetic dyn3d NetCDF files for development.

### Changed
- Side panel no longer has a shared display section below the tabs.
- Status bar errors for invalid LMDZ-zoom combinations flash in **bold red**.
- `loader.py::load_grid` accepts both icosahedral CF-bounds files and LMDZ
  dyn3d files in a single call (signature unchanged).
- Email in `pyproject.toml` switched from a personal address to the
  GitHub noreply address.

### Fixed
- Color-by carryover bug when switching tabs (file-field selection no
  longer leaks into the Ico tab's coloring).
- Spin timer now respects the active tab — auto-rotate enabled on one
  tab no longer spins another tab's mesh.
- File tab's cmap / colorbar / "Symmetric scale" widgets are properly
  enabled when a file loads (previously stayed greyed out).
- Empty-state sphere actor properly removed when switching back to a
  real mesh (avoided stacked-mesh "patterns" bug).

## [0.1.0] — 2026-06-18 (alpha)

First public release.

### Added
- Synthetic Goldberg-grid generator with optional DYNAMICO-style spring
  relaxation (early-stop on edge-length-CV convergence).
- NetCDF loader for the CF-convention icosahedral layout, with auto-detection
  of common variable-name variants.
- Dynamic "Color by" dropdown populated from the loaded file's variables.
- Time slider and play/pause controls for time-varying fields.
- Coastlines (Natural Earth 1:110m) and 30° graticule overlays, both with
  per-element color overrides.
- Themes: Dark, Light, CB-safe. Colormap list restricted to perceptually
  uniform / colorblind-safe options.
- Symmetric-scale-around-0 option for diverging fields (anomalies, vorticity).
- Click-to-pick exact cell (analytic ray-sphere intersection +
  `vtkCellLocator`), with magenta outline highlight and editable lon/lat
  fields that double as a camera-teleport input.
- Auto-rotate at configurable speed.
- PNG screenshot (with optional transparent background) and SVG vector export.
- Help menu with keyboard reference and NetCDF schema notes.
- Test suite covering grid topology, relaxation invariants, and loader
  round-trips against a synthetic file.
