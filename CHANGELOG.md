# Changelog

All notable changes are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
