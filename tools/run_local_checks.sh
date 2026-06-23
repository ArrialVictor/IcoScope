#!/usr/bin/env bash
# Pre-push triple-check: dev venv tests, CI-equivalent venv tests, GUI
# smoke tests, ruff. Run from the repo root before every push.
#
# CI installs only a stripped dep set (no Qt, no pyvista) — module-top
# imports against those pass locally and fail in CI with ModuleNotFoundError.
# The CI-equivalent venv catches that before pushing.
#
# The GUI tests boot a real MainWindow and drive Qt's event loop; they
# are intentionally NOT run in CI (Qt + VTK in CI runners is flaky and
# adds Xvfb maintenance). They live under gui_tests/ and run here as
# part of the local pre-push.
set -euo pipefail

cd "$(dirname "$0")/.."

CI_VENV=/tmp/ci-venv
if [ ! -x "$CI_VENV/bin/python" ]; then
  echo "→ creating CI-equivalent venv at $CI_VENV"
  /opt/homebrew/bin/python3.12 -m venv "$CI_VENV"
  "$CI_VENV/bin/pip" install --quiet --upgrade pip
  "$CI_VENV/bin/pip" install --quiet numpy netCDF4 pytest Pillow
  "$CI_VENV/bin/pip" install --quiet --no-deps -e .
fi

echo "→ dev venv pytest (full deps)"
.venv/bin/python -m pytest tests/ -q

echo "→ CI-equivalent venv pytest (no Qt / no pyvista)"
"$CI_VENV/bin/python" -m pytest tests/ -q

echo "→ gui_tests (real MainWindow, drives Qt event loop)"
.venv/bin/python -m pytest gui_tests/ -q

echo "→ ruff"
uvx ruff check src/ tests/ tools/

echo "✓ all checks passed"
