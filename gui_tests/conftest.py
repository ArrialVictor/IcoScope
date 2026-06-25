"""Shared fixtures for the GUI test harness.

Tests under ``gui_tests/`` boot a real :class:`~icoscope.app.MainWindow`
into a real :class:`QApplication` and drive Qt's event loop by hand with
:func:`QCoreApplication.processEvents`. They are intentionally **not**
run in CI — the CI test job installs only the headless dep set (no Qt,
no pyvista), and Qt + VTK in CI runners is notoriously flaky regardless.
Run locally via ``tools/run_local_checks.sh`` or
``.venv/bin/python -m pytest gui_tests -q`` as part of the pre-push
check.

Fixtures
--------
qapp
    Session-scoped :class:`QApplication` — building a fresh one per test
    is slow and macOS sometimes refuses a second NSApplication, so we
    reuse a single instance for the whole session.
make_main_window
    Function-scoped factory that returns a configured :class:`MainWindow`
    with a fresh synthetic ICOLMDZ NetCDF loaded into the File tab. Use
    this when a test needs full isolation (the default).
make_module_window
    Module-scoped variant of the above. Use this in a module-scoped
    setup fixture when the construction cost dominates the test body
    and the per-test mutations can be cleanly reset (see
    ``test_timeline_strip_phase2.py`` for the pattern). Tears down once
    at end-of-module.
set_field
    Session-scoped callable: ``set_field(win, pane_idx, field)`` selects
    the pane, changes its ``color_by``, and drains the event queue.

Usage
-----
::

    def test_something(make_main_window):
        win = make_main_window()
        win._on_pane_layout(4)
        QCoreApplication.processEvents()
        assert win._pane_container.n_visible == 4
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtCore import QCoreApplication
from qtpy.QtWidgets import QApplication


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Return the shared :class:`QApplication` for this test session."""
    app = QApplication.instance() or QApplication(sys.argv)
    return app


@pytest.fixture(scope="session")
def synthetic_nc(tmp_path_factory) -> Path:
    """Generate a synthetic ICOLMDZ-shaped NetCDF once per session."""
    out = tmp_path_factory.mktemp("synthetic-nc") / "test_grid.nc"
    subprocess.run(
        [sys.executable, "tools/make_test_nc.py", "-o", str(out)],
        check=True, cwd=REPO_ROOT, capture_output=True,
    )
    return out


def _build_main_window(synthetic_nc: Path):
    """Construct a configured :class:`MainWindow` with the synthetic file loaded.

    Shared internals for the :func:`make_main_window` and
    :func:`make_module_window` factories; not a fixture itself so both
    can call it without scope mismatch.
    """
    from icoscope.app import MainWindow
    from icoscope.grid import goldberg
    from icoscope.loader import FileContext, load_grid, read_levels
    from icoscope.tabs import Tab

    verts, cells, centers, _ = goldberg(8, relax=True)
    win = MainWindow(verts, cells, np.asarray(centers), initial_n=8)

    f_verts, f_cells, f_centers, fields = load_grid(str(synthetic_nc))
    levels_result = read_levels(str(synthetic_nc))
    levels, level_units = (
        levels_result if levels_result is not None else (None, "")
    )
    win.file_path = str(synthetic_nc)
    win._file_state.file_fields = fields
    win._file_state.file_levels = levels
    win._file_state.file_level_units = level_units
    win._file_cache = {
        "path": str(synthetic_nc),
        "verts": f_verts,
        "cells": f_cells,
        "centers": np.asarray(f_centers),
        "fields": fields,
        "levels": levels,
        "level_units": level_units,
        "context": FileContext(str(synthetic_nc)),
    }
    win.panel.file_tab.set_file_loaded(True)
    win._sync_file_info(str(synthetic_nc))
    win._activate_file_view()
    win.panel.tabs.setCurrentIndex(Tab.FILE)
    win.show()
    QCoreApplication.processEvents()
    return win


def _teardown_main_window(win) -> None:
    """Free the synthetic NetCDF FileContext and hide the window.

    Hides instead of closing — closing tears down the QtInteractor render
    windows in a way subsequent ``processEvents`` calls can't recover
    from (segfault on macOS).
    """
    ctx = (win._file_cache or {}).get("context")
    if ctx is not None and hasattr(ctx, "close"):
        try:
            ctx.close()
        except Exception:
            pass
    win.hide()


@pytest.fixture
def make_main_window(qapp, synthetic_nc) -> Iterator:
    """Function-scoped factory that builds a configured :class:`MainWindow`.

    The factory loads the synthetic NetCDF into the File tab and shows
    the window before returning. Multiple calls inside one test are
    supported, but typical usage is one window per test.

    The window is hidden and the synthetic NetCDF FileContext is freed
    when the test finishes, so each test starts clean.
    """
    created: list = []

    def _build():
        win = _build_main_window(synthetic_nc)
        created.append(win)
        return win

    yield _build

    for win in created:
        _teardown_main_window(win)
    QCoreApplication.processEvents()
    created.clear()


@pytest.fixture(scope="module")
def make_module_window(qapp, synthetic_nc) -> Iterator:
    """Module-scoped variant of :func:`make_main_window`.

    Returns a callable that builds ONE window for every test in the
    module to share. Tear-down happens after every test in the module
    has run, not after each individual test.

    Use this only when:
    - the per-test construction cost (~1 s of Qt + ~2-3 s of PyVista
      ``add_mesh`` per pane) dominates the test body,
    - **and** the per-test mutations can be cleanly reset back to the
      shared configured state (see the autouse reset fixture pattern
      in ``test_timeline_strip_phase2.py``).

    Otherwise prefer :func:`make_main_window` — fresh-per-test is the
    safer default.
    """
    created: list = []

    def _build():
        win = _build_main_window(synthetic_nc)
        created.append(win)
        return win

    yield _build

    for win in created:
        _teardown_main_window(win)
    QCoreApplication.processEvents()
    created.clear()


@pytest.fixture
def tmp_export_dir() -> Iterator[Path]:
    """Provide a temporary directory for tests that write exported files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture(scope="session")
def set_field():
    """Helper: select a pane, change its ``color_by``, drain the event queue.

    Session-scoped because the returned callable is stateless — sharing
    it lets module-scoped fixtures depend on it without a scope
    mismatch. ``_select_pane`` is synchronous (no signals fire), so only
    one ``processEvents`` drain at the end is required.
    """
    def _set(win, pane_idx: int, field: str) -> None:
        win._select_pane(pane_idx)
        win._on_color_by(field)
        QCoreApplication.processEvents()
    return _set
