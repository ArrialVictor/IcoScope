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
    Factory that returns a configured :class:`MainWindow` with a fresh
    synthetic ICOLMDZ NetCDF loaded into the File tab. Tests get a
    realistic 4-pane-capable window without any GUI interaction.

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


@pytest.fixture
def make_main_window(qapp, synthetic_nc) -> Iterator:
    """Yield a factory that builds a configured :class:`MainWindow`.

    The factory loads the synthetic NetCDF into the File tab and shows
    the window before returning. Multiple calls inside one test are
    supported (e.g. testing two independent sessions), but typical usage
    is one window per test.

    The window is closed and the synthetic NetCDF FileContext is freed
    when the test finishes, so each test starts clean.
    """
    from icoscope.app import MainWindow
    from icoscope.grid import goldberg
    from icoscope.loader import FileContext, load_grid, read_levels
    from icoscope.tabs import Tab

    created: list = []

    def _build():
        verts, cells, centers, _ = goldberg(8, relax=True)
        win = MainWindow(verts, cells, np.asarray(centers), initial_n=8)

        f_verts, f_cells, f_centers, fields = load_grid(str(synthetic_nc))
        levels = read_levels(str(synthetic_nc))
        win.file_path = str(synthetic_nc)
        win._file_state.file_fields = fields
        win._file_state.file_levels = levels
        win._file_cache = {
            "path": str(synthetic_nc),
            "verts": f_verts,
            "cells": f_cells,
            "centers": np.asarray(f_centers),
            "fields": fields,
            "levels": levels,
            "context": FileContext(str(synthetic_nc)),
        }
        win.panel.file_tab.set_file_loaded(True)
        win._sync_file_info(str(synthetic_nc))
        win._activate_file_view()
        win.panel.tabs.setCurrentIndex(Tab.FILE)
        win.show()
        QCoreApplication.processEvents()
        created.append(win)
        return win

    yield _build

    # Deliberately do NOT call ``win.close()`` here. Closing a MainWindow
    # inside the same QApplication lifetime crashes Qt/VTK on macOS
    # (closing tears down the QtInteractor render windows in a way
    # subsequent processEvents calls can't recover from). Instead drop
    # the references and let process exit clean everything up — fine for
    # the run-one-file-per-process pattern that tools/run_gui_tests.sh
    # uses.
    for win in created:
        ctx = (win._file_cache or {}).get("context")
        if ctx is not None and hasattr(ctx, "close"):
            try:
                ctx.close()
            except Exception:
                pass
    created.clear()


@pytest.fixture
def tmp_export_dir() -> Iterator[Path]:
    """Provide a temporary directory for tests that write exported files."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)
