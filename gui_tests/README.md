# gui_tests/

Slow, end-to-end smoke tests that boot a real `MainWindow`, load a
synthetic NetCDF, drive Qt's event loop with `processEvents()`, and
assert observable widget state. They cover what the unit suite under
[`tests/`](../tests/) can't: side effects in `app.py` slot handlers,
multi-pane state propagation, status-bar widgets, picker integration —
the things that previously could only be checked manually before a push.

## Running

```bash
.venv/bin/python -m pytest gui_tests/ -q
```

Or as part of the full pre-push check:

```bash
./tools/run_local_checks.sh
```

A window briefly appears on screen for each test (it's the real
MainWindow — there's no headless mode that doesn't crash inside VTK on
macOS). Total runtime ~15 s for the current suite.

**These tests are NOT run in CI.** The CI test job installs only the
headless dep set (no Qt, no pyvista — see `.github/workflows/ci.yml`)
and Qt + VTK in GitHub Actions runners is flaky enough that hooking
them up wouldn't be a net win. Run them locally before pushing.

## Conventions

- One assertion per concept; tests are by-design slow, so prefer 4
  focused tests over 1 omnibus.
- Use the `make_main_window` fixture from `conftest.py` — it builds a
  ready-to-use window with the synthetic NetCDF loaded.
- Drive Qt with `QCoreApplication.processEvents()` after every state
  mutation (layout switch, color-by change, pick call). Without it
  signals don't fire and assertions race ahead of the widget state.
- Do **not** call `win.close()` at end of test. The fixture's teardown
  intentionally skips it — closing inside the QApplication lifetime
  crashes VTK on macOS. Let process exit handle cleanup.

## Why this exists

PR #28 (cross-pane cell-pick sync) shipped two regressions that the
unit suite missed because they were widget-level side effects:

- mesh-change left stale `value_label` / lon-lat text
- layout shrink+expand kept stale highlight outlines on re-revealed
  panes

Both would have been caught by a 4-line `gui_tests/test_*.py`. The
harness pays for itself the first time it catches a regression that
would have shipped.

## Adding a new test

Mirror the existing files — typically:

```python
def test_my_feature(make_main_window):
    win = make_main_window()
    win._on_pane_layout(4)
    QCoreApplication.processEvents()
    # …drive feature…
    assert <expected widget state>
```

If a setup step needs the per-pane state list to be length 4, call
`_on_pane_layout(4)` **before** writing `state.panes[3].color_by`, etc.
The list is grown lazily on layout expand.
