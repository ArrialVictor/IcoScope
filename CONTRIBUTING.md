# Contributing

Short guide for working on IcoScope. For what the tool does and how to use
it, see [README.md](README.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
The configuration lives in `pyproject.toml`; nothing extra to install.

```bash
ruff check src/ tests/ tools/
```

CI runs the same command on every push and pull request — keep it green.

## Docstrings

**Every public module, class, function, and method has a NumPy-style
docstring.** Ruff's `D` rules enforce this. The convention is set to
`numpy` in `pyproject.toml`.

Minimal example:

```python
def schmidt_stretch(points, factor, lon_deg, lat_deg):
    """Apply DYNAMICO's Schmidt conformal stretch to points on the unit sphere.

    Parameters
    ----------
    points : ndarray
        ``(N, 3)`` array of unit-sphere positions.
    factor : float
        Stretching factor; ``1.0`` is the identity, ``> 1`` concentrates
        cells at the focal point.
    lon_deg, lat_deg : float
        Focal-point coordinates in degrees.

    Returns
    -------
    ndarray
        ``(N, 3)`` stretched positions, re-normalized to the unit sphere.
    """
```

What's exempt (configured as per-file ignores):

- `tests/` — pytest assertions speak for themselves.
- `tools/` — one-off scripts, not user-facing.
- Magic methods (`__init__`, `__repr__`, …) and `__init__.py` files —
  the class or package docstring covers them.

When the rules feel wrong for a specific case, prefer a `# noqa: D...`
comment on the offending line over weakening the global config.

## Tests

```bash
pytest -q
```

Tests live in `tests/`. The Qt-heavy code in `src/icoscope/app.py` and
`src/icoscope/controls.py` is not covered by the suite because Qt needs
a display — CI runs only the grid and loader tests. If you change those
modules, exercise them by launching `icoscope` locally before pushing.

## Commits

- Use the conventional-commit prefixes the existing history uses:
  `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`.
- Keep messages focused — one logical change per commit.
- Do **not** add a `Co-Authored-By:` trailer; the local content-policy
  hook blocks it.
