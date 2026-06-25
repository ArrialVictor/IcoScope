"""Vertical-level unit classification: pick a display unit from the file's units attr.

Pure-logic tests for ``_DisplayBlock._classify_level_unit``: given a raw
levels array + units attribute string, the function returns the
``(factor, suffix, decimals, step)`` config that drives the spinbox
display + the type-in snap. The classifier is the single source of
truth for "what to show next to the slider" and "how to convert a
typed value back to the file's native scale", so it's worth headless
coverage rather than reaching through the Qt widget.
"""
from __future__ import annotations

import numpy as np
import pytest

# Importing _DisplayBlock at module level forces pyvistaqt / qtpy import,
# which is heavy and not present in the CI deps. Guard so the unit-test
# CI run skips this file cleanly when Qt is missing.
qt_missing = False
try:
    from icoscope.display_block import _DisplayBlock
except Exception:
    qt_missing = True

pytestmark = pytest.mark.skipif(qt_missing, reason="Qt/pyvistaqt not installed")


def _classify(values, units):
    return _DisplayBlock._classify_level_unit(
        np.asarray(values, dtype=float), units)


def test_pa_converts_to_hpa():
    cfg = _classify([100000.0, 50000.0, 10000.0], "Pa")
    assert cfg["factor"] == pytest.approx(0.01)   # /100
    assert cfg["suffix"] == " hPa"
    assert cfg["decimals"] == 1


def test_hpa_passes_through_unchanged():
    cfg = _classify([1000.0, 500.0, 100.0], "hPa")
    assert cfg["factor"] == 1.0
    assert cfg["suffix"] == " hPa"


def test_mbar_aliases_hpa():
    cfg = _classify([1000.0, 500.0], "mbar")
    assert cfg["suffix"] == " hPa"
    assert cfg["factor"] == 1.0


def test_meters_below_5km_kept_as_meters():
    cfg = _classify([100.0, 500.0, 1000.0, 2000.0], "m")
    assert cfg["suffix"] == " m"
    assert cfg["factor"] == 1.0
    assert cfg["decimals"] == 0


def test_meters_above_5km_displayed_as_km():
    cfg = _classify([100.0, 5000.0, 20000.0, 80000.0], "m")
    assert cfg["suffix"] == " km"
    assert cfg["factor"] == pytest.approx(0.001)


def test_km_passes_through():
    cfg = _classify([0.5, 5.0, 50.0], "km")
    assert cfg["suffix"] == " km"
    assert cfg["factor"] == 1.0


def test_kelvin_for_isentropic():
    cfg = _classify([300.0, 320.0, 350.0], "K")
    assert cfg["suffix"] == " K"


def test_empty_units_with_index_like_values_shows_idx():
    cfg = _classify([0.0, 1.0, 2.0, 3.0, 4.0], "")
    assert cfg["suffix"] == " (idx)"
    assert cfg["decimals"] == 0


def test_explicit_dimensionless_units_show_idx():
    cfg = _classify([100000.0, 50000.0], "1")
    # Even non-index-like values get "level k" because the units say
    # dimensionless — the file is telling us not to interpret the numbers.
    assert cfg["suffix"] == " (idx)"


def test_unknown_units_pass_through_verbatim():
    cfg = _classify([1.0, 2.0, 3.5], "wibble")
    assert cfg["suffix"] == " wibble"
    assert cfg["factor"] == 1.0


def test_case_insensitive_unit_matching():
    # File-spec units are case-sensitive in CF conventions but real
    # files sometimes drift ("PA", "pa", "Pa"); be forgiving.
    assert _classify([1000.0], "PA")["suffix"] == " hPa"
    assert _classify([1000.0], "pa")["suffix"] == " hPa"
    assert _classify([1000.0], "HPA")["suffix"] == " hPa"
