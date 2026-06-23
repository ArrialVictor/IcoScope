"""Cell-value formatting helper (status-bar display)."""
import numpy as np

from icoscope.formatters import format_cell_value as _fmt
from icoscope.formatters import pretty_units


def test_float_typical_range_4_sig_figs():
    short, full = _fmt(287.41359, "K")
    assert short == "Value: 287.4 K"
    assert "287.4135" in full  # full precision preserved in tooltip


def test_float_scientific_for_small_magnitude():
    short, _ = _fmt(1.234e-5, "kg/kg")
    assert short.startswith("Value: 1.234e-05") or short.startswith("Value: 1.234e-5")
    assert "kg/kg" in short


def test_float_scientific_for_large_magnitude():
    short, _ = _fmt(2.5e6, "Pa")
    assert "e+0" in short
    assert "Pa" in short


def test_zero_renders_as_zero():
    short, _ = _fmt(0.0, "K")
    assert short == "Value: 0 K"


def test_nan_renders_as_no_data():
    short, full = _fmt(float("nan"), "K")
    assert short == "no data"
    assert "no data" in full


def test_integer_scalar_renders_without_decimals():
    short, _ = _fmt(np.int64(6), "")
    assert short == "Value: 6"  # categorical cell-kind, no units


def test_no_units_renders_value_only():
    short, _ = _fmt(45.7, "")
    assert short == "Value: 45.7"


def test_none_value_renders_as_no_data():
    short, _ = _fmt(None, "K")
    assert short == "no data"


def test_full_precision_in_tooltip():
    val = 1.2345678901234567
    _, full = _fmt(val, "K")
    # Tooltip should carry enough digits to be useful for copy-paste
    assert full.startswith("1.2345678901234")


def test_pretty_units_digit_after_letter_becomes_superscript():
    assert pretty_units("m2") == "m²"
    assert pretty_units("s2") == "s²"
    assert pretty_units("m2/s2") == "m²/s²"
    assert pretty_units("W/m2") == "W/m²"


def test_pretty_units_middle_dot_for_multiplication():
    assert pretty_units("kg/(s*m2)") == "kg/(s·m²)"


def test_pretty_units_negative_exponent():
    # CF-style "kg m-2 s-1" → "kg m⁻²·s⁻¹"-ish; the regex is conservative,
    # exact output depends on separators. Verify the superscript transform
    # itself works on negatives.
    assert pretty_units("s-1") == "s⁻¹"
    assert pretty_units("m-2") == "m⁻²"


def test_pretty_units_degrees_compass():
    assert pretty_units("degrees_north") == "°N"
    assert pretty_units("degrees_east") == "°E"
    assert pretty_units("degrees_C") == "°C"
    assert pretty_units("degrees") == "°"


def test_pretty_units_dimensionless():
    assert pretty_units("") == ""
    assert pretty_units("-") == ""
    assert pretty_units("1") == ""
    assert pretty_units(None) == ""


def test_pretty_units_leaves_simple_units_alone():
    assert pretty_units("K") == "K"
    assert pretty_units("Pa") == "Pa"
    assert pretty_units("kg/kg") == "kg/kg"   # no digits-after-letters


def test_format_cell_value_uses_pretty_units():
    short, _ = _fmt(287.4, "m2/s2")
    assert "m²/s²" in short
