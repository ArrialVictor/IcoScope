"""Pure-Python text-formatting helpers used by the GUI status bar.

Kept in their own module (no Qt / VTK / PyVista imports) so the unit tests
can exercise them in CI environments that don't ship the GUI stack.
"""
from __future__ import annotations

import re

import numpy as np

# Map ASCII digits (and the minus sign) to their unicode superscripts.
_SUP_TABLE = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

# CF-style "degrees_X" suffixes that we render as a glyph instead of a word.
_DEGREE_REPLACEMENTS = {
    "degrees_north": "°N",
    "degrees_east": "°E",
    "degrees_south": "°S",
    "degrees_west": "°W",
    "degrees_C": "°C",
    "degrees_K": "K",
    "degrees": "°",
}

# Strings that mean "dimensionless" in CF / XIOS metadata. Render as empty.
_DIMENSIONLESS = {"", "-", "1"}


def pretty_units(s: str) -> str:
    """Render CF-style unit strings with unicode superscripts and middle dots.

    Transforms common patterns: ``m2/s2`` → ``m²/s²``, ``W/m2`` → ``W/m²``,
    ``kg/(s*m2)`` → ``kg/(s·m²)``, ``s-1`` → ``s⁻¹``, ``degrees_north`` →
    ``°N``, ``-`` and ``1`` → empty (dimensionless). Strings with no digits
    after letters are left untouched (e.g. ``kg/kg`` stays ``kg/kg``).

    Unknown strings are returned with the digit-superscript and ``*`` → ``·``
    transformations applied; everything else is left intact so we don't
    accidentally hide units we don't recognise.
    """
    if s is None:
        return ""
    s = s.strip()
    if s in _DIMENSIONLESS:
        return ""
    for src, dst in _DEGREE_REPLACEMENTS.items():
        s = s.replace(src, dst)
    # Digit run (optionally preceded by '-' for negative exponents) immediately
    # after a letter becomes a superscript: m2 → m², s-1 → s⁻¹. Use a regex so
    # we don't touch chemical formulae like H216O (digits already start with a
    # letter-followed-by-digit but those are surrounded by more letters/digits;
    # treat conservatively by only matching when the digits are followed by a
    # non-alphanumeric or end-of-string).
    s = re.sub(
        r"([a-zA-Z])(-?\d+)(?=[^0-9a-zA-Z]|$)",
        lambda m: m.group(1) + m.group(2).translate(_SUP_TABLE),
        s,
    )
    # Multiplication via '*' or stray whitespace between unit factors. Replace
    # '*' with '·' explicitly; whitespace is left alone (some CF strings use
    # spaces as separators, e.g. 'kg m-2 s-1', but turning every space into a
    # dot would mangle multi-word units that we'd rather preserve verbatim).
    s = s.replace("*", "·")
    return s


def format_cell_value(value, units: str) -> tuple[str, str]:
    """Return ``(short, full)`` text for a picked cell's scalar value.

    ``short`` is the status-bar display: 4 significant digits for typical
    floats; scientific notation outside ``[1e-3, 1e4]``; integers as-is;
    ``NaN`` → ``"no data"``. ``full`` is the tooltip body — the value's
    Python ``repr`` (full precision) followed by the units, for users who
    want to copy the number out.

    Units are pretty-printed via :func:`pretty_units` (superscripts, middle
    dots). Pass ``""`` for unit-less / dimensionless quantities.
    """
    units = pretty_units(units)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "no data", "no data at this cell"
    suffix = f" {units}" if units else ""
    v = np.asarray(value)
    if v.dtype.kind in "iu":
        text = f"{int(v)}{suffix}"
        return f"Value: {text}", text
    fv = float(v)
    if fv == 0.0:
        short = "0"
    elif abs(fv) >= 1e4 or abs(fv) < 1e-3:
        short = f"{fv:.3e}"
    else:
        short = f"{fv:.4g}"
    return f"Value: {short}{suffix}", f"{fv!r}{suffix}"
