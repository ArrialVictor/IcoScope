"""Reentrancy probe for ``_on_pane_pick``.

Alternates picks across 4 panes with ``processEvents()`` between each
call, asserting that the displayed value belongs to the just-picked
pane's field. Fields are chosen with very different magnitudes so a
misattributed read jumps out by an order of magnitude.

A FAIL here means a second click re-entered :meth:`_on_pane_pick` while
the first was still in flight (via Qt events queued during the
multi-pane render loop), leaving status-bar state desynced from
selection state. Has stayed PASS since the cross-pane pick sync landed.
"""
from __future__ import annotations

import re

import numpy as np
from qtpy.QtCore import QCoreApplication


FIELDS = {0: "tas", 1: "tas_anomaly", 2: "precip", 3: "tas"}
EXPECTED_RANGE = {
    "tas": (180.0, 360.0),
    "tas_anomaly": (-80.0, 80.0),
    "precip": (0.0, 1000.0),
}


def _parse_value(label_text: str) -> float | None:
    """Pull the first signed-decimal number from a value-label string."""
    m = re.search(r"-?\d+\.?\d*(?:e-?\d+)?", label_text)
    return float(m.group(0)) if m else None


def _nearest_cell(win, lon: float, lat: float) -> int:
    """Return the cell index whose centre is closest to ``(lon, lat)``."""
    centers = np.asarray(win.centers)
    lats = np.degrees(np.arcsin(np.clip(centers[:, 2], -1, 1)))
    lons = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
    return int(np.argmin((lats - lat) ** 2 + (lons - lon) ** 2))


def test_pick_reentrancy_no_drift(make_main_window):
    """200 alternating picks across 4 panes never desync selection ↔ status."""
    win = make_main_window()
    win._on_pane_layout(4)
    QCoreApplication.processEvents()

    for idx, field in FIELDS.items():
        win.state.panes[idx].color_by = field
        win._refresh_scalars(idx)
        win._update_scalars_only(idx)
    win._select_pane(0)
    QCoreApplication.processEvents()

    cell_idx = _nearest_cell(win, lon=0.0, lat=0.0)

    failures: list[str] = []
    for i in range(200):
        target = i % 4
        win._on_pane_pick(target, cell_idx, lon=0.0, lat=0.0)
        QCoreApplication.processEvents()

        if win._selected_pane != target:
            failures.append(
                f"iter {i}: targeted pane {target}, "
                f"_selected_pane is {win._selected_pane}")
            continue

        value = _parse_value(win.value_label.text())
        if value is None:
            failures.append(f"iter {i}: empty value_label on pane {target}")
            continue
        lo, hi = EXPECTED_RANGE[FIELDS[target]]
        if not (lo <= value <= hi):
            failures.append(
                f"iter {i}: pane {target} expects {FIELDS[target]} "
                f"in [{lo}, {hi}] but value_label reads {value!r}")

    assert not failures, "\n  ".join(["reentrancy mismatches:", *failures[:10]])
