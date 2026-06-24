"""Time-axis helpers — cursor resolution across multiple per-field time axes.

The multi-pane viewer drives every pane's time index from one shared
*cursor* datetime. Each pane's variable can sit on its own time axis
(e.g. one daily, another monthly), so the cursor must be resolved
separately for each pane to a nearest sample on its own axis. These
helpers do that math without touching netCDF4 — they only need a
sequence of comparable datetime-like objects.

Pure functions live here so they're cheap to unit-test and don't pull
the data layer into headless code paths.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def nearest_time_index(cursor: T, times: Sequence[T]) -> int:
    """Return the index in ``times`` whose datetime is closest to ``cursor``.

    Parameters
    ----------
    cursor
        A datetime-like value (``cftime.datetime``, ``datetime``,
        ``numpy.datetime64`` — anything that supports subtraction giving
        a comparable, abs-able delta).
    times
        Monotonic-ascending sequence of values of the same kind as
        ``cursor``. Must be non-empty.

    Returns
    -------
    int
        Index of the nearest sample. If two samples are equidistant
        (cursor exactly between them), returns the earlier index.

    Raises
    ------
    ValueError
        If ``times`` is empty — callers must guard against this since
        "no nearest sample" has no sensible integer answer.
    """
    n = len(times)
    if n == 0:
        raise ValueError("times is empty; no nearest sample exists")
    if n == 1:
        return 0
    # Linear scan with absolute-delta tracking. O(N) per call; N is
    # typically in the hundreds, so binary-search optimisation isn't
    # worth the complexity tax until proven necessary on a real file.
    best_idx = 0
    best_delta = abs(times[0] - cursor)
    for i in range(1, n):
        delta = abs(times[i] - cursor)
        # Strictly less than — on a tie, keep the earlier index.
        if delta < best_delta:
            best_idx = i
            best_delta = delta
    return best_idx


def last_previous_time_index(cursor: T, times: Sequence[T]) -> int:
    """Return the index of the latest sample at or before ``cursor``.

    The "physically-correct" resolution mode for climate data, where a
    sample dated ``t`` represents data that became valid AT ``t`` (a
    monthly mean for January is typically stamped 1 Feb, reflecting
    "data is now in for January"). A cursor on 15 Feb should therefore
    surface January's mean (the latest sample whose timestamp ≤ cursor),
    not February's (which isn't computed yet).

    Parameters
    ----------
    cursor
        A datetime-like value (``cftime.datetime``, ``datetime``,
        ``numpy.datetime64``) supporting ``<=`` comparison with the
        entries of ``times``.
    times
        Monotonic-ascending sequence of comparable values. Must be
        non-empty.

    Returns
    -------
    int
        Index of the latest sample with ``times[i] <= cursor``. If the
        cursor falls before every sample, clamps to 0 (no earlier sample
        exists — show the first one rather than blanking the pane).

    Raises
    ------
    ValueError
        If ``times`` is empty — callers must guard against this.
    """
    n = len(times)
    if n == 0:
        raise ValueError("times is empty; no sample exists")
    # Linear scan from the end — typical N is small and tail-of-axis
    # lookups (during a scrub past the end) hit on the very first
    # iteration. Could binary-search if N ever grows past a few thousand.
    for i in range(n - 1, -1, -1):
        if times[i] <= cursor:
            return i
    return 0


def is_in_range(cursor: T, times: Sequence[T]) -> bool:
    """Return ``True`` if ``cursor`` falls inside ``[times[0], times[-1]]``.

    Out-of-range means the pane is showing a clamped nearest sample
    (its first or last), not a genuinely-corresponding sample — the
    caller surfaces a banner like ``"Showing <nearest> (cursor at
    <cursor>)"`` in that case.

    An empty ``times`` is treated as out-of-range (vacuously).
    """
    if len(times) == 0:
        return False
    return times[0] <= cursor <= times[-1]
