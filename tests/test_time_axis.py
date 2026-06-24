"""Pure helpers under icoscope.time_axis — no netCDF4 dep needed."""
from datetime import datetime, timedelta

import pytest

from icoscope.time_axis import is_in_range, nearest_time_index


def _daily(start: datetime, n: int) -> list[datetime]:
    return [start + timedelta(days=i) for i in range(n)]


def test_nearest_single_sample_returns_zero():
    assert nearest_time_index(datetime(2026, 1, 5), [datetime(2026, 1, 1)]) == 0


def test_nearest_exact_match():
    times = _daily(datetime(2026, 1, 1), 10)
    assert nearest_time_index(datetime(2026, 1, 4), times) == 3


def test_nearest_before_first_clamps_to_zero():
    times = _daily(datetime(2026, 1, 10), 5)
    assert nearest_time_index(datetime(2025, 12, 1), times) == 0


def test_nearest_after_last_clamps_to_end():
    times = _daily(datetime(2026, 1, 1), 5)
    assert nearest_time_index(datetime(2026, 3, 1), times) == 4


def test_nearest_picks_closer_neighbor():
    times = [datetime(2026, 1, 1), datetime(2026, 1, 10)]
    # Cursor at Jan 3 — closer to Jan 1 (delta 2 days) than Jan 10 (7 days).
    assert nearest_time_index(datetime(2026, 1, 3), times) == 0
    # Cursor at Jan 8 — closer to Jan 10 (delta 2) than Jan 1 (delta 7).
    assert nearest_time_index(datetime(2026, 1, 8), times) == 1


def test_nearest_tie_returns_earlier_index():
    times = [datetime(2026, 1, 1), datetime(2026, 1, 3)]
    # Exactly between → earlier index wins.
    assert nearest_time_index(datetime(2026, 1, 2), times) == 0


def test_nearest_empty_raises():
    with pytest.raises(ValueError):
        nearest_time_index(datetime(2026, 1, 1), [])


def test_in_range_inside():
    times = _daily(datetime(2026, 1, 1), 10)
    assert is_in_range(datetime(2026, 1, 5), times)


def test_in_range_boundary_first():
    times = _daily(datetime(2026, 1, 1), 5)
    assert is_in_range(times[0], times)


def test_in_range_boundary_last():
    times = _daily(datetime(2026, 1, 1), 5)
    assert is_in_range(times[-1], times)


def test_in_range_before():
    times = _daily(datetime(2026, 1, 10), 5)
    assert not is_in_range(datetime(2025, 12, 1), times)


def test_in_range_after():
    times = _daily(datetime(2026, 1, 1), 5)
    assert not is_in_range(datetime(2026, 6, 1), times)


def test_in_range_empty():
    assert not is_in_range(datetime(2026, 1, 1), [])
