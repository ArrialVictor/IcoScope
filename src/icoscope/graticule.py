"""Latitude/longitude reference lines on the sphere."""
import numpy as np


def _lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray, radius: float) -> np.ndarray:
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    return np.stack([
        radius * np.cos(lat) * np.cos(lon),
        radius * np.cos(lat) * np.sin(lon),
        radius * np.sin(lat),
    ], axis=-1)


_CACHE = {}


def graticule_polydata(radius: float = 1.001, spacing: int = 30, density: int = 180):
    """Return a ``pv.PolyData`` of lat/lon lines at the given spacing (degrees).

    Cached by ``(radius, spacing, density)``.
    """
    key = (radius, spacing, density)
    if key in _CACHE:
        return _CACHE[key]
    import pyvista as pv
    pts, lines = [], []

    def add(arr):
        start = len(pts)
        pts.extend(arr)
        lines.append(len(arr))
        lines.extend(range(start, start + len(arr)))

    # parallels (constant latitude)
    for lat in range(-90 + spacing, 90, spacing):
        lons = np.linspace(-180, 180, density + 1)
        add(_lonlat_to_xyz(lons, np.full_like(lons, lat), radius))

    # meridians (constant longitude)
    for lon in range(-180, 180, spacing):
        lats = np.linspace(-90, 90, density + 1)
        add(_lonlat_to_xyz(np.full_like(lats, lon), lats, radius))

    poly = pv.PolyData(np.array(pts), lines=np.array(lines, dtype=np.int64))
    _CACHE[key] = poly
    return poly
