"""Coastlines on the sphere.

Downloads a low-res Natural Earth coastline GeoJSON (cached locally) and
projects it onto the unit sphere as 3D polylines for PyVista.
"""
import json
import os
import urllib.request
from typing import Any

import numpy as np

URL = ("https://raw.githubusercontent.com/martynafford/"
       "natural-earth-geojson/master/110m/physical/ne_110m_coastline.json")

_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "icoscope",
)
CACHE = os.path.join(_CACHE_DIR, "ne_110m_coastline.json")


def _load_geojson() -> dict:
    if not os.path.exists(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        print(f"Downloading coastlines → {CACHE}")
        urllib.request.urlretrieve(URL, CACHE)
    with open(CACHE) as f:
        return json.load(f)


def _lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray, radius: float = 1.0) -> np.ndarray:
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    return np.stack([
        radius * np.cos(lat) * np.cos(lon),
        radius * np.cos(lat) * np.sin(lon),
        radius * np.sin(lat),
    ], axis=-1)


_CACHE = {}


def coastline_polydata(radius: float = 1.001) -> Any:
    """Return a ``pv.PolyData`` of coastline polylines slightly above the unit sphere.

    Cached: subsequent calls with the same ``radius`` return the same object.
    The return type is ``pyvista.PolyData`` — typed as ``Any`` to keep the
    pyvista import lazy.
    """
    if radius in _CACHE:
        return _CACHE[radius]
    import pyvista as pv
    gj = _load_geojson()
    all_pts = []
    lines = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        segs = (geom["coordinates"] if geom["type"] == "MultiLineString"
                else [geom["coordinates"]])
        for seg in segs:
            arr = np.array(seg, dtype=float)  # (k, 2) lon, lat
            pts = _lonlat_to_xyz(arr[:, 0], arr[:, 1], radius=radius)
            start = len(all_pts)
            all_pts.extend(pts)
            lines.append(len(pts))
            lines.extend(range(start, start + len(pts)))
    poly = pv.PolyData(np.array(all_pts), lines=np.array(lines, dtype=np.int64))
    _CACHE[radius] = poly
    return poly
