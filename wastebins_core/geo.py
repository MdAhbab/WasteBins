"""
Great-circle geometry and matrix helpers.

Distances are metric (metres) throughout; travel *times* are produced by
:mod:`wastebins_core.traffic`, which turns a distance into a duration using a
context-dependent speed rather than a single global constant.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np

R_EARTH_M = 6_371_008.8  # IUGG mean Earth radius

Coord = Tuple[float, float]  # (latitude, longitude) in degrees


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    p1 = math.radians(lat1 or 0.0)
    p2 = math.radians(lat2 or 0.0)
    dp = math.radians((lat2 or 0.0) - (lat1 or 0.0))
    dl = math.radians((lon2 or 0.0) - (lon1 or 0.0))
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return R_EARTH_M * 2.0 * math.asin(math.sqrt(min(1.0, a)))


def dist_matrix(coords: Sequence[Coord], detour_factor: float = 1.0) -> np.ndarray:
    """
    Symmetric great-circle distance matrix in metres.

    ``detour_factor`` scales straight-line distance into on-street distance.  A
    value of 1.0 keeps the pure geodesic used in the original manuscript; 1.3 is
    the usual circuity correction for a dense urban grid, and every routing
    comparison in this project applies the *same* factor to every policy, so the
    relative results are unaffected by the choice.
    """
    coords = list(coords)
    n = len(coords)
    m = np.zeros((n, n), dtype=float)
    for i in range(n):
        lat_i, lon_i = coords[i]
        for j in range(i + 1, n):
            d = haversine(lat_i, lon_i, coords[j][0], coords[j][1]) * detour_factor
            m[i, j] = d
            m[j, i] = d
    return m


def path_distance(order: Sequence[int], matrix: np.ndarray, depot: int,
                  closed: bool = True) -> float:
    """
    Metric length of visiting ``order`` starting at ``depot``.

    ``closed=True`` adds the mandatory return leg to the depot, which the fleet
    planner always requires (a truck must return to tip its load).
    """
    order = list(order)
    if not order:
        return 0.0
    total = matrix[depot, order[0]]
    for a, b in zip(order[:-1], order[1:]):
        total += matrix[a, b]
    if closed:
        total += matrix[order[-1], depot]
    return float(total)


def centroid(coords: Iterable[Coord]) -> Coord:
    pts = list(coords)
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def point_to_segment_m(pt: Coord, a: Coord, b: Coord) -> float:
    """
    Distance in metres from ``pt`` to the segment ``a``-``b``.

    Uses a local equirectangular projection about ``a``; over the few-kilometre
    spans involved here the error is far below the sensor and map noise.
    """
    lat0 = math.radians(a[0])
    kx = math.cos(lat0) * 111_320.0
    ky = 110_540.0

    def proj(c: Coord):
        return ((c[1] - a[1]) * kx, (c[0] - a[0]) * ky)

    px, py = proj(pt)
    bx, by = proj(b)
    seg_len2 = bx * bx + by * by
    if seg_len2 <= 1e-9:
        return math.hypot(px, py)
    t = max(0.0, min(1.0, (px * bx + py * by) / seg_len2))
    return math.hypot(px - t * bx, py - t * by)
