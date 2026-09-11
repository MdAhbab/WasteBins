"""
Street-network distances and free-flow speeds.
==============================================

Every routing number in the earlier version of this work was computed on
great-circle distance multiplied by a single circuity constant of 1.30.  That is
a defensible first approximation and it is wrong in a way that matters for a
collection round: circuity is not a constant.  It varies with the local street
pattern, it is largest for short legs between containers on opposite sides of a
divided arterial, and a planner that cannot see a one-way restriction will
happily build a route no driver can follow.

This module replaces that constant with shortest paths on the real drivable
street graph, taken from OpenStreetMap.  Three quantities come out of it:

``distance_m``
    Shortest-path road distance between every pair of stops, respecting one-way
    restrictions and turn-restricted classes.

``freeflow_kmh``
    The length-weighted mean of the free-flow speeds of the arcs the shortest
    path actually uses.  A leg that runs along a primary road therefore starts
    from a higher uncongested speed than one that threads residential lanes.
    The congestion surface in :mod:`wastebins_core.traffic` then derates that
    per-leg speed instead of derating one global constant.

``circuity``
    Road distance divided by great-circle distance, reported per pair.  This is
    the assumption the earlier version made, now measured.

Reproducibility
---------------
An Overpass query is a live request against a database that changes daily, so a
re-fetch does not reproduce an earlier run.  The graph is therefore compiled
once and cached to a compact ``.npz`` that is committed to the repository, and
every cached graph records the query, the bounding box and the fetch timestamp.
:func:`load` reads the cache and never touches the network; :func:`build` is the
one entry point that does, and it is run by hand when a study area is added.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .geo import Coord, haversine

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "roadnet"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

#: Highway classes kept as drivable by a refuse vehicle.  Footways, cycleways,
#: tracks and steps are excluded; ``service`` is kept because depot yards,
#: supermarket aprons and laneway container standings are tagged that way.
DRIVABLE = (
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
    "tertiary_link",
)

#: Free-flow speed in km/h by highway class when the way carries no ``maxspeed``
#: tag.  These are uncongested design speeds; the congestion surface applies the
#: volume-delay derate on top, so they are deliberately optimistic.
DEFAULT_SPEED_KMH = {
    "motorway": 80.0, "trunk": 60.0, "primary": 50.0, "secondary": 45.0,
    "tertiary": 40.0, "unclassified": 30.0, "residential": 25.0,
    "living_street": 15.0, "service": 15.0,
    "motorway_link": 50.0, "trunk_link": 40.0, "primary_link": 40.0,
    "secondary_link": 35.0, "tertiary_link": 30.0,
}

#: Regional scaling applied to the class defaults.  Dhaka arterials do not run
#: at European design speeds even when uncongested, and the traffic module's own
#: calibration note brackets them near 34 km/h at free flow.
REGION_SPEED_SCALE = {"dhaka": 0.68, "wyndham": 1.0}


# ---------------------------------------------------------------------------
# Overpass fetch
# ---------------------------------------------------------------------------
def _overpass_query(bbox: Tuple[float, float, float, float]) -> str:
    south, west, north, east = bbox
    classes = "|".join(DRIVABLE)
    return (
        "[out:json][timeout:300];"
        f'way({south},{west},{north},{east})["highway"~"^({classes})$"];'
        "out body geom;"
    )


def _fetch(bbox: Tuple[float, float, float, float], retries: int = 3) -> dict:
    query = _overpass_query(bbox)
    body = ("data=" + query).encode("utf-8")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                OVERPASS_URL, data=body,
                headers={"User-Agent": "wastebins-research/1.0 (academic)"})
            with urllib.request.urlopen(req, timeout=420) as fh:
                return json.loads(fh.read().decode("utf-8"))
        except Exception as exc:                       # pragma: no cover - network
            last = exc
            time.sleep(5.0 * (attempt + 1))
    raise RuntimeError(f"Overpass fetch failed after {retries} attempts: {last}")


def _parse_maxspeed(raw: Optional[str]) -> Optional[float]:
    """Turn an OSM ``maxspeed`` tag into km/h, or None if it is not a speed."""
    if not raw:
        return None
    text = str(raw).strip().lower()
    if text in ("none", "signals", "variable", "walk"):
        return 7.0 if text == "walk" else None
    mph = "mph" in text
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if not digits:
        return None
    try:
        value = float(digits)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value * 1.609344 if mph else value


def _is_oneway(tags: Dict[str, str]) -> int:
    """Return +1 forward-only, -1 reverse-only, 0 bidirectional."""
    oneway = str(tags.get("oneway", "")).strip().lower()
    if oneway in ("yes", "true", "1"):
        return 1
    if oneway in ("-1", "reverse"):
        return -1
    if oneway in ("no", "false", "0"):
        return 0
    if str(tags.get("junction", "")).lower() in ("roundabout", "circular"):
        return 1
    if tags.get("highway") in ("motorway", "motorway_link"):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Compiled graph
# ---------------------------------------------------------------------------
@dataclass
class RoadNetwork:
    """A drivable street graph reduced to its largest connected component."""

    name: str
    lat: np.ndarray            # (n,) node latitudes
    lon: np.ndarray            # (n,) node longitudes
    src: np.ndarray            # (m,) arc tail index
    dst: np.ndarray            # (m,) arc head index
    length_m: np.ndarray       # (m,) arc length in metres
    speed_kmh: np.ndarray      # (m,) arc free-flow speed
    meta: Dict[str, object]

    # -- construction ------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return int(self.lat.size)

    @property
    def n_arcs(self) -> int:
        return int(self.src.size)

    def _csr(self, weight: np.ndarray) -> csr_matrix:
        return csr_matrix((weight, (self.src, self.dst)),
                          shape=(self.n_nodes, self.n_nodes))

    # -- snapping ----------------------------------------------------------
    def snap(self, coords: Sequence[Coord]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Nearest graph node for each coordinate.

        Returns the node indices and the snap distance in metres.  A large snap
        distance means the container sits away from any drivable way, which the
        caller should surface rather than silently absorb.
        """
        coords = list(coords)
        idx = np.zeros(len(coords), dtype=int)
        snap_m = np.zeros(len(coords), dtype=float)
        # Equirectangular projection about the graph centroid is accurate well
        # below the snap tolerance over a study area of a few tens of km.
        lat0 = math.radians(float(self.lat.mean()))
        kx = math.cos(lat0) * 111_320.0
        ky = 110_540.0
        nx = self.lon * kx
        ny = self.lat * ky
        for i, (clat, clon) in enumerate(coords):
            dx = nx - clon * kx
            dy = ny - clat * ky
            j = int(np.argmin(dx * dx + dy * dy))
            idx[i] = j
            snap_m[i] = haversine(clat, clon, float(self.lat[j]), float(self.lon[j]))
        return idx, snap_m

    # -- matrices ----------------------------------------------------------
    def matrices(self, coords: Sequence[Coord]
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Shortest-path distance, free-flow speed and snap distance.

        ``distance_m[i][j]`` is the road distance from stop *i* to stop *j*
        respecting one-way arcs, so the matrix is asymmetric.  ``freeflow_kmh``
        is the length-weighted mean free-flow speed of the arcs on that path,
        obtained by solving the same shortest-path problem a second time with
        travel time as the weight: distance divided by time is exactly the
        weighted harmonic mean the physics calls for.
        """
        coords = list(coords)
        idx, snap_m = self.snap(coords)
        sources = np.unique(idx)
        pos = {int(node): k for k, node in enumerate(sources)}

        dist_all = dijkstra(self._csr(self.length_m), directed=True, indices=sources)
        time_h = self.length_m / 1000.0 / np.maximum(self.speed_kmh, 1e-6)
        time_all = dijkstra(self._csr(time_h), directed=True, indices=sources)

        n = len(coords)
        distance_m = np.zeros((n, n), dtype=float)
        # The diagonal is left as NaN rather than zero.  A leg from a stop to
        # itself has no speed, and NaN makes an accidental read fail loudly
        # instead of silently contributing a zero km/h to an average.
        freeflow = np.full((n, n), np.nan, dtype=float)
        for i in range(n):
            row_d = dist_all[pos[int(idx[i])]]
            row_t = time_all[pos[int(idx[i])]]
            for j in range(n):
                if i == j:
                    continue
                d = float(row_d[idx[j]])
                t = float(row_t[idx[j]])
                distance_m[i][j] = d
                freeflow[i][j] = (d / 1000.0) / t if t > 1e-9 and np.isfinite(t) else np.nan
        return distance_m, freeflow, snap_m

    def circuity(self, coords: Sequence[Coord],
                 distance_m: Optional[np.ndarray] = None,
                 min_straight_m: float = 100.0) -> np.ndarray:
        """
        Road distance divided by great-circle distance, per ordered pair.

        Pairs closer than ``min_straight_m`` in a straight line are returned as
        NaN.  The ratio is not meaningful there: two containers at the same
        standing on opposite sides of a one-way street are metres apart as the
        crow flies and a block apart by road, which produces a circuity in the
        tens and says nothing about the street pattern.  Those pairs are worth
        looking at on their own, which :meth:`short_pair_detour` does.
        """
        coords = list(coords)
        if distance_m is None:
            distance_m, _, _ = self.matrices(coords)
        n = len(coords)
        out = np.full((n, n), np.nan, dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                straight = haversine(coords[i][0], coords[i][1],
                                     coords[j][0], coords[j][1])
                if straight >= min_straight_m and np.isfinite(distance_m[i][j]):
                    out[i][j] = distance_m[i][j] / straight
        return out

    def short_pair_detour(self, coords: Sequence[Coord],
                          distance_m: Optional[np.ndarray] = None,
                          max_straight_m: float = 100.0) -> Dict[str, float]:
        """
        What a straight-line model gets wrong about neighbouring stops.

        Returns the count of ordered pairs within ``max_straight_m`` of each
        other, and the road distance actually required to travel between them.
        A planner working on great-circle distance treats these as free; on the
        real graph some of them are a block apart.
        """
        coords = list(coords)
        if distance_m is None:
            distance_m, _, _ = self.matrices(coords)
        straights: List[float] = []
        roads: List[float] = []
        for i in range(len(coords)):
            for j in range(len(coords)):
                if i == j:
                    continue
                s = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                if s < max_straight_m and np.isfinite(distance_m[i][j]):
                    straights.append(s)
                    roads.append(float(distance_m[i][j]))
        if not straights:
            return {"pairs": 0}
        return {
            "pairs": len(straights),
            "mean_straight_m": float(np.mean(straights)),
            "mean_road_m": float(np.mean(roads)),
            "median_road_m": float(np.median(roads)),
            "max_road_m": float(np.max(roads)),
        }

    # -- persistence -------------------------------------------------------
    def save(self, path: pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, lat=self.lat, lon=self.lon, src=self.src, dst=self.dst,
            length_m=self.length_m, speed_kmh=self.speed_kmh,
            name=np.array(self.name), meta=np.array(json.dumps(self.meta)))
        return path


def _compile(osm: dict, name: str, region: str,
             bbox: Tuple[float, float, float, float]) -> RoadNetwork:
    """Turn an Overpass ``out body geom`` response into a connected arc list."""
    scale = REGION_SPEED_SCALE.get(region, 1.0)
    key_of: Dict[Tuple[int, int], int] = {}
    lats: List[float] = []
    lons: List[float] = []

    def node_index(lat: float, lon: float) -> int:
        # Quantise to about 0.1 m so ways that share a junction share a node.
        key = (int(round(lat * 1e6)), int(round(lon * 1e6)))
        if key not in key_of:
            key_of[key] = len(lats)
            lats.append(lat)
            lons.append(lon)
        return key_of[key]

    src: List[int] = []
    dst: List[int] = []
    length: List[float] = []
    speed: List[float] = []
    n_ways = 0

    for el in osm.get("elements", []):
        if el.get("type") != "way":
            continue
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        tags = el.get("tags", {}) or {}
        klass = tags.get("highway")
        if klass not in DEFAULT_SPEED_KMH:
            continue
        n_ways += 1
        tagged = _parse_maxspeed(tags.get("maxspeed"))
        v = tagged if tagged else DEFAULT_SPEED_KMH[klass] * scale
        v = max(5.0, float(v))
        direction = _is_oneway(tags)
        prev = node_index(geom[0]["lat"], geom[0]["lon"])
        for point in geom[1:]:
            cur = node_index(point["lat"], point["lon"])
            if cur == prev:
                continue
            seg = haversine(lats[prev], lons[prev], lats[cur], lons[cur])
            if seg <= 0.0:
                prev = cur
                continue
            if direction >= 0:
                src.append(prev); dst.append(cur); length.append(seg); speed.append(v)
            if direction <= 0:
                src.append(cur); dst.append(prev); length.append(seg); speed.append(v)
            prev = cur

    lat = np.asarray(lats, dtype=float)
    lon = np.asarray(lons, dtype=float)
    src_a = np.asarray(src, dtype=int)
    dst_a = np.asarray(dst, dtype=int)
    len_a = np.asarray(length, dtype=float)
    spd_a = np.asarray(speed, dtype=float)

    # Keep the largest strongly connected component.  A stop snapped onto a
    # stub that cannot be left produces infinite matrix entries, and silently
    # dropping those entries would hide an unroutable instance.
    adj = csr_matrix((len_a, (src_a, dst_a)), shape=(lat.size, lat.size))
    n_comp, labels = connected_components(adj, directed=True, connection="strong")
    keep_label = int(np.bincount(labels).argmax())
    keep = labels == keep_label
    remap = -np.ones(lat.size, dtype=int)
    remap[keep] = np.arange(int(keep.sum()))
    arc_keep = keep[src_a] & keep[dst_a]

    meta = {
        "region": region,
        "bbox": list(bbox),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overpass_query": _overpass_query(bbox),
        "ways_parsed": n_ways,
        "nodes_raw": int(lat.size),
        "arcs_raw": int(src_a.size),
        "strong_components": int(n_comp),
        "nodes_kept": int(keep.sum()),
        "arcs_kept": int(arc_keep.sum()),
        "speed_scale": scale,
        "tagged_maxspeed_share": float(np.mean(
            [1.0 if _parse_maxspeed((e.get("tags") or {}).get("maxspeed")) else 0.0
             for e in osm.get("elements", []) if e.get("type") == "way"] or [0.0])),
    }
    return RoadNetwork(
        name=name,
        lat=lat[keep], lon=lon[keep],
        src=remap[src_a[arc_keep]], dst=remap[dst_a[arc_keep]],
        length_m=len_a[arc_keep], speed_kmh=spd_a[arc_keep],
        meta=meta)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def cache_path(name: str) -> pathlib.Path:
    return CACHE_DIR / f"{name}.npz"


def build(name: str, bbox: Tuple[float, float, float, float], region: str,
          cache: Optional[pathlib.Path] = None) -> RoadNetwork:
    """Fetch, compile and cache a study area.  Touches the network."""
    osm = _fetch(bbox)
    net = _compile(osm, name=name, region=region, bbox=bbox)
    net.save(cache or cache_path(name))
    return net


def load(name: str, cache: Optional[pathlib.Path] = None) -> RoadNetwork:
    """Read a compiled study area from the cache.  Never touches the network."""
    path = pathlib.Path(cache or cache_path(name))
    if not path.exists():
        raise FileNotFoundError(
            f"No cached road network '{name}' at {path}. "
            f"Run: python -m experiments.build_roadnets")
    z = np.load(path, allow_pickle=False)
    return RoadNetwork(
        name=str(z["name"]),
        lat=z["lat"], lon=z["lon"], src=z["src"], dst=z["dst"],
        length_m=z["length_m"], speed_kmh=z["speed_kmh"],
        meta=json.loads(str(z["meta"])))


def available() -> List[str]:
    if not CACHE_DIR.exists():
        return []
    return sorted(p.stem for p in CACHE_DIR.glob("*.npz"))


def fingerprint(net: RoadNetwork) -> str:
    """Short stable hash of the arc list, so a run can record which graph it used."""
    h = hashlib.sha256()
    for arr in (net.src, net.dst, np.round(net.length_m, 3), np.round(net.speed_kmh, 3)):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]
