"""
Contextual traffic friction.
============================

The first submission declared a traffic multiplier but left it as an unpopulated
interface.  This module makes it a working, testable component with two
interchangeable providers behind one API:

``SyntheticTrafficProvider``
    A deterministic, fully reproducible congestion surface built from four
    physically meaningful terms:

      1. *Temporal profile* -- bimodal weekday commute peaks plus a Dhaka working
         week (Friday/Saturday weekend), as a smooth mixture of von-Mises-like
         bumps over the 24 h clock.
      2. *Spatial profile* -- proximity to named arterial corridors.  A bin that
         sits on an arterial inherits far more congestion than one on a service
         lane, so congestion is spatially structured rather than i.i.d. noise.
      3. *Incidents* -- a seeded Poisson process of localised, time-limited
         incidents (breakdowns, waterlogging, processions) with a spatial decay
         kernel.  Seeded means reproducible: the same seed and clock always give
         the same surface.
      4. *Weather* -- rainfall raises friction network-wide, and is the same
         driver used by the weather-correlated sensor-failure mode, so the two
         subsystems degrade together exactly as they do in the field.

``LiveTrafficProvider``
    A working HTTP adapter for a real-time flow feed.  It speaks the segment
    schema published by the common commercial providers
    (``{"flowSegmentData": {"currentSpeed": .., "freeFlowSpeed": ..}}``) as well
    as a plain ``{"segments": [{"lat", "lng", "speed_kmh", "freeflow_kmh"}]}``
    document, caches responses, and degrades to the synthetic surface when the
    feed is unreachable -- so a deployment never loses its planner because an
    upstream API is down.

Friction and speed are related by the Bureau of Public Roads volume-delay
function (BPR, 1964), with the exponent/coefficient pair calibrated for
saturated urban arterials rather than the original freeway values:

    t(x) = t_free * (1 + a * x**b),        v(x) = v_free / (1 + a * x**b)

where ``x`` in [0, 1] is the friction (saturation) level.  With the defaults
below a free-flow 34 km/h arterial degrades to roughly 10 km/h at full
saturation, which brackets published Dhaka arterial speeds.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone as dt_timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .geo import Coord, point_to_segment_m

# ---------------------------------------------------------------------------
# BPR calibration
# ---------------------------------------------------------------------------
BPR_A = 2.40          # congestion sensitivity (urban arterial, saturated flow)
BPR_B = 3.50          # convexity
FREEFLOW_KMH = 34.0
MIN_SPEED_KMH = 5.0


def bpr_speed_kmh(friction: float, freeflow_kmh: float = FREEFLOW_KMH,
                  a: float = BPR_A, b: float = BPR_B,
                  min_kmh: float = MIN_SPEED_KMH) -> float:
    """Operating speed for a saturation level ``friction`` in [0, 1]."""
    x = max(0.0, min(1.0, float(friction)))
    speed = freeflow_kmh / (1.0 + a * (x ** b))
    return max(min_kmh, speed)


def speed_to_friction(speed_kmh: float, freeflow_kmh: float = FREEFLOW_KMH,
                      a: float = BPR_A, b: float = BPR_B) -> float:
    """Inverse of :func:`bpr_speed_kmh`; used to ingest live speed feeds."""
    speed = max(1e-6, float(speed_kmh))
    ratio = max(0.0, (freeflow_kmh / speed) - 1.0) / a
    return max(0.0, min(1.0, ratio ** (1.0 / b)))


# ---------------------------------------------------------------------------
# Arterial corridors around Mirpur, Dhaka (the study area).
# Each corridor is a polyline plus a base saturation weight and an influence
# radius in metres.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Corridor:
    name: str
    points: Tuple[Coord, ...]
    weight: float            # peak saturation contributed on the corridor
    radius_m: float          # influence half-width


MIRPUR_CORRIDORS: Tuple[Corridor, ...] = (
    Corridor("Mirpur Road (arterial)",
             ((23.7790, 90.3660), (23.7930, 90.3648), (23.8069, 90.3687),
              (23.8203, 90.3650), (23.8255, 90.3651)),
             weight=0.92, radius_m=420.0),
    Corridor("Rokeya Sarani",
             ((23.7770, 90.3775), (23.7890, 90.3740), (23.7980, 90.3720),
              (23.8100, 90.3700), (23.8210, 90.3672)),
             weight=0.85, radius_m=380.0),
    Corridor("Begum Rokeya / Kazipara link",
             ((23.7900, 90.3850), (23.7960, 90.3790), (23.8050, 90.3730)),
             weight=0.66, radius_m=300.0),
    Corridor("Mirpur 10 - Mirpur 14 connector",
             ((23.8069, 90.3687), (23.8090, 90.3735), (23.8100, 90.3780)),
             weight=0.72, radius_m=320.0),
    Corridor("Agargaon approach",
             ((23.7780, 90.3800), (23.7840, 90.3790), (23.7900, 90.3770)),
             weight=0.58, radius_m=280.0),
)


# ---------------------------------------------------------------------------
# Temporal profile
# ---------------------------------------------------------------------------
def temporal_multiplier(hour_of_day: float, day_of_week: int) -> float:
    """
    Congestion multiplier in roughly [0.12, 1.0] for a clock position.

    ``day_of_week`` follows Python's convention (Monday = 0 ... Sunday = 6).
    Friday (4) and Saturday (5) are the Bangladeshi weekend, so the commute
    peaks flatten and a late-morning shopping peak takes over.
    """
    h = float(hour_of_day) % 24.0

    def bump(centre: float, width: float, height: float) -> float:
        # circular Gaussian so 23:30 and 00:30 are neighbours
        d = abs(h - centre)
        d = min(d, 24.0 - d)
        return height * math.exp(-(d ** 2) / (2.0 * width ** 2))

    weekend = day_of_week in (4, 5)
    if weekend:
        base = 0.20
        peaks = bump(11.5, 2.4, 0.46) + bump(18.0, 2.2, 0.55) + bump(21.0, 1.6, 0.30)
    else:
        base = 0.16
        peaks = bump(8.8, 1.5, 0.78) + bump(13.0, 1.8, 0.34) + bump(18.2, 1.9, 0.86)

    # deep overnight trough
    night = bump(3.0, 2.2, -0.12)
    return max(0.05, min(1.0, base + peaks + night))


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class TrafficProvider:
    """Interface every provider implements."""

    name = "base"

    def friction(self, lat: float, lng: float, when: datetime) -> float:
        raise NotImplementedError

    def segment_friction(self, a: Coord, b: Coord, when: datetime) -> float:
        """Friction experienced travelling from ``a`` to ``b`` (midpoint sample)."""
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        return self.friction(mid[0], mid[1], when)

    def speed_kmh(self, a: Coord, b: Coord, when: datetime,
                  freeflow_kmh: float = FREEFLOW_KMH) -> float:
        return bpr_speed_kmh(self.segment_friction(a, b, when), freeflow_kmh)

    def describe(self) -> Dict:
        return {"provider": self.name}


@dataclass
class Incident:
    lat: float
    lng: float
    start_h: float          # absolute hours since the epoch used by the provider
    duration_h: float
    severity: float         # additional saturation at the epicentre
    radius_m: float
    label: str = "incident"

    def influence(self, lat: float, lng: float, abs_hour: float) -> float:
        if not (self.start_h <= abs_hour <= self.start_h + self.duration_h):
            return 0.0
        # temporal envelope: ramp up, plateau, ramp down
        phase = (abs_hour - self.start_h) / max(self.duration_h, 1e-6)
        envelope = math.sin(math.pi * min(1.0, max(0.0, phase))) ** 0.5
        d = point_to_segment_m((lat, lng), (self.lat, self.lng), (self.lat, self.lng))
        spatial = math.exp(-(d ** 2) / (2.0 * self.radius_m ** 2))
        return self.severity * envelope * spatial


class SyntheticTrafficProvider(TrafficProvider):
    """Deterministic, reproducible congestion surface."""

    name = "synthetic"

    def __init__(self,
                 corridors: Sequence[Corridor] = MIRPUR_CORRIDORS,
                 seed: int = 42,
                 incident_rate_per_day: float = 2.5,
                 rain_provider=None,
                 freeflow_kmh: float = FREEFLOW_KMH):
        self.corridors = tuple(corridors)
        self.seed = int(seed)
        self.incident_rate_per_day = float(incident_rate_per_day)
        self.freeflow_kmh = float(freeflow_kmh)
        # ``rain_provider(when) -> intensity in [0,1]``; shared with the sensor
        # fault model so weather degrades both subsystems coherently.
        self._rain_provider = rain_provider
        self._incident_cache: Dict[int, List[Incident]] = {}
        self._lock = threading.Lock()

    # -- spatial ---------------------------------------------------------
    def _corridor_saturation(self, lat: float, lng: float) -> float:
        best = 0.0
        for c in self.corridors:
            dmin = float("inf")
            for p, q in zip(c.points[:-1], c.points[1:]):
                dmin = min(dmin, point_to_segment_m((lat, lng), p, q))
            if dmin == float("inf"):
                continue
            # Gaussian falloff away from the corridor centre line
            contribution = c.weight * math.exp(-(dmin ** 2) / (2.0 * c.radius_m ** 2))
            best = max(best, contribution)
        # a floor of local-street friction everywhere
        return max(0.12, best)

    # -- incidents -------------------------------------------------------
    def _day_incidents(self, day_index: int) -> List[Incident]:
        with self._lock:
            cached = self._incident_cache.get(day_index)
            if cached is not None:
                return cached
        # Deterministic per-day RNG: identical output for identical (seed, day).
        import numpy as np

        rng = np.random.default_rng((self.seed * 1_000_003 + day_index) & 0xFFFFFFFF)
        n = int(rng.poisson(self.incident_rate_per_day))
        incidents: List[Incident] = []
        for _ in range(n):
            corridor = self.corridors[int(rng.integers(0, len(self.corridors)))]
            pt = corridor.points[int(rng.integers(0, len(corridor.points)))]
            jitter = 0.004
            incidents.append(Incident(
                lat=float(pt[0] + rng.uniform(-jitter, jitter)),
                lng=float(pt[1] + rng.uniform(-jitter, jitter)),
                start_h=day_index * 24.0 + float(rng.uniform(6.0, 21.0)),
                duration_h=float(rng.uniform(0.4, 2.6)),
                severity=float(rng.uniform(0.15, 0.55)),
                radius_m=float(rng.uniform(250.0, 900.0)),
                label=str(rng.choice(["breakdown", "waterlogging", "roadworks", "procession"])),
            ))
        with self._lock:
            self._incident_cache[day_index] = incidents
            if len(self._incident_cache) > 512:      # bound memory in long runs
                for k in sorted(self._incident_cache)[:256]:
                    self._incident_cache.pop(k, None)
        return incidents

    # -- weather ---------------------------------------------------------
    def rain_intensity(self, when: datetime) -> float:
        if self._rain_provider is not None:
            return float(self._rain_provider(when))
        import numpy as np

        day_index = _day_index(when)
        rng = np.random.default_rng((self.seed * 7_919 + day_index) & 0xFFFFFFFF)
        # Dhaka: pronounced monsoon season, so rain probability is seasonal.
        month = when.month
        monsoon = 1.0 if month in (5, 6, 7, 8, 9) else 0.28
        if rng.random() > 0.34 * monsoon:
            return 0.0
        peak_hour = float(rng.uniform(0.0, 24.0))
        width = float(rng.uniform(0.8, 3.0))
        h = when.hour + when.minute / 60.0
        d = abs(h - peak_hour)
        d = min(d, 24.0 - d)
        return float(min(1.0, rng.uniform(0.3, 1.0) * math.exp(-(d ** 2) / (2 * width ** 2))))

    # -- public ----------------------------------------------------------
    def friction(self, lat: float, lng: float, when: datetime) -> float:
        when = _as_aware(when)
        hour = when.hour + when.minute / 60.0 + when.second / 3600.0
        temporal = temporal_multiplier(hour, when.weekday())
        spatial = self._corridor_saturation(lat, lng)

        base = spatial * temporal

        day_index = _day_index(when)
        abs_hour = day_index * 24.0 + hour
        incident = 0.0
        for inc in self._day_incidents(day_index):
            incident = max(incident, inc.influence(lat, lng, abs_hour))

        rain = self.rain_intensity(when)
        weather = 0.28 * rain              # rain slows the whole network

        return max(0.0, min(1.0, base + incident + weather))

    def active_incidents(self, when: datetime) -> List[Dict]:
        when = _as_aware(when)
        day_index = _day_index(when)
        abs_hour = day_index * 24.0 + when.hour + when.minute / 60.0
        out = []
        for inc in self._day_incidents(day_index):
            if inc.start_h <= abs_hour <= inc.start_h + inc.duration_h:
                out.append({
                    "label": inc.label, "lat": inc.lat, "lng": inc.lng,
                    "severity": round(inc.severity, 3),
                    "radius_m": round(inc.radius_m, 1),
                    "ends_in_h": round(inc.start_h + inc.duration_h - abs_hour, 2),
                })
        return out

    def describe(self) -> Dict:
        return {
            "provider": self.name,
            "seed": self.seed,
            "corridors": [c.name for c in self.corridors],
            "incident_rate_per_day": self.incident_rate_per_day,
            "bpr": {"a": BPR_A, "b": BPR_B, "freeflow_kmh": self.freeflow_kmh},
        }


class LiveTrafficProvider(TrafficProvider):
    """
    Real-time flow adapter with caching and automatic fallback.

    ``url_template`` may contain ``{lat}``, ``{lng}`` and ``{key}`` placeholders.
    Two response shapes are understood out of the box:

    * ``{"flowSegmentData": {"currentSpeed": 21, "freeFlowSpeed": 40}}``
    * ``{"segments": [{"lat": .., "lng": .., "speed_kmh": .., "freeflow_kmh": ..}]}``

    Anything else can be handled by passing a ``parser`` callable.
    """

    name = "live"

    def __init__(self, url_template: str, api_key: str = "",
                 fallback: Optional[TrafficProvider] = None,
                 cache_seconds: float = 120.0, timeout_seconds: float = 3.0,
                 freeflow_kmh: float = FREEFLOW_KMH, parser=None,
                 grid_deg: float = 0.004):
        self.url_template = url_template
        self.api_key = api_key
        self.fallback = fallback or SyntheticTrafficProvider()
        self.cache_seconds = float(cache_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.freeflow_kmh = float(freeflow_kmh)
        self.parser = parser
        self.grid_deg = float(grid_deg)     # spatial quantisation for cache keys
        self._cache: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None
        self.fallback_count = 0
        self.hit_count = 0

    # -- parsing ---------------------------------------------------------
    def _parse(self, doc: dict) -> Optional[float]:
        if self.parser is not None:
            return self.parser(doc)
        seg = doc.get("flowSegmentData")
        if isinstance(seg, dict):
            cur = seg.get("currentSpeed")
            free = seg.get("freeFlowSpeed") or self.freeflow_kmh
            if cur is not None:
                return speed_to_friction(float(cur), float(free) or self.freeflow_kmh)
        segments = doc.get("segments")
        if isinstance(segments, list) and segments:
            fr = []
            for s in segments:
                cur = s.get("speed_kmh") or s.get("currentSpeed")
                if cur is None:
                    continue
                free = s.get("freeflow_kmh") or s.get("freeFlowSpeed") or self.freeflow_kmh
                fr.append(speed_to_friction(float(cur), float(free)))
            if fr:
                return sum(fr) / len(fr)
        if "friction" in doc:
            return max(0.0, min(1.0, float(doc["friction"])))
        return None

    def _fetch(self, lat: float, lng: float) -> Optional[float]:
        import urllib.error
        import urllib.request

        url = (self.url_template
               .replace("{lat}", f"{lat:.5f}")
               .replace("{lng}", f"{lng:.5f}")
               .replace("{key}", self.api_key))
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            if self.api_key and "{key}" not in self.url_template:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                doc = json.loads(resp.read().decode("utf-8"))
            value = self._parse(doc)
            if value is None:
                self.last_error = "unrecognised response schema"
                return None
            self.last_error = None
            return value
        except Exception as exc:                     # network, parse, or schema
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def friction(self, lat: float, lng: float, when: datetime) -> float:
        key = (int(lat / self.grid_deg), int(lng / self.grid_deg))
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
        if cached and (now - cached[0]) < self.cache_seconds:
            self.hit_count += 1
            return cached[1]

        value = self._fetch(lat, lng)
        if value is None:
            self.fallback_count += 1
            return self.fallback.friction(lat, lng, when)

        with self._lock:
            self._cache[key] = (now, value)
            if len(self._cache) > 4096:
                self._cache.clear()
        return value

    def describe(self) -> Dict:
        return {
            "provider": self.name,
            "url_template": self.url_template.replace(self.api_key, "***") if self.api_key
            else self.url_template,
            "cache_seconds": self.cache_seconds,
            "cache_hits": self.hit_count,
            "fallbacks": self.fallback_count,
            "last_error": self.last_error,
            "fallback_provider": self.fallback.name,
        }


# ---------------------------------------------------------------------------
# Travel-time matrices
# ---------------------------------------------------------------------------
@dataclass
class TravelContext:
    """Everything the planner needs to convert distance into time and emissions."""

    coords: List[Coord]
    distance_m: "object"                     # np.ndarray
    provider: TrafficProvider
    when: datetime
    freeflow_kmh: float = FREEFLOW_KMH
    _speed_cache: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def speed_kmh(self, i: int, j: int) -> float:
        key = (i, j) if i <= j else (j, i)
        cached = self._speed_cache.get(key)
        if cached is not None:
            return cached
        v = self.provider.speed_kmh(self.coords[i], self.coords[j], self.when,
                                    self.freeflow_kmh)
        self._speed_cache[key] = v
        return v

    def travel_minutes(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        km = float(self.distance_m[i][j]) / 1000.0
        return 60.0 * km / max(self.speed_kmh(i, j), 1e-6)

    def friction(self, i: int, j: int) -> float:
        return speed_to_friction(self.speed_kmh(i, j), self.freeflow_kmh)


def build_travel_context(coords: Sequence[Coord], distance_m, provider: TrafficProvider,
                         when: datetime, freeflow_kmh: float = FREEFLOW_KMH) -> TravelContext:
    return TravelContext(coords=list(coords), distance_m=distance_m,
                         provider=provider, when=_as_aware(when), freeflow_kmh=freeflow_kmh)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_EPOCH = datetime(2024, 1, 1, tzinfo=dt_timezone.utc)


def _as_aware(when: datetime) -> datetime:
    if when.tzinfo is None:
        return when.replace(tzinfo=dt_timezone.utc)
    return when


def _day_index(when: datetime) -> int:
    return int((_as_aware(when) - _EPOCH).total_seconds() // 86400)


def make_provider(config: Optional[dict] = None) -> TrafficProvider:
    """Build a provider from a plain configuration mapping (Django settings.TRAFFIC)."""
    config = config or {}
    kind = str(config.get("PROVIDER", "synthetic")).lower()
    freeflow = float(config.get("FREEFLOW_SPEED_KMH", FREEFLOW_KMH))
    synthetic = SyntheticTrafficProvider(seed=int(config.get("SEED", 42)),
                                         freeflow_kmh=freeflow)
    if kind == "live" and config.get("API_URL"):
        return LiveTrafficProvider(
            url_template=config["API_URL"],
            api_key=config.get("API_KEY", ""),
            fallback=synthetic,
            cache_seconds=float(config.get("CACHE_SECONDS", 120.0)),
            timeout_seconds=float(config.get("TIMEOUT_SECONDS", 3.0)),
            freeflow_kmh=freeflow,
        )
    return synthetic
