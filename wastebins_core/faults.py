"""
Sensor fault taxonomy and injectors.
====================================

The original ablation tested exactly one failure mode -- a channel reporting
zero -- which is the easiest case to survive because the anomaly is obvious.
Field deployments of low-cost IoT nodes fail in considerably less convenient
ways, and both reviewers asked for those.  This module implements the full
taxonomy as *seeded, reproducible* injectors that operate identically on a
NumPy array (experiments) and on a single live reading (the service).

Taxonomy
--------
========================  ====================================================
Mode                      Physical origin
========================  ====================================================
``dropout``               Radio loss / dead node -- the sample is genuinely
                          missing (NaN), which must be distinguished from a
                          measured zero.
``zero_stuck``            ADC or supply failure reporting a hard zero.  This is
                          the only mode the first submission evaluated.
``stuck_at``              Frozen register: the last valid value repeats forever.
``drift``                 Slow electro-chemical ageing of gas sensors and
                          ultrasonic transducers; an unbounded additive ramp.
``calibration``           Gain and offset error after a bad re-calibration.
``noise_burst``           EMI or an unstable supply inflating the variance.
``bursty_loss``           Gilbert-Elliott two-state channel: real packet loss is
                          strongly autocorrelated, not Bernoulli.
``weather_correlated``    Condensation and waterlogging knock out many nodes in
                          the same area at the same time -- failures that are
                          correlated across the fleet, which is what breaks
                          naive imputation.
``poisoning``             An adversary with node credentials manipulates values
                          to divert the fleet (``amplify``) or to starve a bin
                          of service (``suppress``), constrained to stay within
                          a stealth budget so it is not trivially detectable.
========================  ====================================================

Every injector returns the corrupted signal *and* a boolean ground-truth mask of
which samples were affected, so detection can be scored with precision/recall
rather than asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

FAULT_MODES = (
    "dropout", "zero_stuck", "stuck_at", "drift", "calibration",
    "noise_burst", "bursty_loss", "weather_correlated", "poisoning",
)

# Physically admissible ranges per channel, used by injectors and detectors.
CHANNEL_RANGE: Dict[str, Tuple[float, float]] = {
    "waste_level": (0.0, 1.3),
    "gas_level": (0.0, 1.0),
    "temperature": (-10.0, 80.0),
    "humidity": (0.0, 100.0),
}


@dataclass
class FaultSpec:
    """Declarative description of one fault to inject."""

    mode: str
    rate: float = 0.1                  # probability / intensity, mode-dependent
    magnitude: float = 0.2             # additive or multiplicative strength
    start_frac: float = 0.0            # where in the series the fault begins
    duration_frac: float = 1.0         # how much of the series it covers
    # Gilbert-Elliott parameters (bursty_loss)
    p_good_to_bad: float = 0.02
    p_bad_to_good: float = 0.25
    # poisoning
    direction: str = "amplify"         # "amplify" | "suppress"
    stealth: float = 0.25              # max perturbation as a fraction of range
    targets: Optional[Sequence[int]] = None   # affected node indices
    seed: int = 0
    extra: Dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.mode not in FAULT_MODES:
            raise ValueError(f"unknown fault mode {self.mode!r}; expected one of {FAULT_MODES}")
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("rate must lie in [0, 1]")
        if self.direction not in ("amplify", "suppress"):
            raise ValueError("direction must be 'amplify' or 'suppress'")


# ---------------------------------------------------------------------------
# Window helper
# ---------------------------------------------------------------------------
def _window(n: int, spec: FaultSpec) -> np.ndarray:
    start = int(np.clip(spec.start_frac, 0.0, 1.0) * n)
    length = int(np.clip(spec.duration_frac, 0.0, 1.0) * n)
    mask = np.zeros(n, dtype=bool)
    mask[start:min(n, start + max(length, 1))] = True
    return mask


# ---------------------------------------------------------------------------
# Individual injectors -- each returns (corrupted, affected_mask)
# ---------------------------------------------------------------------------
def inject_dropout(x: np.ndarray, spec: FaultSpec, rng) -> Tuple[np.ndarray, np.ndarray]:
    out = x.astype(float).copy()
    win = _window(len(x), spec)
    hit = win & (rng.random(len(x)) < spec.rate)
    out[hit] = np.nan
    return out, hit


def inject_zero_stuck(x: np.ndarray, spec: FaultSpec, rng) -> Tuple[np.ndarray, np.ndarray]:
    out = x.astype(float).copy()
    win = _window(len(x), spec)
    hit = win & (rng.random(len(x)) < spec.rate)
    out[hit] = 0.0
    return out, hit


def inject_stuck_at(x: np.ndarray, spec: FaultSpec, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Freeze at the value held when the fault begins."""
    out = x.astype(float).copy()
    n = len(x)
    win = _window(n, spec)
    idx = np.flatnonzero(win)
    if idx.size == 0:
        return out, np.zeros(n, dtype=bool)
    start = idx[0]
    held = out[start - 1] if start > 0 else out[start]
    hit = np.zeros(n, dtype=bool)
    # A frozen register stays frozen for the whole window once triggered.
    if rng.random() < max(spec.rate, 1e-9) or spec.rate >= 1.0:
        out[idx] = held
        hit[idx] = True
    return out, hit


def inject_drift(x: np.ndarray, spec: FaultSpec, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Unbounded additive ramp -- the classic ageing gas-sensor failure."""
    out = x.astype(float).copy()
    n = len(x)
    win = _window(n, spec)
    idx = np.flatnonzero(win)
    if idx.size == 0:
        return out, np.zeros(n, dtype=bool)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    shape = str(spec.extra.get("shape", "linear"))
    t = np.linspace(0.0, 1.0, idx.size)
    ramp = t if shape == "linear" else (np.expm1(2.0 * t) / np.expm1(2.0))
    out[idx] = out[idx] + sign * spec.magnitude * ramp
    hit = np.zeros(n, dtype=bool)
    hit[idx] = True
    return out, hit


def inject_calibration(x: np.ndarray, spec: FaultSpec, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Gain and offset error introduced by a bad re-calibration."""
    out = x.astype(float).copy()
    n = len(x)
    win = _window(n, spec)
    gain = float(spec.extra.get("gain", 1.0 + spec.magnitude))
    offset = float(spec.extra.get("offset", spec.magnitude * 0.5))
    out[win] = out[win] * gain + offset
    return out, win


def inject_noise_burst(x: np.ndarray, spec: FaultSpec, rng) -> Tuple[np.ndarray, np.ndarray]:
    out = x.astype(float).copy()
    n = len(x)
    win = _window(n, spec)
    hit = win & (rng.random(n) < spec.rate)
    out[hit] = out[hit] + rng.normal(0.0, spec.magnitude, int(hit.sum()))
    return out, hit


def inject_bursty_loss(x: np.ndarray, spec: FaultSpec, rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gilbert-Elliott channel.

    A two-state Markov chain: in the GOOD state packets arrive, in the BAD state
    they are lost.  Real radio loss is bursty, so an i.i.d. dropout model
    materially understates how long a bin can go unobserved.
    """
    out = x.astype(float).copy()
    n = len(x)
    win = _window(n, spec)
    bad = False
    hit = np.zeros(n, dtype=bool)
    for i in range(n):
        if not win[i]:
            bad = False
            continue
        if bad:
            if rng.random() < spec.p_bad_to_good:
                bad = False
        else:
            if rng.random() < spec.p_good_to_bad:
                bad = True
        if bad:
            out[i] = np.nan
            hit[i] = True
    return out, hit


def inject_weather_correlated(x: np.ndarray, spec: FaultSpec, rng,
                              weather: Optional[np.ndarray] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Failure probability driven by a shared weather series.

    Because ``weather`` is the *same* array for every node in a fleet-level
    injection, the resulting failures are correlated across nodes -- the case
    that defeats "borrow the value from a neighbour" imputation.
    """
    out = x.astype(float).copy()
    n = len(x)
    win = _window(n, spec)
    if weather is None:
        # deterministic fallback: a few multi-hour wet spells
        t = np.arange(n)
        weather = (0.5 * (1 + np.sin(2 * np.pi * t / max(n / 6.0, 1.0)))) ** 3
    w = np.clip(np.asarray(weather, dtype=float)[:n], 0.0, 1.0)
    p = np.clip(spec.rate * (0.15 + 2.2 * w), 0.0, 1.0)
    hit = win & (rng.random(n) < p)
    # condensation first biases the reading, then kills the channel outright
    heavy = hit & (w > 0.6)
    light = hit & ~heavy
    out[light] = out[light] + rng.normal(spec.magnitude, spec.magnitude * 0.5, int(light.sum()))
    out[heavy] = np.nan
    return out, hit


def inject_poisoning(x: np.ndarray, spec: FaultSpec, rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adversarial manipulation by a compromised node.

    ``amplify``  inflates the reading so the fleet is diverted to a bin that does
                 not need service (a denial-of-service on the routing budget).
    ``suppress`` deflates it so a genuinely full or hazardous bin is never
                 dispatched (a targeted starvation attack).

    The perturbation is bounded by ``stealth`` as a fraction of the channel range
    and is applied smoothly, so it does not announce itself as an obvious
    out-of-range spike -- which is precisely what makes it a harder detection
    problem than a dead sensor.
    """
    out = x.astype(float).copy()
    n = len(x)
    win = _window(n, spec)
    lo, hi = CHANNEL_RANGE.get(str(spec.extra.get("channel", "waste_level")), (0.0, 1.0))
    span = hi - lo
    budget = spec.stealth * span
    hit = win & (rng.random(n) < max(spec.rate, 0.0))
    if not hit.any():
        return out, hit
    # smooth, slowly-varying perturbation rather than i.i.d. jitter
    k = np.flatnonzero(hit)
    phase = rng.uniform(0.0, 2 * np.pi)
    shape = 0.55 + 0.45 * np.sin(np.linspace(0, 2 * np.pi, k.size) + phase)
    delta = budget * shape
    if spec.direction == "suppress":
        delta = -delta
    out[k] = np.clip(out[k] + delta, lo, hi)
    return out, hit


_INJECTORS = {
    "dropout": inject_dropout,
    "zero_stuck": inject_zero_stuck,
    "stuck_at": inject_stuck_at,
    "drift": inject_drift,
    "calibration": inject_calibration,
    "noise_burst": inject_noise_burst,
    "bursty_loss": inject_bursty_loss,
    "weather_correlated": inject_weather_correlated,
    "poisoning": inject_poisoning,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def inject(series, spec: FaultSpec, weather=None):
    """
    Apply ``spec`` to a 1-D signal.

    Returns ``(corrupted, affected_mask)`` where ``affected_mask`` is the
    ground-truth label used to score fault detection.
    """
    spec.validate()
    x = np.asarray(series, dtype=float)
    rng = np.random.default_rng(spec.seed)
    fn = _INJECTORS[spec.mode]
    if spec.mode == "weather_correlated":
        return fn(x, spec, rng, weather=weather)
    return fn(x, spec, rng)


def inject_fleet(channel_series: Dict[int, np.ndarray], spec: FaultSpec,
                 weather=None) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Apply ``spec`` across a fleet of nodes.

    ``spec.targets`` selects which node ids are attacked/faulty; ``None`` means
    every node is eligible.  Each node gets its own derived seed, but a
    weather-correlated fault shares one weather series, which is what produces
    genuinely correlated fleet-wide failures.
    """
    spec.validate()
    corrupted, masks = {}, {}
    eligible = set(spec.targets) if spec.targets is not None else set(channel_series)
    for i, (node_id, series) in enumerate(sorted(channel_series.items())):
        if node_id not in eligible:
            corrupted[node_id] = np.asarray(series, dtype=float).copy()
            masks[node_id] = np.zeros(len(series), dtype=bool)
            continue
        node_spec = FaultSpec(**{**spec.__dict__, "seed": spec.seed * 7919 + i})
        corrupted[node_id], masks[node_id] = inject(series, node_spec, weather=weather)
    return corrupted, masks


def describe_taxonomy() -> Dict[str, str]:
    """Human-readable catalogue, surfaced by the API and used in the manuscript."""
    return {
        "dropout": "Independent packet loss; the sample is missing (NaN), not zero.",
        "zero_stuck": "Hard zero from an ADC or supply failure (the only mode evaluated originally).",
        "stuck_at": "Frozen register repeating the last valid value.",
        "drift": "Unbounded additive ramp from electro-chemical sensor ageing.",
        "calibration": "Gain and offset error following a bad re-calibration.",
        "noise_burst": "Variance inflation from electromagnetic interference.",
        "bursty_loss": "Gilbert-Elliott two-state channel; loss is autocorrelated, not i.i.d.",
        "weather_correlated": "Rain/condensation knocks out many nearby nodes simultaneously.",
        "poisoning": "Adversarial amplify/suppress within a stealth budget to divert or starve service.",
    }
