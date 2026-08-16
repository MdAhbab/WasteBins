"""
Runtime sensor validation and trust estimation.
===============================================

The manuscript's contribution on the sensing side is *dynamic weight
renormalisation*: when a channel is unavailable its weight is removed and the
remaining weights are rescaled, instead of substituting a zero that silently
drags the priority score down.  That mechanism is only as good as the decision
about which channels to trust, and a hard available/unavailable flag cannot
express the failure modes that matter (a drifting gas sensor still reports
plausible numbers; a poisoned node reports *deliberately* plausible numbers).

This module therefore replaces the binary flag with a continuous per-channel
**trust weight** in [0, 1] produced by an ensemble of detectors:

=========================  ===================================================
Detector                   Catches
=========================  ===================================================
``range``                  Hard physical-bound violations.
``missing``                Genuine absence (NaN), kept distinct from a zero.
``stuck``                  Frozen registers and zero-variance channels.
``spike``                  Impulsive noise, via a Hampel filter on median/MAD.
``drift``                  Slow bias, via a two-sided Page-Hinkley change test.
``cross_channel``          Physically impossible combinations between channels.
``peer``                   Fleet inconsistency: a robust z-score against the
                           other bins at the same instant, which is the only
                           detector that sees a stealthy poisoning attack whose
                           own time series looks perfectly smooth.
=========================  ===================================================

Trust is updated asymmetrically -- it falls quickly on evidence of a fault and
recovers slowly -- because the cost of trusting a broken sensor for one more
cycle greatly exceeds the cost of down-weighting a healthy one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .faults import CHANNEL_RANGE

STATUS_OK = "ok"
STATUS_SUSPECT = "suspect"
STATUS_DEGRADED = "degraded"
STATUS_FAILED = "failed"

# Trust thresholds separating the four operating states.
TRUST_SUSPECT = 0.75
TRUST_DEGRADED = 0.45
TRUST_FAILED = 0.15

# Resolution below which two consecutive samples count as identical.
CHANNEL_RESOLUTION = {
    "waste_level": 1e-4,
    "gas_level": 1e-4,
    "temperature": 1e-3,
    "humidity": 1e-2,
}

# Expected short-horizon standard deviation, used to scale the drift test.
CHANNEL_SIGMA = {
    "waste_level": 0.03,
    "gas_level": 0.04,
    "temperature": 0.60,
    "humidity": 2.50,
}


@dataclass
class ChannelAssessment:
    """Health verdict for one channel of one node."""

    channel: str
    status: str = STATUS_OK
    trust: float = 1.0
    value: Optional[float] = None            # value to use downstream (None = drop)
    raw_value: Optional[float] = None
    drift_estimate: float = 0.0
    stuck_streak: int = 0
    missing_streak: int = 0
    scores: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "channel": self.channel,
            "status": self.status,
            "trust": round(self.trust, 4),
            "value": None if self.value is None else round(float(self.value), 5),
            "raw_value": None if self.raw_value is None else round(float(self.raw_value), 5),
            "drift_estimate": round(self.drift_estimate, 5),
            "stuck_streak": self.stuck_streak,
            "missing_streak": self.missing_streak,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "flags": list(self.flags),
        }


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------
def median_abs_deviation(x: np.ndarray) -> float:
    """MAD scaled to be a consistent estimator of sigma for Gaussian data."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(1.4826 * np.median(np.abs(x - med)))


def robust_z(value: float, reference: Sequence[float]) -> float:
    """Robust z-score of ``value`` against a reference sample."""
    ref = np.asarray(list(reference), dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size < 3:
        return 0.0
    med = float(np.median(ref))
    mad = median_abs_deviation(ref)
    if mad < 1e-9:
        spread = float(np.std(ref))
        if spread < 1e-9:
            return 0.0
        mad = spread
    return float((value - med) / mad)


def page_hinkley(x: np.ndarray, delta: float, threshold_sigma: float = 4.0,
                 sigma: float = 1.0):
    """
    Two-sided Page-Hinkley change detector.

    Returns ``(detected, magnitude)`` where ``magnitude`` is the signed estimate
    of the mean shift in the units of ``x``.  ``delta`` is the tolerated drift
    per sample (set to a fraction of the channel's natural noise), and the
    alarm threshold is expressed in multiples of ``sigma`` so the same
    parameters transfer across channels with very different scales.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return False, 0.0

    mean = float(x[0])
    m_pos = m_neg = 0.0
    min_pos = max_neg = 0.0
    threshold = threshold_sigma * max(sigma, 1e-9)
    detected = False
    for i in range(1, n):
        mean += (x[i] - mean) / (i + 1)
        m_pos += x[i] - mean - delta
        m_neg += x[i] - mean + delta
        min_pos = min(min_pos, m_pos)
        max_neg = max(max_neg, m_neg)
        if (m_pos - min_pos) > threshold or (max_neg - m_neg) > threshold:
            detected = True

    # Magnitude: difference between the recent and the early mean.
    half = max(2, n // 3)
    magnitude = float(np.mean(x[-half:]) - np.mean(x[:half]))
    return detected, magnitude


# ---------------------------------------------------------------------------
# Individual detectors -- each returns an anomaly score in [0, 1]
# ---------------------------------------------------------------------------
def _score_range(value: Optional[float], channel: str) -> float:
    if value is None or not math.isfinite(value):
        return 0.0                                   # handled by the missing detector
    lo, hi = CHANNEL_RANGE.get(channel, (-math.inf, math.inf))
    if lo <= value <= hi:
        return 0.0
    span = max(hi - lo, 1e-6)
    excess = (lo - value) if value < lo else (value - hi)
    return float(min(1.0, 0.6 + 0.4 * min(1.0, excess / span)))


def _score_missing(missing_streak: int, patience: int = 3) -> float:
    if missing_streak <= 0:
        return 0.0
    return float(min(1.0, missing_streak / max(patience, 1)))


def _score_stuck(values: np.ndarray, channel: str, patience: int = 6) -> (float, int):
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 0.0, 0
    eps = CHANNEL_RESOLUTION.get(channel, 1e-6)
    streak = 1
    for a, b in zip(finite[::-1][:-1], finite[::-1][1:]):
        if abs(a - b) <= eps:
            streak += 1
        else:
            break
    if streak < 2:
        return 0.0, 0
    # A waste level legitimately plateaus; gas and temperature should not.
    tolerance = patience * (2 if channel == "waste_level" else 1)
    return float(min(1.0, max(0.0, (streak - 2) / max(tolerance, 1)))), int(streak)


def _score_spike(value: Optional[float], history: np.ndarray, channel: str) -> float:
    """Hampel-filter style outlier score against the node's own recent window."""
    if value is None or not math.isfinite(value):
        return 0.0
    hist = history[np.isfinite(history)]
    if hist.size < 5:
        return 0.0
    z = abs(robust_z(value, hist))
    if z <= 3.0:
        return 0.0
    return float(min(1.0, (z - 3.0) / 6.0))


def _score_drift(history: np.ndarray, channel: str) -> (float, float):
    sigma = CHANNEL_SIGMA.get(channel, 1.0)
    detected, magnitude = page_hinkley(history, delta=0.25 * sigma,
                                       threshold_sigma=4.0, sigma=sigma)
    if not detected:
        return 0.0, 0.0
    score = float(min(1.0, abs(magnitude) / (4.0 * sigma)))
    return score, float(magnitude)


def _score_cross_channel(values: Dict[str, Optional[float]]) -> Dict[str, float]:
    """
    Physically impossible combinations between simultaneously-read channels.

    Two checks are cheap and genuinely informative for a bin enclosure:

    * Decomposition is exothermic and produces gas, so a high gas reading paired
      with an internal temperature *below* ambient-minimum is inconsistent.
    * Saturated air inside a closed bin cannot sit at very low humidity while
      the contents are hot and gassy.
    """
    out = {k: 0.0 for k in values}
    gas = values.get("gas_level")
    temp = values.get("temperature")
    hum = values.get("humidity")
    waste = values.get("waste_level")

    def ok(v):
        return v is not None and math.isfinite(v)

    if ok(gas) and ok(temp) and gas > 0.55 and temp < 12.0:
        out["gas_level"] = max(out["gas_level"], 0.5)
        out["temperature"] = max(out["temperature"], 0.5)
    if ok(gas) and ok(hum) and gas > 0.6 and hum < 25.0:
        out["humidity"] = max(out["humidity"], 0.4)
    if ok(gas) and ok(waste) and gas > 0.7 and waste < 0.05:
        # Strong odour from an essentially empty bin is not physical.
        out["gas_level"] = max(out["gas_level"], 0.45)
        out["waste_level"] = max(out["waste_level"], 0.45)
    return out


def _score_peer(value: Optional[float], peers: Optional[Sequence[float]],
                channel: str) -> float:
    """
    Fleet-consistency score.

    A stealthy poisoning attack keeps a node's own time series smooth, so no
    single-node detector fires.  Comparing against the simultaneous distribution
    over the rest of the fleet is what exposes it -- provided the comparison is
    robust, since the attacker may control several nodes.
    """
    if value is None or not math.isfinite(value) or not peers:
        return 0.0
    z = abs(robust_z(value, peers))
    # Bins are heterogeneous, so tolerate more spread than a within-node test.
    if z <= 4.0:
        return 0.0
    return float(min(1.0, (z - 4.0) / 8.0))


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------
DETECTOR_WEIGHTS = {
    "range": 1.00,
    "missing": 1.00,
    "stuck": 0.85,
    "spike": 0.70,
    "drift": 0.80,
    "cross_channel": 0.60,
    "peer": 0.75,
}


def _combine(scores: Dict[str, float]) -> float:
    """
    Noisy-OR combination of weighted detector scores.

    Independent evidence should accumulate, but no single moderate score should
    saturate the result, which is what a plain max would do.
    """
    prob_clean = 1.0
    for name, score in scores.items():
        w = DETECTOR_WEIGHTS.get(name, 0.5)
        prob_clean *= (1.0 - max(0.0, min(1.0, score)) * w)
    return float(1.0 - prob_clean)


def status_for_trust(trust: float) -> str:
    if trust >= TRUST_SUSPECT:
        return STATUS_OK
    if trust >= TRUST_DEGRADED:
        return STATUS_SUSPECT
    if trust >= TRUST_FAILED:
        return STATUS_DEGRADED
    return STATUS_FAILED


def update_trust(previous: float, anomaly: float,
                 drop_rate: float = 0.60, recover_rate: float = 0.08) -> float:
    """
    Asymmetric exponential trust update.

    Trust collapses within a couple of cycles once a fault is evident but takes
    roughly a dozen clean cycles to be restored, matching the operational rule
    that a suspect sensor must prove itself again before it is believed.
    """
    evidence = 1.0 - max(0.0, min(1.0, anomaly))
    rate = drop_rate if evidence < previous else recover_rate
    return float(max(0.0, min(1.0, previous + rate * (evidence - previous))))


def assess_node(history: Dict[str, Sequence[float]],
                previous_trust: Optional[Dict[str, float]] = None,
                peers: Optional[Dict[str, Sequence[float]]] = None,
                correct_drift: bool = True) -> Dict[str, ChannelAssessment]:
    """
    Assess every channel of one node.

    Parameters
    ----------
    history
        ``{channel: [oldest, ..., newest]}``.  The final element is the current
        sample; ``NaN`` marks a genuinely missing reading.
    previous_trust
        Trust carried over from the last cycle, so the state is recursive rather
        than recomputed from scratch.
    peers
        ``{channel: [values from other nodes at this instant]}`` for the
        fleet-consistency detector.
    correct_drift
        When a drift is detected with a confident magnitude estimate, subtract it
        from the value handed downstream instead of discarding the channel.

    Returns
    -------
    ``{channel: ChannelAssessment}``.  Consumers should use ``value`` (which is
    ``None`` when the channel must be dropped) together with ``trust`` as the
    renormalisation weight.
    """
    previous_trust = previous_trust or {}
    peers = peers or {}
    out: Dict[str, ChannelAssessment] = {}

    current: Dict[str, Optional[float]] = {}
    arrays: Dict[str, np.ndarray] = {}
    for channel, series in history.items():
        arr = np.asarray(list(series), dtype=float)
        arrays[channel] = arr
        last = arr[-1] if arr.size else np.nan
        current[channel] = float(last) if np.isfinite(last) else None

    cross = _score_cross_channel(current)

    for channel, arr in arrays.items():
        value = current[channel]
        past = arr[:-1] if arr.size > 1 else np.asarray([], dtype=float)

        missing_streak = 0
        for v in arr[::-1]:
            if np.isfinite(v):
                break
            missing_streak += 1

        stuck_score, stuck_streak = _score_stuck(arr, channel)
        drift_score, drift_magnitude = _score_drift(arr, channel)

        scores = {
            "range": _score_range(value, channel),
            "missing": _score_missing(missing_streak),
            "stuck": stuck_score,
            "spike": _score_spike(value, past, channel),
            "drift": drift_score,
            "cross_channel": cross.get(channel, 0.0),
            "peer": _score_peer(value, peers.get(channel), channel),
        }

        anomaly = _combine(scores)
        trust = update_trust(float(previous_trust.get(channel, 1.0)), anomaly)
        status = status_for_trust(trust)

        flags = [name for name, s in scores.items() if s > 0.15]

        # --- decide what value, if any, to pass downstream ---------------
        usable: Optional[float] = value
        if value is None:
            usable = None
        elif status == STATUS_FAILED:
            usable = None                     # drop entirely; weights renormalise
        elif scores["range"] > 0.0:
            lo, hi = CHANNEL_RANGE.get(channel, (-math.inf, math.inf))
            usable = float(min(max(value, lo), hi))
            flags.append("clamped")
        if usable is not None and correct_drift and drift_score > 0.35:
            usable = float(usable - drift_magnitude)
            lo, hi = CHANNEL_RANGE.get(channel, (-math.inf, math.inf))
            usable = float(min(max(usable, lo), hi))
            flags.append("drift_corrected")

        out[channel] = ChannelAssessment(
            channel=channel,
            status=status,
            trust=trust,
            value=usable,
            raw_value=value,
            drift_estimate=drift_magnitude if drift_score > 0.0 else 0.0,
            stuck_streak=stuck_streak,
            missing_streak=missing_streak,
            scores=scores,
            flags=sorted(set(flags)),
        )
    return out


def fleet_peers(node_histories: Dict[int, Dict[str, Sequence[float]]],
                exclude: Optional[int] = None) -> Dict[str, List[float]]:
    """Collect the latest value of each channel across the fleet for peer scoring."""
    peers: Dict[str, List[float]] = {}
    for node_id, hist in node_histories.items():
        if exclude is not None and node_id == exclude:
            continue
        for channel, series in hist.items():
            arr = np.asarray(list(series), dtype=float)
            if arr.size and np.isfinite(arr[-1]):
                peers.setdefault(channel, []).append(float(arr[-1]))
    return peers


def assess_fleet(node_histories: Dict[int, Dict[str, Sequence[float]]],
                 previous_trust: Optional[Dict[int, Dict[str, float]]] = None,
                 correct_drift: bool = True
                 ) -> Dict[int, Dict[str, ChannelAssessment]]:
    """Assess every node, using leave-one-out fleet peers for the consistency test."""
    previous_trust = previous_trust or {}
    results = {}
    for node_id, hist in node_histories.items():
        peers = fleet_peers(node_histories, exclude=node_id)
        results[node_id] = assess_node(
            hist,
            previous_trust=previous_trust.get(node_id),
            peers=peers,
            correct_drift=correct_drift,
        )
    return results


def detection_metrics(assessments: Dict[int, Dict[str, ChannelAssessment]],
                      truth: Dict[int, Dict[str, bool]]) -> Dict[str, float]:
    """
    Precision/recall/F1 of fault detection against injected ground truth.

    A channel counts as *detected* when its status is anything other than ``ok``.
    """
    tp = fp = fn = tn = 0
    for node_id, channels in assessments.items():
        for channel, assessment in channels.items():
            flagged = assessment.status != STATUS_OK
            actual = bool(truth.get(node_id, {}).get(channel, False))
            if flagged and actual:
                tp += 1
            elif flagged and not actual:
                fp += 1
            elif not flagged and actual:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
    }
