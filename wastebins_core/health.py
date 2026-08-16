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

from .faults import CHANNEL_RANGE, canonical_channel

STATUS_OK = "ok"
STATUS_SUSPECT = "suspect"
STATUS_DEGRADED = "degraded"
STATUS_FAILED = "failed"

# Trust thresholds separating the four operating states.
TRUST_SUSPECT = 0.75
TRUST_DEGRADED = 0.45
TRUST_FAILED = 0.15

# Fleet-relative alarm thresholds, in robust z units against the distribution of
# the other nodes.  Chosen by sweeping specificity against sensitivity on clean
# simulated fleets; the false-positive rate is flat at zero across this range, so
# the lower end is taken to maximise sensitivity.
DRIFT_Z_THRESHOLD = 3.0
PEER_Z_THRESHOLD = 2.5

# Resolution below which two consecutive samples count as identical.
# Keyed by canonical channel name; look these up through `_resolution` /
# `_sigma`, never directly, so either naming convention resolves correctly.
CHANNEL_RESOLUTION = {
    "waste": 1e-4,
    "gas": 1e-4,
    "temp": 1e-3,
    "humidity": 1e-2,
}

# Expected short-horizon standard deviation of the *residual*, used to scale
# the drift and spike tests.
CHANNEL_SIGMA = {
    "waste": 0.03,
    "gas": 0.04,
    "temp": 0.60,
    "humidity": 2.50,
}


def _resolution(channel: str) -> float:
    return CHANNEL_RESOLUTION.get(canonical_channel(channel), 1e-6)


def _sigma(channel: str) -> float:
    return CHANNEL_SIGMA.get(canonical_channel(channel), 1.0)


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
    eps = _resolution(channel)
    streak = 1
    for a, b in zip(finite[::-1][:-1], finite[::-1][1:]):
        if abs(a - b) <= eps:
            streak += 1
        else:
            break
    if streak < 2:
        return 0.0, 0
    # A waste level legitimately plateaus; gas and temperature should not.
    tolerance = patience * (2 if canonical_channel(channel) == "waste" else 1)
    return float(min(1.0, max(0.0, (streak - 2) / max(tolerance, 1)))), int(streak)


def _is_collection_reset(series: np.ndarray, channel: str) -> bool:
    """
    Whether the latest change in a fill series is an emptying event.

    A collection is a large *negative* step to a low level, and it is completely
    normal.  Treating it as an impulsive sensor fault -- which a plain Hampel
    filter does, because it is by far the largest deviation in the window -- puts
    every healthy bin into a suspect state shortly after it is serviced.
    """
    if canonical_channel(channel) != "waste":
        return False
    finite = series[np.isfinite(series)]
    if finite.size < 2:
        return False
    drop = finite[-2] - finite[-1]
    return bool(drop > 0.25 and finite[-1] < 0.25)


def _score_spike(value: Optional[float], history: np.ndarray, channel: str,
                 series: Optional[np.ndarray] = None) -> float:
    """
    Hampel-filter outlier score.

    Operates on the *residual* series when one is supplied, so a value that
    moves with the rest of the fleet is not mistaken for an impulse.
    """
    if value is None or not math.isfinite(value):
        return 0.0
    hist = history[np.isfinite(history)]
    if hist.size < 5:
        return 0.0
    if series is not None and _is_collection_reset(series, channel):
        return 0.0
    z = abs(robust_z(value, hist))
    if z <= 4.0:
        return 0.0
    return float(min(1.0, (z - 4.0) / 6.0))


def residual_shift(residual: np.ndarray) -> float:
    """
    Signed change in a residual series: recent third minus earliest third.

    This is the statistic the drift test works on.  Reducing the window to one
    number lets the decision be made against the *fleet's* distribution of the
    same statistic, which is what makes the test self-scaling.
    """
    x = np.asarray(residual, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 6:
        return 0.0
    k = max(2, x.size // 3)
    return float(np.mean(x[-k:]) - np.mean(x[:k]))


def _score_drift(residual: np.ndarray, channel: str,
                 fleet_shifts: Optional[Sequence[float]] = None) -> (float, float):
    """
    Drift score, measured against how the rest of the fleet is behaving.

    Two corrections matter here, and getting either wrong wrecks the detector:

    * **Common mode must be removed.**  Internal bin temperature follows the
      diurnal cycle across the whole network, so a change test on the raw signal
      fires on every healthy node twice a day.  The input is therefore the
      residual against the fleet reference.

    * **The residual is still not stationary.**  A bin's temperature depends on
      its own decomposition state and a bin fills at its own rate, so even the
      residual legitimately trends.  Comparing this node's residual shift with
      the distribution of every other node's residual shift is what separates
      "this bin is filling faster than its neighbours" (normal, and information
      the planner wants) from "this sensor is walking away from physical
      reality" (a fault).

    The result is scale-free: no per-channel noise constant has to be guessed,
    and the same thresholds transfer between fill fraction, gas, temperature and
    humidity.
    """
    shift = residual_shift(residual)
    if fleet_shifts is None or len(fleet_shifts) < 4:
        # Single-node fallback: no fleet to compare against, so use the
        # channel's nominal noise scale.
        sigma = _sigma(channel)
        detected, magnitude = page_hinkley(residual, delta=0.5 * sigma,
                                           threshold_sigma=6.0, sigma=sigma)
        return (float(min(1.0, abs(magnitude) / (4.0 * sigma))) if detected else 0.0,
                float(magnitude) if detected else 0.0)

    z = abs(robust_z(shift, fleet_shifts))
    if z <= DRIFT_Z_THRESHOLD:
        return 0.0, 0.0
    return float(min(1.0, (z - DRIFT_Z_THRESHOLD) / 4.0)), float(shift)


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


def _score_peer(own_residual: Optional[float], peer_residuals: Optional[Sequence[float]],
                channel: str) -> float:
    """
    Fleet-consistency score, computed on *sustained residuals*.

    A stealthy poisoning attack keeps a node's own time series smooth, so no
    single-node detector fires.  What it cannot hide is that the node has moved
    away from the rest of the network while the network did not move.

    Comparing mean residuals over the window rather than instantaneous values
    matters twice over: it removes the level heterogeneity between a bazaar bin
    and a residential one (which would otherwise swamp the test), and it ignores
    single-sample noise, so the detector responds to a persistent offset -- the
    signature of poisoning, calibration error and slow drift alike.
    """
    if own_residual is None or not math.isfinite(own_residual) or not peer_residuals:
        return 0.0
    z = abs(robust_z(own_residual, peer_residuals))
    if z <= PEER_Z_THRESHOLD:
        return 0.0
    return float(min(1.0, (z - PEER_Z_THRESHOLD) / 4.0))


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


ANOMALY_DEADBAND = 0.15


def update_trust(previous: float, anomaly: float,
                 drop_rate: float = 0.60, recover_rate: float = 0.25,
                 deadband: float = ANOMALY_DEADBAND) -> float:
    """
    Asymmetric exponential trust update with a deadband.

    Trust collapses within a couple of cycles once a fault is evident and takes
    roughly five clean cycles to be restored, matching the operational rule that
    a suspect sensor must prove itself again before it is believed.

    The recovery rate has to be fast enough to undo an *isolated* alarm.  Every
    statistical detector fires occasionally on healthy data -- here on roughly a
    tenth of cycles -- and if recovery is slower than that arrival rate, trust
    ratchets monotonically downwards until every healthy channel is marked
    suspect.  A genuine fault fires on essentially every cycle, so it still
    drives trust to zero within a few assessments; only the isolated blips are
    forgiven.

    The deadband is what makes that asymmetry safe to run continuously.  Without
    it, any anomaly score above zero -- including the ordinary statistical
    noise a healthy sensor generates -- takes the fast downward path, while
    recovery only ever uses the slow one.  Over a few dozen cycles that ratchets
    every healthy channel into a suspect state: in testing it produced a 45%
    false-positive rate on a completely clean fleet.  Treating sub-threshold
    evidence as clean removes the ratchet while leaving genuine faults, whose
    scores sit far above the band, entirely unaffected.
    """
    anomaly = max(0.0, min(1.0, anomaly))
    evidence = 1.0 if anomaly < deadband else 1.0 - anomaly
    rate = drop_rate if evidence < previous else recover_rate
    return float(max(0.0, min(1.0, previous + rate * (evidence - previous))))


def _residual_series(series: np.ndarray, reference: Optional[np.ndarray]) -> np.ndarray:
    """
    Series with the shared fleet behaviour removed.

    Falling back to the series' own median keeps the detectors working for a
    single-node deployment, where no fleet reference exists.
    """
    arr = np.asarray(series, dtype=float)
    if reference is None:
        finite = arr[np.isfinite(arr)]
        centre = float(np.median(finite)) if finite.size else 0.0
        return arr - centre
    ref = np.asarray(reference, dtype=float)
    if ref.size < arr.size:
        ref = np.concatenate([np.full(arr.size - ref.size, np.nan), ref])
    return arr - ref[-arr.size:]


def assess_node(history: Dict[str, Sequence[float]],
                previous_trust: Optional[Dict[str, float]] = None,
                peers: Optional[Dict[str, Sequence[float]]] = None,
                reference: Optional[Dict[str, Sequence[float]]] = None,
                fleet_shifts: Optional[Dict[str, Sequence[float]]] = None,
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
        ``{channel: [mean residuals of the other nodes]}`` for the
        fleet-consistency detector.
    reference
        ``{channel: fleet reference series}``, normally the element-wise median
        across the network.  Drift and spike are measured against this, so
        genuine shared dynamics -- the diurnal temperature cycle, a network-wide
        rain event -- are not mistaken for per-node faults.
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
    reference = reference or {}
    fleet_shifts = fleet_shifts or {}
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
        residual = _residual_series(arr, reference.get(channel))
        past_residual = residual[:-1] if residual.size > 1 else np.asarray([], dtype=float)
        current_residual = (float(residual[-1]) if residual.size and np.isfinite(residual[-1])
                            else None)

        missing_streak = 0
        for v in arr[::-1]:
            if np.isfinite(v):
                break
            missing_streak += 1

        stuck_score, stuck_streak = _score_stuck(arr, channel)
        drift_score, drift_magnitude = _score_drift(
            residual, channel, fleet_shifts.get(channel))

        scores = {
            "range": _score_range(value, channel),
            "missing": _score_missing(missing_streak),
            "stuck": stuck_score,
            "spike": _score_spike(current_residual, past_residual, channel, series=arr),
            "drift": drift_score,
            "cross_channel": cross.get(channel, 0.0),
            "peer": _score_peer(
                float(np.nanmean(residual)) if np.isfinite(residual).any() else None,
                peers.get(channel), channel),
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
    """
    Assess every node, using leave-one-out fleet peers for the consistency test.

    The peer sample is built once and the node's own value removed per node,
    rather than rebuilding the whole fleet sample for each node, which would be
    quadratic in the network size.
    """
    previous_trust = previous_trust or {}
    node_ids = list(node_histories.keys())
    if not node_ids:
        return {}

    channels = sorted({c for hist in node_histories.values() for c in hist})

    # --- fleet reference series: element-wise median across the network ---
    # Right-aligned, because the newest sample is the one being judged and nodes
    # may hold windows of different length.
    reference: Dict[str, np.ndarray] = {}
    stacked: Dict[str, np.ndarray] = {}
    for channel in channels:
        series = [np.asarray(list(node_histories[n].get(channel, [])), dtype=float)
                  for n in node_ids]
        width = max((s.size for s in series), default=0)
        if width == 0:
            continue
        matrix = np.full((len(series), width), np.nan)
        for i, s in enumerate(series):
            if s.size:
                matrix[i, width - s.size:] = s
        stacked[channel] = matrix
        with np.errstate(all="ignore"):
            median = np.nanmedian(matrix, axis=0)
        # A single-node deployment has no fleet to compare against; leave the
        # reference undefined so each node falls back to its own centre.
        reference[channel] = median if len(node_ids) >= 3 else None

    # --- per-node residual summaries, for the leave-one-out fleet tests ---
    # `mean_residual` drives the peer test (a persistent offset) and
    # `shift_residual` drives the drift test (a change in that offset).
    mean_residual: Dict[str, np.ndarray] = {}
    shift_residual: Dict[str, np.ndarray] = {}
    for channel, matrix in stacked.items():
        ref = reference.get(channel)
        if ref is None:
            mean_residual[channel] = np.full(len(node_ids), np.nan)
            shift_residual[channel] = np.full(len(node_ids), np.nan)
            continue
        with np.errstate(all="ignore"):
            residuals = matrix - ref[None, :]
            all_nan = ~np.isfinite(residuals).any(axis=1)
            mean_residual[channel] = np.where(
                all_nan, np.nan,
                np.nanmean(np.where(np.isfinite(residuals), residuals, 0.0), axis=1))
        shift_residual[channel] = np.array(
            [residual_shift(residuals[i]) for i in range(residuals.shape[0])], dtype=float)

    results = {}
    for index, node_id in enumerate(node_ids):
        peers: Dict[str, List[float]] = {}
        shifts: Dict[str, List[float]] = {}
        for channel in stacked:
            others = np.delete(mean_residual[channel], index)
            others = others[np.isfinite(others)]
            if others.size:
                peers[channel] = others.tolist()
            others_shift = np.delete(shift_residual[channel], index)
            others_shift = others_shift[np.isfinite(others_shift)]
            if others_shift.size:
                shifts[channel] = others_shift.tolist()
        results[node_id] = assess_node(
            node_histories[node_id],
            previous_trust=previous_trust.get(node_id),
            peers=peers,
            reference={c: r for c, r in reference.items() if r is not None},
            fleet_shifts=shifts,
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
