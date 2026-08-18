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
``range``                  Hard physical-bound violations, beyond a noise
                           tolerance so a rounding artefact is not a "fault".
``missing``                Genuine absence (NaN), kept distinct from a zero.
``stuck``                  Frozen registers, via both a trailing run of
                           identical samples and the share of the window taken
                           by a single modal value (an intermittent hard-zero
                           never produces a clean run).
``spike``                  Impulsive noise, via a *local* Hampel filter with a
                           step veto so a level shift is not read as an impulse.
``drift``                  Slow bias, via the median of first differences --
                           a trend estimate that a single step change (a
                           collection) cannot move.
``dispersion``             Variance inflation (EMI) and variance collapse
                           (quantisation / a dying ADC), as the log spread of
                           the increments against the fleet.
``redundancy``             Analytical redundancy: each channel is predicted
                           from the others by a model fitted across the fleet,
                           and a node whose channels stop being mutually
                           consistent is flagged.  This is the detector that
                           sees a *constant* bias -- calibration error and
                           steady poisoning -- which no single-channel time
                           series test can see at all.
``physics``                Hard impossible combinations between channels.
``peer``                   Fleet inconsistency: a robust z-score of the node's
                           level against the other bins.
=========================  ===================================================

Three design decisions carry most of the detection performance, and each was
measured rather than assumed (see ``experiments/eval_sensor_health.py``):

*Every statistical test is fleet-relative.*  Absolute tests fire on the shared
diurnal cycle and on legitimate collection resets, which is worse than useless:
the alarms arrive precisely when the fleet is behaving normally.

*Every statistical test is step-immune.*  Emptying a bin is a large, abrupt,
completely legitimate level change.  Statistics built on means over a window
(``mean(recent) - mean(early)``) are dominated by it; statistics built on the
median of the increments ignore it.  Replacing the former with the latter cut
the clean-fleet false-positive rate by roughly two thirds on its own.

*Evidence must be confirmed before it is fully believed.*  A statistical
detector fires occasionally on healthy data; a real fault fires on essentially
every cycle.  Each detector therefore carries a confirmation memory, and an
unconfirmed alarm contributes only a fraction of its score.  Deterministic
detectors (range, missing, stuck) are exempt -- there is nothing statistical
about a NaN.

Trust is updated asymmetrically -- it falls quickly on evidence of a fault and
recovers slowly -- because the cost of trusting a broken sensor for one more
cycle greatly exceeds the cost of down-weighting a healthy one.

The combined score is an evidence score, not a probability
----------------------------------------------------------
The nine detector scores are aggregated by the noisy-OR form in
:func:`_combine`, which is monotone in every input and stops any single
moderate score from saturating the result.  It is not a posterior probability
that the channel is faulty, and it is never reported as one.  Reading it that
way would require the detectors to be conditionally independent, and they are
not: ``spike``, ``drift``, ``dispersion`` and ``peer`` all read one residual
series, and on channels carrying a fault the drift and dispersion scores
correlate at Pearson r = 0.61.  The number is consumed only by
:func:`update_trust`, which compares it against fixed thresholds, so the
ordering it induces is what the system relies on.  :func:`_combine` records the
full measurement.

Status and trust answer different questions
-------------------------------------------
``trust`` is a *data-quality weight*: how much should this reading count in the
priority right now.  It must recover quickly, otherwise isolated alarms ratchet
it to zero.  ``status`` is a *maintenance state*: should a technician look at
this node.  A channel that dropped a burst of packets ten minutes ago is
perfectly usable now (trust ~ 1) but is emphatically not healthy, and reporting
it as ``ok`` is how intermittent hardware faults get ignored until the node dies.
The two are therefore decoupled: a slowly-recovering ``reliability`` counter,
driven only by *unambiguous* faults, floors the reported status without ever
touching the renormalisation weight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# the *same statistic* computed on the other nodes.  All of them are two-sided.
#
# These are higher than the textbook "3 sigma" because the reference sample is a
# fleet of a few dozen nodes, not an asymptotic population: a MAD estimated from
# ~20 points is itself noisy, which fattens the effective tails.  Sweeping them
# on clean fleets (experiments/eval_sensor_health.py) put the knee here.
DRIFT_Z_THRESHOLD = 3.2
PEER_Z_THRESHOLD = 4.0
DISPERSION_Z_THRESHOLD = 3.2
REDUNDANCY_Z_THRESHOLD = 3.5
SPIKE_Z_THRESHOLD = 5.0
# Used only when a fleet reference exists but the fleet is too small for a
# cross-sectional comparison; the trend is then judged against the channel's own
# increment noise, which is weaker evidence and gets a stricter threshold.
DRIFT_SELF_Z_THRESHOLD = 6.0

# Analytical redundancy is only meaningful when the channels genuinely predict
# one another.  Humidity in an open network is driven by the weather and not by
# the bin, so its cross-channel model explains nothing (measured R^2 ~ 0.05) and
# its residual is pure noise; scoring it would manufacture false alarms.
REDUNDANCY_MIN_R2 = 0.55
# Minimum fleet size before the cross-sectional tests mean anything.  A MAD
# estimated from a handful of peers is so noisy that the z-scores built on it are
# close to meaningless, so below this a node falls back to the within-node
# detectors, which need no reference population.
#
# The threshold matters more than it looks, because the failure is not gradual:
# switching the fleet tests on while the peer MAD is still under-determined is
# worse than leaving them off.  Sweeping clean fixtures, 8 seeds per size:
#
#     nodes   clean FPR   seeds w/ FP   precision
#         8      0.0000          0/8         0.94
#         9      0.0000          0/8         0.94
#        10      0.0906          6/8         0.49
#        11      0.1079          8/8         0.42
#        12      0.0182          4/8         0.82
#        14      0.0089          2/8         0.81
#        20      0.0016          1/8         0.93
#
# There is a distinct instability band at 10-11 nodes -- at 11 every seed
# produced false alarms and precision fell to 0.42 -- which is precisely where a
# threshold of 10 used to place the switch-over.  12 is the first size whose
# false-positive rate is inside the 0.05 tolerance, so that is where it goes.
# Recall below the threshold is 0.77 against 0.94+ above it: a real cost, but a
# quiet loss of sensitivity is a fair trade for not crying wolf on a healthy
# small network, and the operator can see which regime a fleet is in.
MIN_FLEET_FOR_PEERS = 12

# The redundancy model has its own, lower, minimum -- and it must stay lower.
# On a channel it can model, redundancy strictly dominates the raw peer
# comparison and switches it off (see `_redundancy_scores`).  If peers were to
# activate at a smaller fleet than redundancy, there would be a window where the
# weaker test runs with nothing to suppress it: raising a single shared constant
# to 12 did exactly that, moving the instability band to 10-11 -> 12-13 (FPR
# 0.047 and 0.055, 7 of 8 seeds) instead of removing it.  A fitted model needs
# only enough complete rows for the trimmed regression, which is a weaker
# requirement than a trustworthy MAD, so the two thresholds are not the same
# number and must not share one.
MIN_FLEET_FOR_REDUNDANCY = 10

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
# the range tolerance and to floor the dispersion statistic.
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


# ---------------------------------------------------------------------------
# Recursive state
# ---------------------------------------------------------------------------
# Detectors that are allowed to act on a single cycle's evidence.  A NaN, an
# out-of-range value or a frozen register is not a statistical inference, so
# waiting for confirmation would only delay a certain diagnosis.
DETERMINISTIC_DETECTORS = ("range", "missing", "stuck", "physics")

# How much of a statistical detector's score counts before it has been
# confirmed.  Low enough that one isolated alarm cannot cross a status
# boundary on its own, high enough that a genuine fault is not invisible on
# the cycle it starts.
UNCONFIRMED_FLOOR = 0.30
CONFIRM_ALPHA = 0.45          # EWMA rate of the confirmation memory
CONFIRM_ARM = 0.10            # score above which a detector counts as "armed"
CONFIRM_LO = 0.50             # confirmation below this contributes nothing extra
CONFIRM_HI = 0.75             # confirmation at or above this is fully believed

# Unambiguous-fault memory.  One confirmed hard fault costs the full budget and
# it is earned back over ~16 cycles, i.e. a channel must deliver roughly half a
# day of clean data (at the default 30-minute cadence) before it is called
# healthy again.  Only DETERMINISTIC detectors feed this, so it can never fire
# on a statistical false alarm.
RELIABILITY_RECOVERY = 1.0 / 16.0
RELIABILITY_SUSPECT = 0.55
RELIABILITY_DEGRADED = 0.20


@dataclass
class ChannelState:
    """
    Everything one channel remembers between assessments.

    Kept JSON-serialisable so the service can round-trip it through
    ``SensorHealth.detail`` without a schema migration.
    """

    trust: float = 1.0
    reliability: float = 1.0
    confirm: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {
            "trust": round(float(self.trust), 5),
            "reliability": round(float(self.reliability), 5),
            "confirm": {k: round(float(v), 4) for k, v in self.confirm.items()
                        if v > 1e-3},
        }

    @classmethod
    def coerce(cls, value: Any) -> "ChannelState":
        """
        Accept a bare trust float or a persisted state dict.

        The float form is what the first version of this module stored, and the
        service upgrades in place rather than resetting every node's history.
        """
        if isinstance(value, ChannelState):
            return cls(trust=value.trust, reliability=value.reliability,
                       confirm=dict(value.confirm))
        if isinstance(value, dict):
            return cls(
                trust=float(value.get("trust", 1.0)),
                reliability=float(value.get("reliability", 1.0)),
                confirm={str(k): float(v)
                         for k, v in (value.get("confirm") or {}).items()},
            )
        try:
            return cls(trust=float(value))
        except (TypeError, ValueError):
            return cls()


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
    reliability: float = 1.0
    scores: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    state: Dict = field(default_factory=dict)

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
            "reliability": round(self.reliability, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "flags": list(self.flags),
        }


@dataclass
class FleetStats:
    """
    Leave-one-out fleet statistics for one channel of one node.

    Every field is the distribution of a statistic over the *other* nodes, so a
    node can never be compared against a reference that contains itself.
    """

    reference: Optional[np.ndarray] = None      # fleet median series
    peer_levels: Optional[List[float]] = None   # others' median residual
    peer_trends: Optional[List[float]] = None   # others' robust trend
    peer_dispersions: Optional[List[float]] = None   # others' log increment spread
    redundancy_z: float = 0.0                   # already fleet-normalised
    has_redundancy: bool = False                # a usable cross-channel model exists


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


def robust_scale(reference: Sequence[float], floor: float = 0.0) -> float:
    """
    Dispersion of a reference sample, with a floor in physical units.

    The floor matters: without it, a channel whose peers happen to agree very
    closely on one cycle produces a divide-by-almost-zero and every node looks
    like a five-sigma outlier.  The floor is the smallest spread that is
    physically meaningful for the channel, so agreement tighter than the sensor
    noise cannot manufacture significance.
    """
    ref = np.asarray(list(reference), dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size < 3:
        return 0.0
    mad = median_abs_deviation(ref)
    if mad < 1e-12:
        mad = float(np.std(ref))
    return float(max(mad, floor))


def robust_z(value: float, reference: Sequence[float], floor: float = 0.0) -> float:
    """Robust z-score of ``value`` against a reference sample."""
    ref = np.asarray(list(reference), dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size < 3 or not np.isfinite(value):
        return 0.0
    scale = robust_scale(ref, floor=floor)
    if scale <= 1e-12:
        return 0.0
    return float((value - float(np.median(ref))) / scale)


def _row_nanmedian(matrix: np.ndarray) -> np.ndarray:
    """
    Per-row median ignoring NaN, returning NaN for rows that are entirely NaN.

    ``np.nanmedian`` warns and returns NaN for an all-NaN slice; a node that has
    gone completely dark is an ordinary, expected state here, not something to
    print a RuntimeWarning about on every dashboard refresh.
    """
    out = np.full(matrix.shape[0], np.nan)
    for i in range(matrix.shape[0]):
        row = matrix[i][np.isfinite(matrix[i])]
        if row.size:
            out[i] = float(np.median(row))
    return out


def _ramp(z: float, threshold: float, width: float = 4.0) -> float:
    """Map a two-sided z-score onto an anomaly score in [0, 1]."""
    excess = abs(z) - threshold
    if excess <= 0.0:
        return 0.0
    return float(min(1.0, excess / max(width, 1e-9)))


def page_hinkley(x: np.ndarray, delta: float, threshold_sigma: float = 4.0,
                 sigma: float = 1.0):
    """
    Two-sided Page-Hinkley change detector.

    Retained for the single-node fallback, where there is no fleet to compare
    against and an absolute test is the only option available.  Returns
    ``(detected, magnitude)`` where ``magnitude`` is the signed estimate of the
    mean shift in the units of ``x``.
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

    half = max(2, n // 3)
    magnitude = float(np.mean(x[-half:]) - np.mean(x[:half]))
    return detected, magnitude


def residual_shift(residual: np.ndarray) -> float:
    """
    Signed change in a residual series: recent third minus earliest third.

    Kept because it is the natural *magnitude* estimate for drift correction,
    but it is deliberately **not** the drift test statistic: a single collection
    reset inside the window moves it by far more than any realistic sensor
    drift, which made it fire on healthy bins for several cycles after every
    service (measured |z| up to 10 on a clean fleet).  Use
    :func:`robust_trend` for detection and this only for reporting.
    """
    x = np.asarray(residual, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 6:
        return 0.0
    k = max(2, x.size // 3)
    return float(np.mean(x[-k:]) - np.mean(x[:k]))


def robust_trend(residual: np.ndarray) -> float:
    """
    Step-immune estimate of the total monotone change across the window.

    The median of the first differences, scaled back up by the number of
    increments.  A drifting sensor biases *every* increment slightly, so the
    median moves with it.  A collection reset -- or any other single step --
    contributes exactly one extreme increment, which a median discards.  That
    single substitution is what makes a fleet-relative drift test usable on a
    channel that is legitimately reset twice a week.
    """
    x = np.asarray(residual, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 6:
        return 0.0
    return float(np.median(np.diff(x)) * (x.size - 1))


def increment_dispersion(residual: np.ndarray, channel: str) -> float:
    """
    Log spread of the residual's increments.

    Works in logs because the interesting deviations are multiplicative in both
    directions: electromagnetic interference inflates the spread by an order of
    magnitude, a frozen or coarsely quantised register collapses it to zero.
    The floor keeps the collapsed case finite instead of ``-inf``.
    """
    x = np.asarray(residual, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 6:
        return float("nan")
    spread = median_abs_deviation(np.diff(x))
    return float(math.log(max(spread, 1e-3 * _sigma(channel))))


# ---------------------------------------------------------------------------
# Individual detectors -- each returns an anomaly score in [0, 1]
# ---------------------------------------------------------------------------
def _score_range(value: Optional[float], channel: str) -> float:
    """
    Hard physical-bound violation.

    A tolerance of a couple of noise standard deviations is subtracted first: a
    fill fraction reading 1.301 against a 1.300 bound is a rounding artefact,
    and treating it as a sensor failure was putting healthy channels into the
    unambiguous-fault memory, where a statistical false alarm can never reach.
    """
    if value is None or not math.isfinite(value):
        return 0.0                                   # handled by the missing detector
    lo, hi = CHANNEL_RANGE.get(channel, (-math.inf, math.inf))
    tolerance = 2.0 * _sigma(channel)
    if (lo - tolerance) <= value <= (hi + tolerance):
        return 0.0
    span = max(hi - lo, 1e-6)
    excess = (lo - value) if value < lo else (value - hi)
    return float(min(1.0, 0.6 + 0.4 * min(1.0, excess / span)))


def _score_missing(missing_streak: int, window_size: int = 0,
                   missing_count: int = 0, patience: int = 3,
                   loss_budget: float = 0.20) -> float:
    """
    Absence score from the trailing gap *and* the loss rate over the window.

    The trailing streak alone is close to blind to bursty radio loss, which is
    what real packet loss looks like: a Gilbert-Elliott burst of a few samples
    sits in the middle of the window on all but one cycle, so a
    streak-only test sees nothing at all on four cycles out of five.  Scoring
    the share of the window that never arrived closes that gap, and it costs
    nothing in false alarms because the evidence is a NaN -- a healthy node
    reporting on a fixed cadence produces exactly zero of them.

    ``loss_budget`` is the fraction of a window that may go missing before the
    channel is considered failed; losing more than one sample in seven already
    scores half.
    """
    streak_score = (0.0 if missing_streak <= 0
                    else float(min(1.0, missing_streak / max(patience, 1))))
    rate_score = 0.0
    if window_size > 0 and missing_count > 0:
        rate_score = float(min(1.0, (missing_count / window_size)
                               / max(loss_budget, 1e-9)))
    return max(streak_score, rate_score)


def _score_stuck(values: np.ndarray, channel: str,
                 patience: int = 6) -> Tuple[float, int]:
    """
    Frozen-register score from two complementary views.

    The trailing run of identical samples catches a register that has locked up
    and stayed locked.  The *modal share* -- the fraction of the window occupied
    by a single value -- catches the intermittent case, where an ADC or supply
    fault drops the reading to a hard zero on most but not all samples and so
    never produces a clean run.  A live channel carrying real noise has a modal
    share of about one sample, so the test has essentially no cost on healthy
    data.
    """
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
    # A waste level legitimately plateaus; gas and temperature should not.
    tolerance = patience * (2 if canonical_channel(channel) == "waste" else 1)
    run_score = (0.0 if streak < 2
                 else float(min(1.0, max(0.0, (streak - 2) / max(tolerance, 1)))))

    modal_score = 0.0
    if finite.size >= 8:
        quantised = np.round(finite / max(eps, 1e-12))
        _, counts = np.unique(quantised, return_counts=True)
        share = float(counts.max()) / float(finite.size)
        # Half the window sharing one exact value is already implausible for a
        # noisy analogue channel; ramp to full score by 90%.
        modal_score = float(min(1.0, max(0.0, (share - 0.5) / 0.4)))

    return max(run_score, modal_score), int(streak if streak >= 2 else 0)


def _score_spike(residual: np.ndarray, channel: str, local: int = 8) -> float:
    """
    Local Hampel outlier score with a step veto.

    Two changes from a textbook Hampel filter, both forced by the data:

    * The reference is the *immediately preceding* samples, not the whole
      window.  After a bin is emptied, every remaining sample in the window sits
      far from the pre-collection median, so a whole-window filter keeps firing
      for as long as the reset stays inside the window -- five consecutive
      cycles at the default stride.
    * A deviation that the previous sample already showed, in the same
      direction, is a level shift, not an impulse.  Level shifts belong to the
      drift and redundancy tests, which can tell a service event from a fault;
      double-counting them here just inflates the false-positive rate.  Genuine
      impulsive noise alternates sign, so it survives the veto.
    """
    x = np.asarray(residual, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < local + 2:
        return 0.0
    scale_floor = 0.25 * _sigma(channel)

    z_now = robust_z(x[-1], x[-(local + 1):-1], floor=scale_floor)
    if abs(z_now) <= SPIKE_Z_THRESHOLD:
        return 0.0
    z_prev = robust_z(x[-2], x[-(local + 2):-2], floor=scale_floor)
    if abs(z_prev) > SPIKE_Z_THRESHOLD and (z_prev > 0) == (z_now > 0):
        return 0.0                                   # sustained shift, not an impulse
    return _ramp(z_now, SPIKE_Z_THRESHOLD, width=6.0)


def _score_drift(residual: np.ndarray, channel: str,
                 fleet_trends: Optional[Sequence[float]] = None,
                 have_reference: bool = True) -> Tuple[float, float]:
    """
    Drift score, measured against how the rest of the fleet is behaving.

    Three corrections matter here, and getting any of them wrong wrecks the
    detector:

    * **Common mode must be removed.**  Internal bin temperature follows the
      diurnal cycle across the whole network, so a change test on the raw signal
      fires on every healthy node twice a day.  The input is therefore the
      residual against the fleet reference.

    * **The residual is still not stationary.**  A bin's temperature depends on
      its own decomposition state and a bin fills at its own rate, so even the
      residual legitimately trends.  Comparing this node's trend with the
      distribution of every other node's trend is what separates "this bin is
      filling faster than its neighbours" (normal, and information the planner
      wants) from "this sensor is walking away from physical reality" (a fault).

    * **The trend estimator must ignore steps.**  See :func:`robust_trend`.

    The result is scale-free: no per-channel noise constant has to be guessed,
    and the same thresholds transfer between fill fraction, gas, temperature and
    humidity.
    """
    magnitude = residual_shift(residual)
    trend = robust_trend(residual)

    if fleet_trends is not None and len(fleet_trends) >= 3:
        z = robust_z(trend, fleet_trends, floor=0.1 * _sigma(channel))
        score = _ramp(z, DRIFT_Z_THRESHOLD)
        return score, (float(magnitude) if score > 0.0 else 0.0)

    if not have_reference:
        # A lone node cannot separate "the sensor is drifting" from "the weather
        # is changing": both produce a monotone trend in the only series
        # available, and no statistic computed from that series can tell them
        # apart.  Reporting a drift here is guessing, and on a channel with real
        # common-mode dynamics -- internal temperature follows the diurnal cycle
        # -- the guess is wrong most of the time.  Measured on a single-node
        # fixture, the old absolute Page-Hinkley fallback put three quarters of
        # all healthy channels into a non-ok state.  The deterministic detectors
        # still run; this one honestly abstains.
        return 0.0, 0.0

    # A fleet reference exists but there are too few peers for a robust
    # cross-sectional z.  Common mode is already removed, so the trend can be
    # judged against the channel's own increment noise -- self-normalising keeps
    # it scale-free -- with a conservative threshold to reflect the weaker
    # evidence.
    finite = np.asarray(residual, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 8:
        return 0.0, 0.0
    scale = median_abs_deviation(np.diff(finite)) * math.sqrt(finite.size - 1)
    if scale < 1e-9:
        return 0.0, 0.0
    score = _ramp(trend / scale, DRIFT_SELF_Z_THRESHOLD)
    return score, (float(magnitude) if score > 0.0 else 0.0)


def _score_dispersion(residual: np.ndarray, channel: str,
                      fleet_dispersions: Optional[Sequence[float]] = None) -> float:
    """
    Variance inflation or collapse relative to the fleet.

    Both tails are faults and both are common: interference and an unstable
    supply inflate the spread, while a dying ADC or a coarsely quantising
    firmware update collapses it.  Because the statistic is a log spread, the
    same threshold covers both without any per-channel calibration.
    """
    if fleet_dispersions is None or len(fleet_dispersions) < 3:
        return 0.0
    own = increment_dispersion(residual, channel)
    if not math.isfinite(own):
        return 0.0
    return _ramp(robust_z(own, fleet_dispersions, floor=0.10),
                 DISPERSION_Z_THRESHOLD)


def _score_physics(values: Dict[str, Optional[float]]) -> Dict[str, float]:
    """
    Physically impossible combinations between simultaneously-read channels.

    Cheap, deterministic and independent of any fleet statistic, so these fire
    immediately and are exempt from confirmation.  The lookups go through
    :func:`canonical_channel` because this function is called with whichever
    vocabulary the caller uses -- the previous version keyed on the Django field
    names only, which meant it silently scored nothing at all when driven from
    the core (every lookup missed, and every result key missed on the way out).
    """
    by_canonical: Dict[str, Optional[float]] = {}
    original: Dict[str, str] = {}
    for key, value in values.items():
        canonical = canonical_channel(key)
        by_canonical[canonical] = value
        original[canonical] = key

    out = {key: 0.0 for key in values}

    def bump(canonical: str, score: float) -> None:
        key = original.get(canonical)
        if key is not None:
            out[key] = max(out[key], score)

    def ok(v):
        return v is not None and isinstance(v, (int, float)) and math.isfinite(v)

    gas = by_canonical.get("gas")
    temp = by_canonical.get("temp")
    hum = by_canonical.get("humidity")
    waste = by_canonical.get("waste")

    # Decomposition is exothermic and produces gas, so a high gas reading paired
    # with an internal temperature below any plausible ambient is inconsistent.
    if ok(gas) and ok(temp) and gas > 0.55 and temp < 12.0:
        bump("gas", 0.5)
        bump("temp", 0.5)
    # Saturated air inside a closed bin cannot sit at very low humidity while
    # the contents are hot and gassy.
    if ok(gas) and ok(hum) and gas > 0.6 and hum < 25.0:
        bump("humidity", 0.4)
    # Strong odour from an essentially empty bin is not physical.
    if ok(gas) and ok(waste) and gas > 0.7 and waste < 0.05:
        bump("gas", 0.45)
        bump("waste", 0.45)
    return out


# Backwards-compatible alias: the detector used to be called ``cross_channel``.
_score_cross_channel = _score_physics


def _score_peer(own_level: Optional[float],
                peer_levels: Optional[Sequence[float]], channel: str,
                has_redundancy: bool = False) -> float:
    """
    Fleet-consistency score on the node's sustained residual level.

    Comparing median residuals over the window rather than instantaneous values
    removes single-sample noise, so the detector responds to a persistent offset
    -- the signature of poisoning, calibration error and slow drift alike.

    This is the *weakest* member of the ensemble and it is deliberately
    subordinate to :func:`_score_redundancy`.  Bins are genuinely heterogeneous:
    a bazaar bin sits near full and a residential one near empty, by design and
    for reasons the planner needs to know about.  So the fleet distribution of
    levels is wide and heavy-tailed, and -- fatally -- the same healthy outlier
    node is an outlier on *every* cycle, which is a false alarm that no amount
    of confirmation can wash out.  Measured on clean fleets, this single test
    produced almost all of the remaining false positives.

    It is therefore skipped entirely whenever the channel has a usable
    cross-channel model, which conditions on the node's own other channels and
    so is immune to level heterogeneity.  It stays enabled only where no such
    model exists -- humidity in an open network is driven by the weather rather
    than by the bin -- because there a crude fleet comparison is the only fleet
    evidence available at all.
    """
    if has_redundancy:
        return 0.0
    if own_level is None or not math.isfinite(own_level) or not peer_levels:
        return 0.0
    return _ramp(robust_z(own_level, peer_levels, floor=0.5 * _sigma(channel)),
                 PEER_Z_THRESHOLD)


def _score_redundancy(redundancy_z: float) -> float:
    """Analytical-redundancy score; the z-score is produced fleet-wide."""
    return _ramp(float(redundancy_z), REDUNDANCY_Z_THRESHOLD)


# ---------------------------------------------------------------------------
# Cross-channel analytical redundancy, fitted across the fleet
# ---------------------------------------------------------------------------
def _trimmed_least_squares(X: np.ndarray, y: np.ndarray,
                           keep: float = 0.75, rounds: int = 3
                           ) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
    """
    Least squares refitted after discarding the worst-fitting rows.

    A plain fit is dragged towards exactly the minority of faulty nodes the test
    exists to isolate, and with three bad nodes in twenty that bias is enough to
    hide them.  Trimming anchors the model on the healthy majority; the
    discarded rows are still scored against it.

    Returns ``(coefficients, r2_on_kept_rows, kept_mask)``.
    """
    n, p = X.shape
    if n < p + 4:
        return None, 0.0, None
    keep_n = max(p + 2, int(round(keep * n)))
    mask = np.ones(n, dtype=bool)
    beta: Optional[np.ndarray] = None
    r2 = 0.0
    for _ in range(max(1, rounds)):
        try:
            beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        except np.linalg.LinAlgError:              # pragma: no cover - singular
            return None, 0.0, None
        residual = y - X @ beta
        centred = np.abs(residual - np.median(residual[mask]))
        new_mask = np.zeros(n, dtype=bool)
        new_mask[np.argsort(centred)[:keep_n]] = True
        kept_var = float(np.var(y[new_mask]))
        r2 = (1.0 - float(np.var(residual[new_mask])) / kept_var) if kept_var > 1e-12 else 0.0
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return beta, float(r2), mask


def _leverage(X: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """
    Hat-matrix diagonal of every row against the design actually fitted.

    ``h_ii = x_i' (Xk' Xk)^-1 x_i`` where ``Xk`` is the sub-design the trimmed
    regression kept.  For the kept rows this is the ordinary hat diagonal, which
    the fixtures confirm: the leverages of the kept rows sum to 4.0000 against a
    design of exactly 4 columns, which is the identity ``sum_i h_ii = rank``.
    For a discarded row it is still a well-defined quadratic form, but it is a
    leverage against a design the row was not part of, so it is not bounded by 1
    and in practice is not (measured maximum 10.4).

    A node sitting at the edge of the fleet's operating range, the fullest bin in
    the network say, is not being interpolated by the cross-channel model, it is
    being extrapolated to.  How that should change the weight given to its
    residual depends on whether the fit saw it, which is what
    :func:`_residual_scale` decides.
    """
    Xk = X[kept]
    try:
        gram = np.linalg.pinv(Xk.T @ Xk)
    except np.linalg.LinAlgError:                  # pragma: no cover - singular
        return np.zeros(X.shape[0])
    return np.clip(np.einsum("ij,jk,ik->i", X, gram, X), 0.0, None)


# Smallest value the factor ``1 - h_ii`` is allowed to take for a row that was
# inside the fit.  A row whose leverage approaches 1 is one the regression passes
# almost exactly through, so its raw residual approaches zero and dividing by
# sqrt(1 - h_ii) approaches a division by zero.  This is not hypothetical:
# measured over the fleet fixtures, 0.19% of fitted rows have h_ii > 0.9 and the
# smallest observed 1 - h_ii is 0.0040, which unfloored would inflate a residual
# by a factor of 16.  The floor caps the inflation at sqrt(1 / 0.05) = 4.5, well
# above the 1.9 that the 95th percentile of ordinary rows reaches, so it binds
# only on the degenerate rows it exists for.  It is a numerical guard with a
# stated cost, not a modelling choice.
LEVERAGE_FLOOR = 0.05


def _residual_scale(h: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """
    Per-row standard error of the residual, in units of the residual sigma.

    Two different quantities are needed, because the trimmed fit splits the fleet
    into rows the regression was fitted on and rows it was not.

    * A row that was fitted has residual variance ``sigma^2 (1 - h_ii)``.  The
      fit is pulled towards such a row, so its raw residual is systematically too
      small, and dividing by ``sqrt(1 - h_ii)`` is what restores a constant
      variance.  This is the internally studentised residual of standard
      regression diagnostics (Belsley, Kuh and Welsch 1980, chapter 2; Cook and
      Weisberg 1982, section 2.2).
    * A row the trim discarded was not part of the fit, so its residual is a
      genuine out-of-sample prediction error with variance ``sigma^2 (1 + h_ii)``.

    Earlier versions of this module used ``sqrt(1 + h_ii)`` for every row and
    described it as the textbook correction, which it is not: it is the correct
    form for the discarded rows only.  Applied to a fitted row it shrinks a
    residual that was already too small, so the test under-flags exactly the
    high-leverage nodes it was reasoning about.  The two cases cannot be merged
    into one expression, because ``h_ii`` is computed against the kept design and
    so exceeds 1 for 4% of discarded rows, where ``sqrt(1 - h_ii)`` is undefined.
    """
    scale = np.sqrt(1.0 + h)                       # out-of-sample prediction error
    scale[kept] = np.sqrt(np.maximum(1.0 - h[kept], LEVERAGE_FLOOR))
    return scale


def _redundancy_scores(levels: Dict[str, np.ndarray], channels: Sequence[str]
                       ) -> Tuple[Dict[str, np.ndarray], set]:
    """
    Per-node, per-channel analytical-redundancy z-scores.

    Each channel is regressed on the others across the fleet at the current
    instant, and a node whose observed value departs from what its *own other
    channels* imply is flagged.  This is the only test in the ensemble that sees
    a perfectly steady bias: a mis-calibrated or steadily poisoned gas sensor has
    a smooth time series, a stable variance and a stable fleet rank, so every
    time-series test and the peer test are blind to it -- but it no longer agrees
    with the fill level and temperature measured on the same node.

    Single-fault isolation is applied afterwards.  A broken channel makes the
    whole system of relations inconsistent, so *every* channel of that node
    shows a large residual; attributing the inconsistency to all of them would
    turn one real fault into four alarms.  The residual is therefore assigned to
    the channel that explains it best, exactly as in structured-residual FDI.

    Returns ``(scores_by_channel, channels_with_a_usable_model)``.  The second
    element matters downstream: on a channel that *is* modelled, this test
    strictly dominates the raw peer comparison, so the weaker test is switched
    off rather than allowed to contribute duplicate false alarms.
    """
    usable = [c for c in channels if c in levels and np.isfinite(levels[c]).sum() >= 8]
    n_nodes = len(next(iter(levels.values()))) if levels else 0
    out = {c: np.zeros(n_nodes) for c in channels}
    if len(usable) < 3 or n_nodes < MIN_FLEET_FOR_REDUNDANCY:
        return out, set()

    raw: Dict[str, np.ndarray] = {}
    for target in usable:
        predictors = [c for c in usable if c != target]
        columns = [levels[c] for c in predictors]
        complete = np.ones(n_nodes, dtype=bool)
        for col in columns + [levels[target]]:
            complete &= np.isfinite(col)
        if complete.sum() < MIN_FLEET_FOR_REDUNDANCY + 2:
            continue
        X = np.column_stack(columns + [np.ones(n_nodes)])
        Xc, yc = X[complete], levels[target][complete]
        beta, r2, kept = _trimmed_least_squares(Xc, yc)
        if beta is None or kept is None or r2 < REDUNDANCY_MIN_R2:
            continue                               # no usable redundancy here
        scale = _residual_scale(_leverage(Xc, kept), kept)
        residual = np.full(n_nodes, np.nan)
        residual[complete] = (yc - Xc @ beta) / scale
        raw[target] = residual

    if not raw:
        return out, set()

    # Leave-one-out normalisation: a node is never part of its own reference.
    z_by_channel: Dict[str, np.ndarray] = {}
    for channel, residual in raw.items():
        z = np.zeros(n_nodes)
        for i in range(n_nodes):
            if not np.isfinite(residual[i]):
                continue
            z[i] = robust_z(residual[i], np.delete(residual, i))
        z_by_channel[channel] = z

    # Single-fault isolation across the channels of each node.
    stack = np.vstack([z_by_channel[c] for c in z_by_channel])
    names = list(z_by_channel)
    winner = np.argmax(np.abs(stack), axis=0)
    for j, channel in enumerate(names):
        out[channel] = np.where(winner == j, np.abs(stack[j]), 0.0)
    return out, set(names)


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------
DETECTOR_WEIGHTS = {
    "range": 1.00,
    "missing": 1.00,
    "stuck": 0.85,
    "spike": 0.65,
    "drift": 0.80,
    "dispersion": 0.75,
    "redundancy": 0.85,
    "physics": 0.60,
    "peer": 0.70,
}


def _confirmation_gate(name: str, score: float, confirm: Dict[str, float]) -> float:
    """
    Update a detector's confirmation memory and return its believed score.

    Every statistical test fires occasionally on healthy data and continuously
    on a real fault, so *persistence* -- not magnitude -- is the cleanest signal
    separating the two.  An unconfirmed alarm is still reported (it is real
    evidence) but is worth only ``UNCONFIRMED_FLOOR`` of its face value, which
    is not enough on its own to move a channel out of ``ok``.  Deterministic
    detectors bypass this entirely.
    """
    if name in DETERMINISTIC_DETECTORS:
        return score
    armed = 1.0 if score >= CONFIRM_ARM else 0.0
    previous = float(confirm.get(name, 0.0))
    updated = previous + CONFIRM_ALPHA * (armed - previous)
    confirm[name] = updated
    if score <= 0.0:
        return 0.0
    gate = (updated - CONFIRM_LO) / max(CONFIRM_HI - CONFIRM_LO, 1e-9)
    gate = min(1.0, max(0.0, gate))
    return float(score * (UNCONFIRMED_FLOOR + (1.0 - UNCONFIRMED_FLOOR) * gate))


def _combine(scores: Dict[str, float]) -> float:
    """
    Aggregate the weighted detector scores into one evidence score in [0, 1].

    The arithmetic is the noisy-OR form, one minus the product of the
    complements.  It is used because evidence should accumulate while no single
    moderate score saturates the result, which is what a plain max would do, and
    because it is monotone in every input.  That is the whole of the
    justification: it is an aggregation rule, not an inference.

    The result is NOT a probability and must not be reported as one.  Reading
    this expression as "the probability that at least one detector is right"
    requires the detectors to be conditionally independent given the channel's
    state, and they are demonstrably not.  Measured on the project's own
    fixtures (experiments/eval_sensor_health.py, 99,200 assessments over 4 clean
    fleets and 4 seeds by 9 fault modes):

    * Four of the nine detectors read one series.  ``spike``, ``drift`` and
      ``dispersion`` are handed the same residual array object by
      :func:`assess_node` on every call (80 of 80 in a single fleet assessment),
      and ``peer`` is handed the median of that same array.  This is a shared
      input by construction, not an incidental correlation.
    * On channels carrying an injected fault, which is the population where this
      function is actually accumulating evidence, that group co-varies:
      ``drift`` against ``dispersion`` gives Pearson r = 0.61 (n = 3348).
      Pooled over every assessment, clean and faulted alike, the same pair gives
      r = 0.29 on the raw scores and r = 0.46 on the confirmation-gated scores
      that are the arguments to this function.
    * The dependence is not only through the shared series.  ``stuck`` and
      ``redundancy`` read different inputs and still reach r = 0.35 on faulted
      channels, because a real fault is a common cause that moves both.
    * On clean fleets every pairwise correlation is below 0.02, so the
      dependence appears precisely where the score is used to decide something.

    Shared evidence is therefore counted more than once, and the result sits
    above whatever an independent combination would give.  The size of that
    effect was measured rather than assumed: collapsing the four detectors that
    share the residual to their single strongest member changes the combined
    score on 0.25% of assessments (247 of 99,200), by 0.05 on average and by at
    most 0.21 where it bites, and moves 20 of them across the trust deadband.
    Grouping is deliberately not applied, because the only consumer of this
    number is :func:`update_trust`, which compares it against fixed thresholds;
    the grouping would shift every one of those thresholds for a gain that is
    invisible on 99.75% of assessments.  What the code owes the reader is an
    accurate description, which is a monotone evidence score, not a fault
    probability.
    """
    # `remaining` is the running product of the complements.  It is the
    # arithmetic of a noisy-OR, but it is not read as P(no fault): see above.
    remaining = 1.0
    for name, score in scores.items():
        w = DETECTOR_WEIGHTS.get(name, 0.5)
        remaining *= (1.0 - max(0.0, min(1.0, score)) * w)
    return float(1.0 - remaining)


def status_for_trust(trust: float) -> str:
    if trust >= TRUST_SUSPECT:
        return STATUS_OK
    if trust >= TRUST_DEGRADED:
        return STATUS_SUSPECT
    if trust >= TRUST_FAILED:
        return STATUS_DEGRADED
    return STATUS_FAILED


_STATUS_ORDER = (STATUS_OK, STATUS_SUSPECT, STATUS_DEGRADED, STATUS_FAILED)


def status_for(trust: float, reliability: float = 1.0) -> str:
    """
    Reported maintenance state: the worse of the trust and reliability verdicts.

    ``reliability`` only ever *floors* the status at ``suspect`` or ``degraded``
    -- never at ``failed`` -- because a channel that is delivering good data
    right now must keep contributing to the priority even while it is queued for
    a technician visit.
    """
    from_trust = status_for_trust(trust)
    if reliability < RELIABILITY_DEGRADED:
        from_reliability = STATUS_DEGRADED
    elif reliability < RELIABILITY_SUSPECT:
        from_reliability = STATUS_SUSPECT
    else:
        from_reliability = STATUS_OK
    return max(from_trust, from_reliability, key=_STATUS_ORDER.index)


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
    statistical detector fires occasionally on healthy data, and if recovery is
    slower than that arrival rate, trust ratchets monotonically downwards until
    every healthy channel is marked suspect.  A genuine fault fires on
    essentially every cycle, so it still drives trust to zero within a few
    assessments; only the isolated blips are forgiven.

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


def update_reliability(previous: float, hard_evidence: float) -> float:
    """
    Slow, *integrating* memory of unambiguous faults; floors the reported status.

    Only NaNs, out-of-range values and frozen registers feed this, so it cannot
    be triggered by a statistical false alarm -- on a clean fleet its input is
    identically zero and the term is inert, which is what makes it safe to give
    it a long memory.  What it buys is honesty about intermittent hardware: a
    radio that drops a burst of packets every few hours is a failing radio, and
    a monitor that reports ``ok`` in the gaps between bursts is the reason such
    nodes stay in the field until they die completely.

    It integrates rather than latching on a threshold because that is the
    difference between "one packet went missing" and "this link is failing":
    each cycle debits the evidence seen and credits a fixed recovery, so a
    single lost sample costs a little and a link losing samples every window
    cannot climb back at all.  Full recovery from a total failure takes
    ``1 / RELIABILITY_RECOVERY`` clean cycles -- half a day at the default
    30-minute cadence, which is the interval an operator would want a node to
    prove itself over before it is called healthy again.
    """
    return float(min(1.0, max(0.0, previous + RELIABILITY_RECOVERY
                              - float(max(0.0, hard_evidence)))))


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
                previous_trust: Optional[Dict[str, Any]] = None,
                fleet: Optional[Dict[str, FleetStats]] = None,
                correct_drift: bool = True) -> Dict[str, ChannelAssessment]:
    """
    Assess every channel of one node.

    Parameters
    ----------
    history
        ``{channel: [oldest, ..., newest]}``.  The final element is the current
        sample; ``NaN`` marks a genuinely missing reading.
    previous_trust
        State carried over from the last cycle, so the assessment is recursive
        rather than recomputed from scratch.  Values may be a bare trust float
        (the format the first version persisted) or a :class:`ChannelState`
        dict; both are accepted so a running deployment upgrades in place.
    fleet
        ``{channel: FleetStats}`` holding the leave-one-out fleet reference and
        the distributions of every fleet-relative statistic.  Omit it for a
        single-node deployment: the detectors fall back to absolute tests, which
        are weaker but still meaningful.
    correct_drift
        When a drift is detected with a confident magnitude estimate, subtract it
        from the value handed downstream instead of discarding the channel.

    Returns
    -------
    ``{channel: ChannelAssessment}``.  Consumers should use ``value`` (which is
    ``None`` when the channel must be dropped) together with ``trust`` as the
    renormalisation weight, and feed ``state`` back in on the next cycle.
    """
    previous_trust = previous_trust or {}
    fleet = fleet or {}
    out: Dict[str, ChannelAssessment] = {}

    current: Dict[str, Optional[float]] = {}
    arrays: Dict[str, np.ndarray] = {}
    for channel, series in history.items():
        arr = np.asarray(list(series), dtype=float)
        arrays[channel] = arr
        last = arr[-1] if arr.size else np.nan
        current[channel] = float(last) if np.isfinite(last) else None

    physics = _score_physics(current)

    for channel, arr in arrays.items():
        stats = fleet.get(channel) or FleetStats()
        state = ChannelState.coerce(previous_trust.get(channel, 1.0))
        value = current[channel]
        residual = _residual_series(arr, stats.reference)

        missing_streak = 0
        for v in arr[::-1]:
            if np.isfinite(v):
                break
            missing_streak += 1

        stuck_score, stuck_streak = _score_stuck(arr, channel)
        drift_score, drift_magnitude = _score_drift(
            residual, channel, stats.peer_trends,
            have_reference=stats.reference is not None)
        own_level = (float(np.nanmedian(residual))
                     if np.isfinite(residual).any() else None)

        raw_scores = {
            "range": _score_range(value, channel),
            "missing": _score_missing(missing_streak, window_size=int(arr.size),
                                      missing_count=int((~np.isfinite(arr)).sum())),
            "stuck": stuck_score,
            "spike": _score_spike(residual, channel),
            "drift": drift_score,
            "dispersion": _score_dispersion(residual, channel,
                                            stats.peer_dispersions),
            "redundancy": _score_redundancy(stats.redundancy_z),
            "physics": physics.get(channel, 0.0),
            "peer": _score_peer(own_level, stats.peer_levels, channel,
                                has_redundancy=stats.has_redundancy),
        }

        # Confirmation gating mutates `state.confirm` in place, so the memory is
        # carried forward whether or not the detector fired this cycle.
        believed = {name: _confirmation_gate(name, score, state.confirm)
                    for name, score in raw_scores.items()}

        anomaly = _combine(believed)
        trust = update_trust(state.trust, anomaly)
        hard_evidence = max(raw_scores[name] for name in DETERMINISTIC_DETECTORS)
        reliability = update_reliability(state.reliability, hard_evidence)
        status = status_for(trust, reliability)

        flags = [name for name, s in raw_scores.items() if s > 0.15]

        # --- decide what value, if any, to pass downstream ---------------
        usable: Optional[float] = value
        if value is None:
            usable = None
        elif trust < TRUST_FAILED:
            # Keyed on trust, not on status: the reliability floor is a
            # maintenance signal and must never discard a usable reading.
            usable = None                     # drop entirely; weights renormalise
        elif raw_scores["range"] > 0.0:
            lo, hi = CHANNEL_RANGE.get(channel, (-math.inf, math.inf))
            usable = float(min(max(value, lo), hi))
            flags.append("clamped")
        if usable is not None and correct_drift and believed["drift"] > 0.35:
            usable = float(usable - drift_magnitude)
            lo, hi = CHANNEL_RANGE.get(channel, (-math.inf, math.inf))
            usable = float(min(max(usable, lo), hi))
            flags.append("drift_corrected")

        state.trust = trust
        state.reliability = reliability
        out[channel] = ChannelAssessment(
            channel=channel,
            status=status,
            trust=trust,
            value=usable,
            raw_value=value,
            drift_estimate=drift_magnitude if drift_score > 0.0 else 0.0,
            stuck_streak=stuck_streak,
            missing_streak=missing_streak,
            reliability=reliability,
            scores=raw_scores,
            flags=sorted(set(flags)),
            state=state.as_dict(),
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
                 previous_trust: Optional[Dict[int, Dict[str, Any]]] = None,
                 correct_drift: bool = True
                 ) -> Dict[int, Dict[str, ChannelAssessment]]:
    """
    Assess every node, using leave-one-out fleet references for every test.

    Fleet statistics are computed once for the whole network and the node's own
    contribution is removed per node, rather than rebuilding the sample for each
    node, which would be quadratic in the network size.
    """
    previous_trust = previous_trust or {}
    node_ids = list(node_histories.keys())
    if not node_ids:
        return {}

    channels = sorted({c for hist in node_histories.values() for c in hist})

    # --- fleet reference series: element-wise median across the network ---
    # Right-aligned, because the newest sample is the one being judged and nodes
    # may hold windows of different length.
    reference: Dict[str, Optional[np.ndarray]] = {}
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
        # A single-node (or pair) deployment has no fleet to compare against;
        # leave the reference undefined so each node falls back to its own centre.
        reference[channel] = median if len(node_ids) >= 3 else None

    # --- per-node residual statistics, for the leave-one-out fleet tests ---
    # `level` drives the peer test (a persistent offset), `trend` the drift test
    # (a step-immune change in that offset) and `dispersion` the noise test.
    level: Dict[str, np.ndarray] = {}
    trend: Dict[str, np.ndarray] = {}
    dispersion: Dict[str, np.ndarray] = {}
    for channel, matrix in stacked.items():
        ref = reference.get(channel)
        blank = np.full(len(node_ids), np.nan)
        if ref is None:
            level[channel] = blank.copy()
            trend[channel] = blank.copy()
            dispersion[channel] = blank.copy()
            continue
        residuals = matrix - ref[None, :]
        level[channel] = _row_nanmedian(residuals)
        trend[channel] = np.array([robust_trend(residuals[i])
                                   for i in range(residuals.shape[0])], dtype=float)
        dispersion[channel] = np.array(
            [increment_dispersion(residuals[i], channel)
             for i in range(residuals.shape[0])], dtype=float)

    # --- cross-channel analytical redundancy, fitted on the whole fleet ---
    # The window median is the level each channel is currently sitting at; it is
    # far less noisy than the latest sample and unaffected by a single dropout.
    levels_raw = {channel: _row_nanmedian(matrix)
                  for channel, matrix in stacked.items()}
    redundancy, modelled = _redundancy_scores(levels_raw, list(stacked))

    enough_peers = len(node_ids) >= MIN_FLEET_FOR_PEERS
    results = {}
    for index, node_id in enumerate(node_ids):
        node_fleet: Dict[str, FleetStats] = {}
        for channel in stacked:
            stats = FleetStats(reference=reference.get(channel))
            if enough_peers:
                for attribute, source in (("peer_levels", level),
                                          ("peer_trends", trend),
                                          ("peer_dispersions", dispersion)):
                    others = np.delete(source[channel], index)
                    others = others[np.isfinite(others)]
                    if others.size:
                        setattr(stats, attribute, others.tolist())
            if channel in redundancy:
                stats.redundancy_z = float(redundancy[channel][index])
            stats.has_redundancy = channel in modelled
            node_fleet[channel] = stats
        results[node_id] = assess_node(
            node_histories[node_id],
            previous_trust=previous_trust.get(node_id),
            fleet=node_fleet,
            correct_drift=correct_drift,
        )
    return results


def carry_state(results: Dict[int, Dict[str, ChannelAssessment]]
                ) -> Dict[int, Dict[str, Dict]]:
    """
    Extract the recursive state to feed into the next assessment.

    Callers must use this rather than harvesting ``assessment.trust``: the trust
    scalar is only part of the memory, and dropping the rest resets every
    detector's confirmation counter on every cycle, which is precisely what
    makes an isolated statistical alarm indistinguishable from a real fault.
    """
    return {node_id: {channel: dict(assessment.state)
                      for channel, assessment in channels.items()}
            for node_id, channels in results.items()}


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
