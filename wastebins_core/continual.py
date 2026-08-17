"""
Continual learning under a hard serving budget.
===============================================

A deployed bin network is non-stationary: sensors age, collection schedules
change, new bins appear, and seasons shift waste composition.  A model frozen at
training time therefore decays.  Retraining the gradient-boosting ensemble on
every batch of telemetry would fix that but is far too expensive to sit on the
request path -- and the operator explicitly must not pay a training cost while
using the application.

The design separates the two timescales:

**Slow path (offline, explicit).**
    The gradient-boosting bundle is trained by a management command.  It is never
    fitted implicitly during a request.

**Fast path (online, bounded).**
    A small linear model learns the *residual* of the frozen base model by
    stochastic gradient descent.  One update touches at most
    ``max_samples_per_update`` rows and is abandoned if it exceeds
    ``max_ms_per_update``, so the worst case is bounded by construction rather
    than by hope.  The final prediction is ``base(x) + residual(x)``.

Why a residual learner rather than replacing the model
------------------------------------------------------
The base ensemble captures the stable non-linear structure; the drift that
actually accumulates in the field is closer to a slowly-changing bias (a sensor
reads 4% high, a neighbourhood's collection interval changes). A linear residual
corrector captures that cheaply, cannot catastrophically destroy the base model,
and can be switched off instantly if it stops helping -- which the module checks
continuously by tracking its own prequential error against the base model's.

Drift handling
--------------
* **Detection** -- an ADWIN-style adaptive window over the prequential absolute
  error, plus a Page-Hinkley test.  Both are online and O(1) amortised.
* **Response** -- on a confirmed drift the corrector's learning rate is reset to
  its initial value and it is refitted from the replay buffer, so it re-adapts
  in one step rather than crawling there over hundreds of samples.
* **Forgetting control** -- a reservoir replay buffer keeps a bounded,
  representative sample of history so re-fits do not overfit the newest regime.
* **Escalation** -- persistent drift raises ``needs_retrain``.  The service
  surfaces that as an operator alert; it never triggers a batch fit by itself.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Drift detectors
# ---------------------------------------------------------------------------
class PageHinkley:
    """
    Online change detector on a stream of non-negative errors.

    Both the tolerated drift ``delta`` and the alarm ``threshold`` are expressed
    in multiples of the stream's own running standard deviation rather than in
    absolute units.  A fixed absolute threshold is only meaningful if you already
    know the error scale, and here that scale differs by orders of magnitude
    between the time-to-overflow head (hours) and the hazard head (probability),
    and changes again the moment the model is retrained.

    Threshold calibration
    ---------------------
    The threshold governs the false-alarm rate *per excursion*, not per stream.
    Over a long stream there are many excursions, so a value that looks safe for
    one is not safe for a thousand samples.  At the original 8.0 the detector
    fired on 198 of 200 stationary streams of 1000 samples, which is not a
    detector.  Swept on stationary streams and on streams with a step change,
    120 streams per point:

        threshold   false alarms   detects d=1   detects d=3
              8.0           0.99          1.00          1.00
             15.0           0.33          1.00          1.00
             25.0           0.00          1.00          1.00
             40.0           0.00          1.00          1.00

    25.0 removes the false alarms and costs no sensitivity at either shift size,
    so that is the default.  ADWIN, which runs alongside, fired on 0 of 200 of
    the same stationary streams and was never the problem.

    The practical consequence of the old value was contained rather than
    harmless: the corrector's activation gate meant the spurious alarms did not
    degrade predictions, so the do-no-harm property held for a reason unrelated
    to detection quality.  The alarms were still wrong.
    """

    def __init__(self, delta_sigma: float = 0.15, threshold_sigma: float = 25.0,
                 alpha: float = 0.9999, warmup: int = 40):
        self.delta_sigma = float(delta_sigma)
        self.threshold_sigma = float(threshold_sigma)
        self.alpha = float(alpha)
        self.warmup = int(warmup)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0

    @property
    def sigma(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(max(self.m2 / (self.n - 1), 0.0))

    def update(self, error: float) -> bool:
        e = float(error)
        self.n += 1
        delta_mean = e - self.mean
        self.mean += delta_mean / self.n
        self.m2 += delta_mean * (e - self.mean)

        sigma = self.sigma
        if self.n < self.warmup or sigma <= 1e-9:
            return False

        self.cumulative = (self.cumulative * self.alpha
                           + (e - self.mean - self.delta_sigma * sigma))
        self.minimum = min(self.minimum, self.cumulative)
        return (self.cumulative - self.minimum) > self.threshold_sigma * sigma


class AdaptiveWindow:
    """
    ADWIN2 detector over the error stream (Bifet & Gavalda 2007).

    Maintains exponentially-sized buckets and flags drift when the means of a
    prefix and a suffix of the window differ by more than a statistical bound.

    The bound is the **variance-aware** ADWIN2 cut, not the plain Hoeffding
    bound.  That distinction matters here: the plain bound assumes observations
    live in [0, 1], while the quantity being monitored is an unbounded
    regression error.  Feeding an error of magnitude 5 into a bound derived for
    unit-range data makes the detector fire constantly on a perfectly stable
    stream, which is exactly the failure it is supposed to catch elsewhere.

        eps_cut = sqrt( (2/m) * var_W * ln(2/delta') ) + (2/(3m)) * R * ln(2/delta')

    with ``m`` the harmonic mean of the two sub-window sizes, ``var_W`` the
    window variance, ``R`` the observed range, and ``delta' = delta / ln(W)``.
    Every term scales with the data, so the detector is unit-free.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 40,
                 min_window: int = 60):
        self.delta = float(delta)
        self.max_buckets = int(max_buckets)
        self.min_window = int(min_window)
        # buckets of (sum, sum_of_squares, count)
        self.buckets: List[Tuple[float, float, int]] = []
        # Samples still to observe before another detection may fire.  Without
        # it, halving the window on a detection leaves a window that trips the
        # same test again on the very next sample, producing a burst of
        # duplicate "drifts" for a single real change.
        self._cooldown = 0

    @property
    def width(self) -> int:
        return sum(c for _, _, c in self.buckets)

    @property
    def mean(self) -> float:
        total = self.width
        return sum(s for s, _, _ in self.buckets) / total if total else 0.0

    @property
    def variance(self) -> float:
        n = self.width
        if n < 2:
            return 0.0
        total = sum(s for s, _, _ in self.buckets)
        total_sq = sum(q for _, q, _ in self.buckets)
        return max(0.0, total_sq / n - (total / n) ** 2)

    def reset(self) -> None:
        self.buckets = []
        self._cooldown = 0

    def update(self, value: float) -> bool:
        v = float(value)
        self.buckets.append((v, v * v, 1))
        if self._cooldown > 0:
            self._cooldown -= 1

        # Merge equal-sized adjacent buckets to keep the summary compact.
        i = len(self.buckets) - 1
        while i > 0 and self.buckets[i][2] == self.buckets[i - 1][2]:
            s1, q1, c1 = self.buckets.pop(i)
            s0, q0, c0 = self.buckets.pop(i - 1)
            self.buckets.insert(i - 1, (s0 + s1, q0 + q1, c0 + c1))
            i -= 1
        while len(self.buckets) > self.max_buckets:
            self.buckets.pop(0)

        total_n = self.width
        if total_n < self.min_window or self._cooldown > 0:
            return False

        var_w = self.variance
        # Scale proxy for the ADWIN2 range term.  Deriving it from the window
        # variance keeps the detector unit-free without tracking extremes, which
        # would otherwise collapse to near-zero right after a window halving and
        # make the bound spuriously tight.
        observed_range = max(1e-9, 6.0 * math.sqrt(var_w))
        delta_prime = self.delta / max(math.log(max(total_n, 3)), 1e-9)
        log_term = math.log(2.0 / max(delta_prime, 1e-12))

        total_sum = sum(s for s, _, _ in self.buckets)
        left_sum = 0.0
        left_n = 0
        for s, _, c in self.buckets[:-1]:
            left_sum += s
            left_n += c
            right_n = total_n - left_n
            if left_n < 10 or right_n < 10:
                continue
            m = 1.0 / (1.0 / left_n + 1.0 / right_n)
            eps_cut = (math.sqrt((2.0 / m) * var_w * log_term)
                       + (2.0 / (3.0 * m)) * observed_range * log_term)
            right_sum = total_sum - left_sum
            if abs(left_sum / left_n - right_sum / right_n) > eps_cut:
                # Drop the stale prefix and report a change.
                keep = max(1, len(self.buckets) // 2)
                self.buckets = self.buckets[-keep:]
                self._cooldown = self.min_window
                return True
        return False


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReservoirBuffer:
    """
    Uniform reservoir sample of the stream, with a recency-biased tail.

    A pure reservoir forgets the current regime; a pure recency window forgets
    everything else.  Keeping both halves lets a re-fit see the new regime
    without discarding the structure it learned earlier.
    """

    def __init__(self, capacity: int = 2000, recent_fraction: float = 0.35,
                 seed: int = 42):
        self.capacity = int(capacity)
        self.recent_capacity = max(1, int(capacity * recent_fraction))
        self.reservoir_capacity = max(1, self.capacity - self.recent_capacity)
        self.rng = np.random.default_rng(seed)
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._recent_X: List[np.ndarray] = []
        self._recent_y: List[float] = []
        self.seen = 0

    def add(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=float).ravel()
        self.seen += 1

        self._recent_X.append(x)
        self._recent_y.append(float(y))
        if len(self._recent_X) > self.recent_capacity:
            self._recent_X.pop(0)
            self._recent_y.pop(0)

        if len(self._X) < self.reservoir_capacity:
            self._X.append(x)
            self._y.append(float(y))
        else:
            j = int(self.rng.integers(0, self.seen))
            if j < self.reservoir_capacity:
                self._X[j] = x
                self._y[j] = float(y)

    def __len__(self) -> int:
        return len(self._X) + len(self._recent_X)

    def sample(self, n: Optional[int] = None):
        if not len(self):
            return np.zeros((0, 1)), np.zeros(0)
        X = np.vstack([np.asarray(self._X + self._recent_X, dtype=float)])
        y = np.asarray(self._y + self._recent_y, dtype=float)
        if n is not None and n < len(y):
            idx = self.rng.choice(len(y), size=int(n), replace=False)
            return X[idx], y[idx]
        return X, y


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
@dataclass
class Budget:
    max_samples_per_update: int = 256
    max_ms_per_update: float = 40.0
    min_seconds_between_updates: float = 15.0
    enabled: bool = True

    @classmethod
    def from_settings(cls, cfg: Dict) -> "Budget":
        return cls(
            max_samples_per_update=int(cfg.get("CONTINUAL_MAX_SAMPLES_PER_UPDATE", 256)),
            max_ms_per_update=float(cfg.get("CONTINUAL_MAX_MS_PER_UPDATE", 40.0)),
            min_seconds_between_updates=float(
                cfg.get("CONTINUAL_MIN_SECONDS_BETWEEN_UPDATES", 15.0)),
            enabled=bool(cfg.get("CONTINUAL_ENABLED", True)),
        )


@dataclass
class UpdateReport:
    applied: bool
    reason: str = ""
    n_samples: int = 0
    elapsed_ms: float = 0.0
    drift_detected: bool = False
    corrector_active: bool = False
    prequential_mae_base: float = 0.0
    prequential_mae_corrected: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "n_samples": self.n_samples,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "drift_detected": self.drift_detected,
            "corrector_active": self.corrector_active,
            "prequential_mae_base": round(self.prequential_mae_base, 5),
            "prequential_mae_corrected": round(self.prequential_mae_corrected, 5),
        }


# ---------------------------------------------------------------------------
# The learner
# ---------------------------------------------------------------------------
class ContinualResidualLearner:
    """
    Bounded online corrector on top of a frozen base regressor.

    The corrector is only applied when it is *earning its place*: its running
    prequential error must beat the base model's by a margin, otherwise
    predictions pass through untouched.  That guard is what makes it safe to run
    unattended.
    """

    def __init__(self, n_features: int,
                 budget: Optional[Budget] = None,
                 learning_rate: float = 0.02,
                 l2: float = 1e-4,
                 buffer_capacity: int = 2000,
                 activation_margin: float = 0.02,
                 warmup_samples: int = 60,
                 seed: int = 42):
        self.n_features = int(n_features)
        self.budget = budget or Budget()
        self.initial_learning_rate = float(learning_rate)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.activation_margin = float(activation_margin)
        self.warmup_samples = int(warmup_samples)

        self.weights = np.zeros(self.n_features, dtype=float)
        self.bias = 0.0
        # Running feature standardisation, so one badly-scaled column cannot
        # dominate the gradient.
        self.feature_mean = np.zeros(self.n_features, dtype=float)
        self.feature_var = np.ones(self.n_features, dtype=float)
        self.n_seen = 0

        self.buffer = ReservoirBuffer(buffer_capacity, seed=seed)
        self.page_hinkley = PageHinkley()
        self.adwin = AdaptiveWindow()

        self.err_base = 0.0          # exponentially weighted prequential MAE
        self.err_corrected = 0.0
        self.err_decay = 0.995
        self.drift_events = 0
        self.updates_applied = 0
        self.updates_skipped = 0
        self.needs_retrain = False
        self.last_update_ts = 0.0
        self.last_drift_ts: Optional[float] = None
        # Sample positions of recent drifts.  Escalation to "retrain the base
        # model" requires several drifts *close together*: isolated detections
        # over a long stream are normal and the corrector absorbs them, so
        # escalating on a lifetime count would cry wolf on every deployment.
        self._drift_positions: List[int] = []
        self.retrain_window = 800
        self.retrain_threshold = 3
        # Share of the error the linear corrector must be removing before the
        # base ensemble is judged stale enough to warrant an offline retrain.
        self.staleness_improvement = 0.25

    # -- scaling ---------------------------------------------------------
    def _observe_scale(self, x: np.ndarray) -> None:
        self.n_seen += 1
        delta = x - self.feature_mean
        self.feature_mean += delta / self.n_seen
        self.feature_var += (delta * (x - self.feature_mean) - self.feature_var) / self.n_seen
        np.maximum(self.feature_var, 1e-8, out=self.feature_var)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feature_mean) / np.sqrt(self.feature_var)

    # -- prediction ------------------------------------------------------
    @property
    def active(self) -> bool:
        """Whether the corrector currently improves on the base model."""
        if self.n_seen < self.warmup_samples:
            return False
        return self.err_corrected < self.err_base * (1.0 - self.activation_margin)

    def residual(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.n_features:
            return 0.0
        return float(np.dot(self.weights, self._scale(x)) + self.bias)

    def predict(self, x: np.ndarray, base_prediction: float) -> float:
        if not self.active:
            return float(base_prediction)
        return float(base_prediction + self.residual(x))

    # -- learning --------------------------------------------------------
    def _sgd_step(self, x_scaled: np.ndarray, target_residual: float) -> None:
        prediction = float(np.dot(self.weights, x_scaled) + self.bias)
        error = prediction - target_residual
        self.weights -= self.learning_rate * (error * x_scaled + self.l2 * self.weights)
        self.bias -= self.learning_rate * error

    def observe(self, x: np.ndarray, y_true: float, base_prediction: float) -> bool:
        """
        Record one supervised outcome and take a single gradient step.

        Returns whether a drift was detected on this sample.  This is O(d) and
        costs microseconds, so it is safe to call from the ingestion path.
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.n_features:
            return False

        self._observe_scale(x)
        x_scaled = self._scale(x)

        target_residual = float(y_true) - float(base_prediction)
        corrected = float(base_prediction) + float(np.dot(self.weights, x_scaled) + self.bias)

        err_b = abs(float(y_true) - float(base_prediction))
        err_c = abs(float(y_true) - corrected)
        self.err_base = self.err_decay * self.err_base + (1 - self.err_decay) * err_b
        self.err_corrected = self.err_decay * self.err_corrected + (1 - self.err_decay) * err_c

        self.buffer.add(x, float(y_true) - float(base_prediction))
        self._sgd_step(x_scaled, target_residual)

        drifted = self.adwin.update(err_b) or self.page_hinkley.update(err_b)
        if drifted:
            self._on_drift()
        elif self.n_seen % 50 == 0:
            # Staleness is a slow property; checking it periodically catches the
            # case where the base model decays without tripping a change alarm.
            self._evaluate_staleness()
        return drifted

    def _on_drift(self) -> None:
        self.drift_events += 1
        self.last_drift_ts = time.time()
        # Re-adapt immediately: restore the initial step size and re-fit the
        # corrector from the replay buffer rather than crawling back by SGD.
        self.learning_rate = self.initial_learning_rate
        self.page_hinkley.reset()
        self._refit_from_buffer()

        self._drift_positions.append(self.n_seen)
        cutoff = self.n_seen - self.retrain_window
        self._drift_positions = [p for p in self._drift_positions if p >= cutoff]
        self._evaluate_staleness()

    def _evaluate_staleness(self) -> None:
        """
        Decide whether the *base* model needs an offline retrain.

        Counting drift alarms is the wrong signal: an alarm only says the error
        stream changed, and a healthy system will raise some over a long
        deployment.  What actually matters is whether the frozen base model has
        become materially wrong -- which shows up as the tiny linear corrector
        carrying a large share of the accuracy.  When a linear residual on top of
        a gradient-boosting ensemble improves error by more than
        ``staleness_improvement`` for a sustained period, the ensemble itself is
        out of date and should be refitted offline.
        """
        if self.n_seen < max(self.warmup_samples * 3, 200):
            return
        if self.err_base <= 1e-9:
            return
        # ``err_*`` are already exponentially smoothed, so this is a sustained
        # measurement rather than a single-sample verdict.
        improvement = 1.0 - (self.err_corrected / self.err_base)
        if improvement >= self.staleness_improvement:
            self.needs_retrain = True

    def _refit_from_buffer(self) -> None:
        X, y = self.buffer.sample(self.budget.max_samples_per_update)
        if X.shape[0] < 10 or X.shape[1] != self.n_features:
            return
        Xs = (X - self.feature_mean) / np.sqrt(self.feature_var)
        # Closed-form ridge on a bounded sample: exact, and far cheaper than an
        # iterative fit at this size.
        A = Xs.T @ Xs + self.l2 * len(y) * np.eye(self.n_features)
        b = Xs.T @ (y - y.mean())
        try:
            self.weights = np.linalg.solve(A, b)
            self.bias = float(y.mean())
        except np.linalg.LinAlgError:
            pass

    def update(self, batch: Sequence[Tuple[np.ndarray, float, float]]) -> UpdateReport:
        """
        Apply a bounded batch of ``(features, y_true, base_prediction)`` triples.

        Refuses to run when disabled, when called too soon after the previous
        update, and abandons the batch the moment it exceeds its millisecond
        budget -- so this can never become the reason a request is slow.
        """
        if not self.budget.enabled:
            return UpdateReport(applied=False, reason="disabled")

        now = time.monotonic()
        if (now - self.last_update_ts) < self.budget.min_seconds_between_updates:
            self.updates_skipped += 1
            return UpdateReport(applied=False, reason="rate_limited")

        started = time.perf_counter()
        drift = False
        n = 0
        for x, y_true, base_pred in list(batch)[: self.budget.max_samples_per_update]:
            drift = self.observe(x, y_true, base_pred) or drift
            n += 1
            if (time.perf_counter() - started) * 1000.0 > self.budget.max_ms_per_update:
                break

        self.last_update_ts = now
        self.updates_applied += 1
        elapsed = (time.perf_counter() - started) * 1000.0
        return UpdateReport(
            applied=True, reason="ok", n_samples=n, elapsed_ms=elapsed,
            drift_detected=drift, corrector_active=self.active,
            prequential_mae_base=self.err_base,
            prequential_mae_corrected=self.err_corrected,
        )

    # -- persistence -----------------------------------------------------
    def state_dict(self) -> Dict:
        return {
            "n_features": self.n_features,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "feature_mean": self.feature_mean.tolist(),
            "feature_var": self.feature_var.tolist(),
            "n_seen": self.n_seen,
            "err_base": self.err_base,
            "err_corrected": self.err_corrected,
            "learning_rate": self.learning_rate,
            "drift_events": self.drift_events,
            "updates_applied": self.updates_applied,
            "updates_skipped": self.updates_skipped,
            "needs_retrain": self.needs_retrain,
        }

    def load_state_dict(self, state: Dict) -> None:
        if int(state.get("n_features", self.n_features)) != self.n_features:
            return
        self.weights = np.asarray(state.get("weights", self.weights), dtype=float)
        self.bias = float(state.get("bias", 0.0))
        self.feature_mean = np.asarray(state.get("feature_mean", self.feature_mean), dtype=float)
        self.feature_var = np.asarray(state.get("feature_var", self.feature_var), dtype=float)
        self.n_seen = int(state.get("n_seen", 0))
        self.err_base = float(state.get("err_base", 0.0))
        self.err_corrected = float(state.get("err_corrected", 0.0))
        self.learning_rate = float(state.get("learning_rate", self.initial_learning_rate))
        self.drift_events = int(state.get("drift_events", 0))
        self.updates_applied = int(state.get("updates_applied", 0))
        self.updates_skipped = int(state.get("updates_skipped", 0))
        self.needs_retrain = bool(state.get("needs_retrain", False))

    def status(self) -> Dict:
        return {
            "active": self.active,
            "samples_seen": self.n_seen,
            "buffer_size": len(self.buffer),
            "drift_events": self.drift_events,
            "updates_applied": self.updates_applied,
            "updates_skipped": self.updates_skipped,
            "needs_retrain": self.needs_retrain,
            "prequential_mae_base": round(self.err_base, 5),
            "prequential_mae_corrected": round(self.err_corrected, 5),
            "improvement_pct": round(
                100.0 * (1.0 - self.err_corrected / self.err_base), 2)
            if self.err_base > 1e-9 else 0.0,
            "budget": {
                "enabled": self.budget.enabled,
                "max_samples_per_update": self.budget.max_samples_per_update,
                "max_ms_per_update": self.budget.max_ms_per_update,
                "min_seconds_between_updates": self.budget.min_seconds_between_updates,
            },
        }


def prequential_evaluation(learner: ContinualResidualLearner,
                           X: np.ndarray, y: np.ndarray,
                           base_predictions: np.ndarray,
                           report_every: int = 200) -> Dict:
    """
    Test-then-train evaluation, the standard protocol for streaming learners.

    Each sample is predicted before it is learned from, so the reported error is
    an honest estimate of what the deployed system would have achieved as the
    stream arrived.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    base = np.asarray(base_predictions, dtype=float).ravel()

    err_base: List[float] = []
    err_model: List[float] = []
    curve: List[Dict] = []
    drift_points: List[int] = []

    for i in range(len(y)):
        prediction = learner.predict(X[i], base[i])
        err_base.append(abs(y[i] - base[i]))
        err_model.append(abs(y[i] - prediction))
        if learner.observe(X[i], y[i], base[i]):
            drift_points.append(i)
        if report_every and (i + 1) % report_every == 0:
            window = slice(max(0, i + 1 - report_every), i + 1)
            curve.append({
                "step": i + 1,
                "mae_base": round(float(np.mean(err_base[window])), 5),
                "mae_continual": round(float(np.mean(err_model[window])), 5),
                "corrector_active": learner.active,
            })

    mae_base = float(np.mean(err_base)) if err_base else 0.0
    mae_model = float(np.mean(err_model)) if err_model else 0.0
    return {
        "n": int(len(y)),
        "prequential_mae_base": round(mae_base, 5),
        "prequential_mae_continual": round(mae_model, 5),
        "improvement_pct": round(100.0 * (1.0 - mae_model / mae_base), 3)
        if mae_base > 1e-9 else 0.0,
        "drift_detections": len(drift_points),
        "drift_steps": drift_points[:50],
        "curve": curve,
        "protocol": "prequential (test-then-train); each sample predicted before it is learned",
    }
