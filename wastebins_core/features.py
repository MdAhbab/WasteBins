"""
Feature engineering: one specification, used everywhere.
========================================================

Training, offline evaluation and online serving all call into this module, so a
feature can never be computed one way in the experiment that produced the
published numbers and another way in the service that ships.  The previous code
base had exactly that problem: the deployed predictor padded its rolling
statistics with placeholder constants (``mean = current value``, ``std = 0``),
which silently moved every inference off the training manifold.

Design rules
------------
* **Past only.**  Every statistic is computed from samples at or before the
  prediction instant.  Nothing may look forward, or the reported skill is
  leakage rather than prediction.
* **Irregular sampling.**  Real telemetry arrives at uneven intervals, so trends
  are estimated by least squares against elapsed *hours*, not by differencing
  row positions.
* **Cyclical time.**  Hour-of-day and day-of-week enter as sine/cosine pairs so
  23:00 and 01:00 are neighbours rather than the extremes of a linear scale.
* **Explicit missingness.**  A gap is forward-filled for the rolling statistics
  but its age is exposed as ``staleness_h``, letting the model learn to discount
  stale evidence instead of treating it as fresh.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

# Channels the model consumes, in their canonical order.
CHANNELS = ("waste", "gas", "temp", "humidity")

# Rolling window length, in samples.  Matches the manuscript's stated window.
ROLL = 10

# Prediction targets.
HORIZON_H = 6.0        # hazard look-ahead
TTO_CAP_H = 24.0       # time-to-overflow is right-censored at this value
GAS_DANGER = 0.62      # latent gas level treated as a biological hazard
OVERFLOW_LEVEL = 1.0

# Tolerance, in hours, for deciding that a time-to-overflow label sits at the
# censoring cap.  Downstream code recognises a censored row with the same test
# (``y >= TTO_CAP_H - 1e-9``), so the constant is shared rather than repeated.
_CAP_TOL_H = 1e-9


def _statistic_names() -> List[str]:
    names: List[str] = []
    for ch in CHANNELS:
        names += [ch, f"mean_{ch}", f"std_{ch}", f"trend_{ch}", f"ewma_{ch}"]
    return names


FEATURE_COLS: List[str] = _statistic_names() + [
    "range_waste",        # peak-to-trough over the window: burstiness of arrivals
    "accel_waste",        # change in fill rate: is the bin filling faster than before?
    "gas_per_fill",       # odour intensity normalised by how full the bin is
    "temp_excess",        # internal temperature above the window mean: decomposition
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "dwell_h",            # hours since the bin was last emptied
    "staleness_h",        # age of the most recent successful reading
    "n_observed",         # how many of the last ROLL samples were real
]

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "waste": "Current fill fraction reported by the ultrasonic sensor.",
    "mean_waste": "Mean fill over the recent window.",
    "std_waste": "Variability of fill over the window.",
    "trend_waste": "Fill rate in units per hour, by least squares over the window.",
    "ewma_waste": "Exponentially weighted fill, favouring the most recent samples.",
    "gas": "Current gas/odour level.",
    "mean_gas": "Mean gas level over the window.",
    "std_gas": "Variability of gas over the window.",
    "trend_gas": "Rate of change of gas per hour.",
    "ewma_gas": "Exponentially weighted gas level.",
    "temp": "Current internal temperature.",
    "mean_temp": "Mean internal temperature over the window.",
    "std_temp": "Temperature variability over the window.",
    "trend_temp": "Temperature change per hour.",
    "ewma_temp": "Exponentially weighted temperature.",
    "humidity": "Current relative humidity.",
    "mean_humidity": "Mean humidity over the window.",
    "std_humidity": "Humidity variability over the window.",
    "trend_humidity": "Humidity change per hour.",
    "ewma_humidity": "Exponentially weighted humidity.",
    "range_waste": "Peak-to-trough fill over the window, capturing bursty dumping.",
    "accel_waste": "Change in fill rate between the two halves of the window.",
    "gas_per_fill": "Gas level per unit fill: odour intensity independent of volume.",
    "temp_excess": "Current temperature above the window mean, indicating decomposition heat.",
    "hour_sin": "Sine encoding of hour-of-day.",
    "hour_cos": "Cosine encoding of hour-of-day.",
    "dow_sin": "Sine encoding of day-of-week.",
    "dow_cos": "Cosine encoding of day-of-week.",
    "dwell_h": "Hours since this bin was last emptied.",
    "staleness_h": "Age of the most recent successful reading.",
    "n_observed": "Number of real (non-imputed) samples in the window.",
}


@dataclass
class FeatureRow:
    values: Dict[str, float]

    def to_array(self, columns: Sequence[str] = FEATURE_COLS) -> np.ndarray:
        return np.array([[float(self.values.get(c, 0.0)) for c in columns]], dtype=float)


# ---------------------------------------------------------------------------
# Primitive statistics over an irregularly-sampled window
# ---------------------------------------------------------------------------
def _forward_fill(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    last = np.nan
    for i, v in enumerate(out):
        if np.isfinite(v):
            last = v
        else:
            out[i] = last
    # Any leading gap is back-filled from the first real observation.
    if not np.isfinite(out[0]):
        first_real = next((v for v in out if np.isfinite(v)), 0.0)
        for i in range(len(out)):
            if np.isfinite(out[i]):
                break
            out[i] = first_real
    return out


def _slope_per_hour(values: np.ndarray, hours: np.ndarray) -> float:
    """Least-squares slope against elapsed hours; robust to uneven sampling."""
    mask = np.isfinite(values) & np.isfinite(hours)
    x, y = hours[mask], values[mask]
    if x.size < 2:
        return 0.0
    span = float(x.max() - x.min())
    if span < 1e-6:
        return 0.0
    x_centred = x - x.mean()
    denom = float(np.dot(x_centred, x_centred))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x_centred, y - y.mean()) / denom)


def _ewma(values: np.ndarray, halflife: float = 3.0) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    decay = 0.5 ** (1.0 / max(halflife, 1e-6))
    weights = decay ** np.arange(v.size - 1, -1, -1, dtype=float)
    return float(np.dot(weights, v) / weights.sum())


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------
def build_row(window: Dict[str, Sequence[float]],
              hours: Sequence[float],
              timestamp_hour: float = 0.0,
              timestamp_dow: int = 0,
              dwell_h: float = 0.0,
              staleness_h: float = 0.0) -> FeatureRow:
    """
    Build one feature row from a per-channel window.

    ``window`` maps each channel in :data:`CHANNELS` to its recent samples,
    oldest first, with ``NaN`` for missing readings.  ``hours`` gives the elapsed
    hours of each sample relative to the oldest, so trends are per-hour rather
    than per-row.
    """
    hrs = np.asarray(list(hours), dtype=float)
    out: Dict[str, float] = {}
    observed_counts: List[int] = []

    for ch in CHANNELS:
        raw = np.asarray(list(window.get(ch, [])), dtype=float)
        if raw.size == 0:
            raw = np.array([np.nan])
        observed_counts.append(int(np.isfinite(raw).sum()))
        filled = _forward_fill(raw)
        if not np.isfinite(filled).any():
            filled = np.zeros_like(filled)

        h = hrs[-filled.size:] if hrs.size >= filled.size else np.arange(filled.size, dtype=float)

        out[ch] = float(filled[-1])
        out[f"mean_{ch}"] = float(np.mean(filled))
        out[f"std_{ch}"] = float(np.std(filled, ddof=1)) if filled.size > 1 else 0.0
        out[f"trend_{ch}"] = _slope_per_hour(filled, h)
        out[f"ewma_{ch}"] = _ewma(filled)

    waste = _forward_fill(np.asarray(list(window.get("waste", [0.0])), dtype=float))
    if not np.isfinite(waste).any():
        waste = np.zeros_like(waste)
    out["range_waste"] = float(np.max(waste) - np.min(waste)) if waste.size else 0.0

    # Fill acceleration: slope of the recent half minus slope of the earlier half.
    if waste.size >= 4:
        half = waste.size // 2
        h = hrs[-waste.size:] if hrs.size >= waste.size else np.arange(waste.size, dtype=float)
        out["accel_waste"] = (_slope_per_hour(waste[half:], h[half:])
                              - _slope_per_hour(waste[:half], h[:half]))
    else:
        out["accel_waste"] = 0.0

    out["gas_per_fill"] = float(out["gas"] / max(out["waste"], 0.05))
    out["temp_excess"] = float(out["temp"] - out["mean_temp"])

    hour = float(timestamp_hour) % 24.0
    dow = int(timestamp_dow) % 7
    out["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
    out["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
    out["dow_sin"] = float(np.sin(2 * np.pi * dow / 7.0))
    out["dow_cos"] = float(np.cos(2 * np.pi * dow / 7.0))

    out["dwell_h"] = float(max(0.0, dwell_h))
    out["staleness_h"] = float(max(0.0, staleness_h))
    out["n_observed"] = float(min(observed_counts) if observed_counts else 0)

    return FeatureRow(values={c: float(out.get(c, 0.0)) for c in FEATURE_COLS})


# ---------------------------------------------------------------------------
# Batch builder over a full per-bin time series
# ---------------------------------------------------------------------------
def build_matrix(series: Dict[str, Sequence[float]],
                 hours: Sequence[float],
                 dow: Optional[Sequence[int]] = None,
                 dwell: Optional[Sequence[float]] = None,
                 roll: int = ROLL) -> np.ndarray:
    """
    Build the full feature matrix for one bin's history.

    Row ``i`` uses only samples ``[max(0, i-roll+1) .. i]``, so the matrix can be
    handed straight to a temporally-split validation without leakage.
    """
    hrs = np.asarray(list(hours), dtype=float)
    n = hrs.size
    rows = np.zeros((n, len(FEATURE_COLS)), dtype=float)

    channel_arrays = {ch: np.asarray(list(series.get(ch, [])), dtype=float)
                      for ch in CHANNELS}
    for ch, arr in channel_arrays.items():
        if arr.size != n:
            padded = np.full(n, np.nan)
            padded[:min(n, arr.size)] = arr[:min(n, arr.size)]
            channel_arrays[ch] = padded

    for i in range(n):
        lo = max(0, i - roll + 1)
        window = {ch: arr[lo:i + 1] for ch, arr in channel_arrays.items()}
        hour_of_day = float(hrs[i] % 24.0)
        day = int(dow[i]) if dow is not None and i < len(dow) else int((hrs[i] // 24) % 7)
        dwell_h = float(dwell[i]) if dwell is not None and i < len(dwell) else 0.0

        waste_window = window["waste"]
        finite = np.flatnonzero(np.isfinite(waste_window))
        staleness = float(hrs[i] - hrs[lo + finite[-1]]) if finite.size else float(roll)

        rows[i] = build_row(window, hrs[lo:i + 1],
                            timestamp_hour=hour_of_day, timestamp_dow=day,
                            dwell_h=dwell_h, staleness_h=staleness
                            ).to_array()[0]
    return rows


# ---------------------------------------------------------------------------
# Forward-looking labels
# ---------------------------------------------------------------------------
def forward_labels(waste: Sequence[float], gas: Sequence[float], hours: Sequence[float],
                   horizon_h: float = HORIZON_H, tto_cap_h: float = TTO_CAP_H,
                   overflow_level: float = OVERFLOW_LEVEL,
                   gas_danger: float = GAS_DANGER):
    """
    Labels derived strictly from the *future* of the series.

    Returns ``(hazard_within_h, time_to_overflow, overflow_observed)``.

    ``hazard_within_h``   -- 1 when the bin overflows or the gas level crosses
                             the danger threshold within ``horizon_h`` hours.
    ``time_to_overflow``  -- hours until the fill level first reaches
                             ``overflow_level``.  When no such crossing is seen,
                             the value is the *censoring time*: how much future
                             the row actually had to be judged against, which is
                             ``tto_cap_h`` in the body of a record and less than
                             that near its end.
    ``overflow_observed`` -- 1 when ``time_to_overflow`` is a crossing that was
                             really seen, 0 when it is a censoring time.

    Censoring convention
    --------------------
    The pair (time, event) is the representation this code base already uses for
    a right-censored target: the row carries the time at which observation
    stopped, and an event flag says whether that time is an outcome or a
    stopping point.  The trainer's concordance index takes exactly that flag, so
    pairs whose earlier time is censored are excluded rather than guessed at.
    What used to be implicit is now returned, because the flag can be recovered
    from the time alone only while every row happens to be censored at the same
    cap, and that is true of a trimmed training set rather than of the function.

    Truncation at the end of a record
    ---------------------------------
    A record is a finite window.  A row that sits ``r`` hours from the end of it
    has only ``r`` hours of future to look at, so a bin that does not overflow
    inside them is censored at ``min(tto_cap_h, r)``, not at ``tto_cap_h``.
    Returning the cap there would assert a full day of observed safety on
    evidence that ran out long before, which is a fabricated label rather than a
    missing one, and it is fabricated in the long-horizon regime the planner
    leans on hardest.

    The same truncation reaches ``hazard_within_h``: a row with less than
    ``horizon_h`` of future cannot have a hazard refuted.  Those rows are
    identifiable as ``hazard_within_h == 0`` together with
    ``overflow_observed == 0`` and ``time_to_overflow < horizon_h``, so a caller
    that needs a trustworthy hazard label can drop them.  A caller wanting both
    targets fully observed should keep only the rows at least
    ``max(horizon_h, tto_cap_h)`` hours from the end of the record.

    The features never observe these quantities, so a good score is genuine
    forecasting skill rather than the self-consistency the original model
    measured by regressing its own priority formula.
    """
    w = np.asarray(list(waste), dtype=float)
    g = np.asarray(list(gas), dtype=float)
    h = np.asarray(list(hours), dtype=float)
    n = h.size
    hazard = np.zeros(n, dtype=int)
    tto = np.full(n, float(tto_cap_h), dtype=float)
    overflow_observed = np.zeros(n, dtype=int)
    if n == 0:
        return hazard, tto, overflow_observed

    # The end of the observation window, taken as a maximum rather than as the
    # final element so an unsorted or partly missing clock cannot shorten it.
    finite_hours = h[np.isfinite(h)]
    last_h = float(finite_hours.max()) if finite_hours.size else 0.0

    for i in range(n):
        dt = h - h[i]
        ahead = (dt > 0) & (dt <= horizon_h)
        if ahead.any():
            future_w = w[ahead]
            future_g = g[ahead]
            if (np.nanmax(future_w) >= overflow_level if future_w.size else False) or \
               (np.nanmax(future_g) >= gas_danger if future_g.size else False):
                hazard[i] = 1
        within = (dt >= 0) & (dt <= tto_cap_h) & (w >= overflow_level)
        if within.any():
            tto[i] = float(min(tto_cap_h, float(np.min(dt[within]))))
            overflow_observed[i] = 1
        else:
            # No crossing was seen, so the label is the length of the look-ahead
            # that produced that verdict.  In the body of a record this is the
            # cap and the value is unchanged; near the end it is shorter, and
            # saying so is the difference between "did not overflow within a
            # day" and "was not watched for a day".
            followup = last_h - float(h[i])
            if followup >= float(tto_cap_h) - _CAP_TOL_H:
                # A follow-up that reaches the cap is snapped onto it, using the
                # same tolerance downstream code uses to recognise a censored
                # row.  Without this, arithmetic on an irregular clock could
                # leave a full-horizon row a few nanoseconds short of the cap
                # and have it read as an observed overflow.
                tto[i] = float(tto_cap_h)
            else:
                tto[i] = max(0.0, followup)
    return hazard, tto, overflow_observed


def describe() -> Dict:
    """Machine-readable feature contract, published by the API and the paper."""
    return {
        "n_features": len(FEATURE_COLS),
        "columns": list(FEATURE_COLS),
        "descriptions": dict(FEATURE_DESCRIPTIONS),
        "window_samples": ROLL,
        "channels": list(CHANNELS),
        "targets": {
            "time_to_overflow": (f"hours until fill >= {OVERFLOW_LEVEL}, right-censored at "
                                 f"{TTO_CAP_H} h or at the end of the record, whichever "
                                 f"comes first"),
            "overflow_observed": ("1 when time_to_overflow is an observed crossing, "
                                  "0 when it is a censoring time"),
            "hazard_within_h": f"overflow or gas >= {GAS_DANGER} within {HORIZON_H} h",
        },
        "leakage_controls": [
            "all statistics computed from samples at or before the prediction instant",
            "trends fitted against elapsed hours, not row positions",
            "labels derived from the strictly future trajectory",
            "censoring reported rather than imputed: a row watched for less than "
            "the cap is censored at its own follow-up, not at the cap",
        ],
    }
