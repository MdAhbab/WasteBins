"""
Physically-grounded, reproducible smart-bin simulator.
=====================================================

Purpose
-------
The manuscript's Random Forest was trained against a *deterministic* priority
formula computed from the same sensor inputs that were also fed in as features
(see bins/utils/ai/train_model.py::_calculate_priority_score).  Regressing a
closed-form function of your own inputs yields an artificially high R^2 that
measures self-consistency, not predictive skill.

This module replaces that with an HONEST generative process:

  * Each bin has a LATENT fill/decomposition state that evolves in time with
    diurnal + weekly demand, random event bursts, temperature-driven
    decomposition (Arrhenius q10), and periodic collection.
  * Sensors OBSERVE that latent state with noise and dropouts.
  * The prediction TARGET is forward-looking (time-to-overflow / hazard within
    the next H hours), derived from the LATENT FUTURE trajectory that the model
    never observes.

Because the label depends on the future latent state and the features depend
only on the noisy observed past, a high score is genuine predictive skill.
Everything is clearly labelled as simulation and is fully seeded/reproducible.

Author: (regenerated for the SCS revision)
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bin geography (real coordinates around Mirpur / Dhaka, extended from the
# 6 bins hard-coded in send_dummy_data.py to a 30-node network).
# ---------------------------------------------------------------------------
_BASE_BINS = [
    ("Mirpur 10",       23.8069, 90.3687),
    ("Sony Square",     23.7910, 90.3550),
    ("Mirpur Stadium",  23.8050, 90.3630),
    ("Mirpur 11",       23.8203, 90.3650),
    ("Mirpur 14",       23.8100, 90.3780),
    ("Pallabi",         23.8250, 90.3650),
    ("Kazipara",        23.7980, 90.3720),
    ("Shewrapara",      23.7890, 90.3740),
    ("Agargaon",        23.7780, 90.3800),
    ("Kafrul",          23.7900, 90.3850),
]


def build_bins(n_bins: int = 30, seed: int = 7) -> pd.DataFrame:
    """Return a DataFrame of bins with coordinates and heterogeneous demand."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_bins):
        base = _BASE_BINS[i % len(_BASE_BINS)]
        # jitter coordinates for the synthetic extra nodes
        jitter = 0.0 if i < len(_BASE_BINS) else 0.012
        lat = base[1] + rng.uniform(-jitter, jitter)
        lng = base[2] + rng.uniform(-jitter, jitter)
        # per-bin heterogeneity
        arrival = rng.uniform(0.008, 0.030)         # latent fill added per hour (base)
        organic = rng.uniform(0.25, 0.85)           # organic fraction -> odor/gas potential
        collect_period = int(rng.choice([36, 48, 60, 72]))  # scheduled collection interval (h)
        rows.append(dict(
            bin_id=i,
            name=f"{base[0]}" + ("" if i < len(_BASE_BINS) else f"-{i}"),
            latitude=lat, longitude=lng,
            arrival=arrival, organic=organic, collect_period=collect_period,
            collect_phase=int(rng.integers(0, collect_period)),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Environmental drivers
# ---------------------------------------------------------------------------
def _ambient_temperature(t_hour: int, rng: np.random.Generator) -> float:
    """Diurnal + seasonal ambient temperature (deg C) for Dhaka-like climate."""
    day = t_hour / 24.0
    seasonal = 28.0 + 4.0 * math.sin(2 * math.pi * (day - 120) / 365.0)  # warm year-round
    diurnal = 5.0 * math.sin(2 * math.pi * (t_hour % 24 - 9) / 24.0)     # peak ~15:00
    return seasonal + diurnal + rng.normal(0, 1.2)


def _demand_multiplier(t_hour: int) -> float:
    """Diurnal + weekly waste-generation multiplier (dimensionless)."""
    hod = t_hour % 24
    dow = (t_hour // 24) % 7
    # two daily peaks (morning + evening)
    diurnal = 0.6 + 0.7 * math.exp(-((hod - 9) ** 2) / 8.0) \
                  + 0.9 * math.exp(-((hod - 20) ** 2) / 6.0)
    weekend = 1.35 if dow >= 5 else 1.0          # Fri/Sat commercial spikes (Dhaka week)
    return diurnal * weekend


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
GAS_DANGER = 0.62          # latent gas level considered a biological hazard
HORIZON_H = 6              # look-ahead horizon for the hazard label


def simulate(n_bins: int = 30, days: int = 180, seed: int = 42):
    """
    Run the hourly latent simulation.

    Returns
    -------
    bins_df : DataFrame of bin metadata
    latent  : dict bin_id -> DataFrame with per-hour latent + observed values
              and forward-looking labels.
    """
    rng = np.random.default_rng(seed)
    bins_df = build_bins(n_bins, seed=seed + 1)
    T = days * 24

    # pre-compute ambient series (shared across bins)
    temp_series = np.array([_ambient_temperature(t, rng) for t in range(T)])
    # rain events depress temperature slightly and raise humidity
    rain = (rng.random(T) < 0.04).astype(float)
    humid_series = np.clip(62.0 - 0.8 * (temp_series - 28.0) + 22.0 * rain
                           + rng.normal(0, 4.0, T), 25.0, 100.0)

    latent = {}
    for _, b in bins_df.iterrows():
        fill = rng.uniform(0.0, 0.3)
        decomp = 0.0                      # accumulated decomposition (odor potential)
        dwell = 0.0                       # hours since last collection
        recs = []
        # Poisson event bursts (parties, spoiled-delivery dumps)
        burst_times = np.where(rng.random(T) < 0.01)[0]
        burst_set = set(burst_times.tolist())

        for t in range(T):
            amb_t = temp_series[t]
            amb_h = humid_series[t]

            # --- waste arrival ---
            add = b.arrival * _demand_multiplier(t) * (1.0 + rng.normal(0, 0.15))
            if t in burst_set:
                add += rng.uniform(0.15, 0.45)     # sudden dump
            fill = min(fill + max(add, 0.0), 1.3)  # allow overflow beyond 1.0
            dwell += 1.0

            # --- decomposition / gas (Arrhenius-like q10 with temperature) ---
            q10 = 2.0 ** ((amb_t - 20.0) / 10.0)
            decomp += b.organic * fill * 0.010 * q10
            decomp *= 0.98                          # slow decay/venting
            gas_latent = 1.0 - math.exp(-1.4 * decomp)   # saturating 0..1

            # internal bin temperature = ambient + exothermic decomposition
            bin_temp = amb_t + 6.0 * gas_latent

            recs.append(dict(
                t=t, hour=t % 24, dow=(t // 24) % 7,
                fill=fill, gas=gas_latent, decomp=decomp,
                temp=bin_temp, humidity=amb_h, dwell=dwell,
                overflow=1 if fill >= 1.0 else 0,
            ))

            # --- collection (scheduled) OR forced when overflowing badly ---
            due = ((t - b.collect_phase) % b.collect_period == 0) and t > 0
            if due or fill >= 1.25:
                fill = rng.uniform(0.0, 0.05)
                decomp = 0.0
                dwell = 0.0

        df = pd.DataFrame(recs)

        # --- forward-looking labels (from the LATENT future) ---
        fill_arr = df["fill"].to_numpy()
        gas_arr = df["gas"].to_numpy()
        n = len(df)
        hazard = np.zeros(n, dtype=int)
        tto = np.full(n, float(HORIZON_H * 4), dtype=float)   # cap
        for i in range(n):
            hi = min(n, i + HORIZON_H + 1)
            window_fill = fill_arr[i + 1:hi]
            window_gas = gas_arr[i + 1:hi]
            if window_fill.size and (window_fill.max() >= 1.0 or window_gas.max() >= GAS_DANGER):
                hazard[i] = 1
            # continuous time-to-overflow within a longer horizon
            future = fill_arr[i:min(n, i + HORIZON_H * 4)]
            hit = np.where(future >= 1.0)[0]
            if hit.size:
                tto[i] = float(hit[0])
        df["hazard_within_h"] = hazard
        df["time_to_overflow"] = tto
        latent[b.bin_id] = df

    return bins_df, latent


# ---------------------------------------------------------------------------
# Sensor observation + feature engineering (past-only; no leakage)
# ---------------------------------------------------------------------------
def observe(latent_df: pd.DataFrame, seed: int, p_missing: float = 0.0):
    """Add noisy, possibly-missing sensor observations to a latent frame."""
    rng = np.random.default_rng(seed)
    n = len(latent_df)
    out = latent_df.copy()

    def noisy(col, sd, lo, hi):
        v = latent_df[col].to_numpy() + rng.normal(0, sd, n)
        return np.clip(v, lo, hi)

    obs = {
        "o_waste": noisy("fill", 0.03, 0.0, 1.3),
        "o_gas":   noisy("gas", 0.04, 0.0, 1.0),
        "o_temp":  noisy("temp", 0.6, 0.0, 60.0),
        "o_humidity": noisy("humidity", 2.5, 0.0, 100.0),
    }
    if p_missing > 0:
        for k in obs:
            mask = rng.random(n) < p_missing
            obs[k] = obs[k].astype(float)
            obs[k][mask] = np.nan
    for k, v in obs.items():
        out[k] = v
    return out


ROLL = 10  # rolling window matching the manuscript


def build_features(obs_df: pd.DataFrame) -> pd.DataFrame:
    """Past-only rolling features (mean/std/slope) -> no temporal leakage."""
    df = obs_df.copy().reset_index(drop=True)
    feats = {}
    for src, name in [("o_waste", "waste"), ("o_gas", "gas"),
                      ("o_temp", "temp"), ("o_humidity", "humidity")]:
        s = df[src]
        # fill short gaps for the rolling stats using forward fill (realistic)
        s_ff = s.ffill()
        feats[name] = s_ff
        feats[f"mean_{name}"] = s_ff.rolling(ROLL, min_periods=1).mean()
        feats[f"std_{name}"] = s_ff.rolling(ROLL, min_periods=2).std().fillna(0.0)
        # trend = slope over the window (last - first)/ROLL
        feats[f"trend_{name}"] = (s_ff - s_ff.shift(ROLL)).fillna(0.0) / ROLL
    feats["hour"] = df["hour"]
    feats["dow"] = df["dow"]
    fdf = pd.DataFrame(feats)
    fdf["time_to_overflow"] = df["time_to_overflow"]
    fdf["hazard_within_h"] = df["hazard_within_h"]
    fdf["t"] = df["t"]
    fdf["bin_id"] = df["bin_id"] if "bin_id" in df else -1
    return fdf


if __name__ == "__main__":
    bins_df, latent = simulate(n_bins=6, days=20, seed=1)
    df0 = latent[0]
    print("bins:", len(bins_df), "| hours:", len(df0))
    print("overflow rate bin0:", df0["overflow"].mean().round(3),
          "| hazard rate:", df0["hazard_within_h"].mean().round(3))
    print(df0[["t", "fill", "gas", "temp", "hazard_within_h", "time_to_overflow"]].head(8).to_string())
