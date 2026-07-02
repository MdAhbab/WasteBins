"""
Forward-looking (non-circular) priority model for the SCS revision.
===================================================================
The legacy train_model.py regresses a deterministic priority *formula* computed
from the same sensor inputs it is given as features -> the reported R^2 measures
self-consistency, not predictive skill.

This module trains an HONEST, deployment-realistic model:

  Target (from the FUTURE, never observed at inference):
    * time_to_overflow  -- hours until waste_level >= 1.0
    * hazard_within_h   -- overflow OR gas >= danger within HORIZON_H hours

  Features (PAST only, no leakage): current reading + rolling mean/std/trend of
  the last ROLL readings per bin + hour/day-of-week.

  Models:
    * HistGradientBoostingRegressor       (time-to-overflow)   [Friedman 2001]
    * quantile HGB P10/P50/P90            (uncertainty)        [Koenker&Bassett 1978]
    * CalibratedClassifierCV(HGB)         (calibrated hazard)  [Platt 1999;
                                                                Niculescu-Mizil 2005]

  Validation: temporal split (earliest 80% train / latest 20% test), reported in
  the metadata so the numbers are auditable.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from django.utils import timezone
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (r2_score, mean_absolute_error, roc_auc_score,
                             average_precision_score, brier_score_loss)

from bins.models import SensorReading
from .model_store import save_forward_bundle

ROLL = 10
HORIZON_H = 6.0
GAS_DANGER = 0.62
TTO_CAP_H = 24.0

FEATURE_COLS = [
    "waste", "mean_waste", "std_waste", "trend_waste",
    "gas", "mean_gas", "std_gas", "trend_gas",
    "temp", "mean_temp", "std_temp", "trend_temp",
    "humidity", "mean_humidity", "std_humidity", "trend_humidity",
    "hour", "dow",
]


def _readings_frame() -> pd.DataFrame:
    qs = (SensorReading.objects
          .select_related("node")
          .order_by("node_id", "timestamp")
          .values("node_id", "timestamp", "waste_level", "gas_level",
                  "temperature", "humidity"))
    df = pd.DataFrame(list(qs))
    if df.empty:
        return df
    df = df.rename(columns={"waste_level": "waste", "gas_level": "gas",
                            "temperature": "temp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _build_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp").reset_index(drop=True)
    hrs = (g["timestamp"] - g["timestamp"].iloc[0]).dt.total_seconds() / 3600.0
    feats = {}
    for col in ["waste", "gas", "temp", "humidity"]:
        s = g[col].astype(float)
        feats[col] = s
        feats[f"mean_{col}"] = s.rolling(ROLL, min_periods=1).mean()
        feats[f"std_{col}"] = s.rolling(ROLL, min_periods=2).std().fillna(0.0)
        feats[f"trend_{col}"] = (s - s.shift(ROLL)).fillna(0.0) / ROLL
    feats["hour"] = g["timestamp"].dt.hour
    feats["dow"] = g["timestamp"].dt.dayofweek
    fdf = pd.DataFrame(feats)

    # forward labels from future readings within the horizon (uses real hours)
    waste = g["waste"].to_numpy(); gas = g["gas"].to_numpy()
    hrs_arr = hrs.to_numpy()
    n = len(g)
    hazard = np.zeros(n, dtype=int)
    tto = np.full(n, TTO_CAP_H, dtype=float)
    for i in range(n):
        dt = hrs_arr - hrs_arr[i]
        fut = (dt > 0) & (dt <= HORIZON_H)
        if fut.any() and (waste[fut].max() >= 1.0 or gas[fut].max() >= GAS_DANGER):
            hazard[i] = 1
        of = (dt >= 0) & (dt <= TTO_CAP_H) & (waste >= 1.0)
        if of.any():
            tto[i] = float(min(TTO_CAP_H, dt[of].min()))
    fdf["hazard_within_h"] = hazard
    fdf["time_to_overflow"] = tto
    fdf["timestamp"] = g["timestamp"].values
    return fdf


def train_forward(test_frac: float = 0.2, random_state: int = 42):
    df = _readings_frame()
    if df.empty:
        raise ValueError("No sensor data available to train.")
    frames = [_build_group(g) for _, g in df.groupby("node_id")]
    data = pd.concat(frames, ignore_index=True).dropna(subset=FEATURE_COLS)
    if len(data) < 50:
        raise ValueError("Need at least ~50 readings across bins to train a forward model.")

    data = data.sort_values("timestamp").reset_index(drop=True)
    split = int(len(data) * (1 - test_frac))
    tr, te = data.iloc[:split], data.iloc[split:]
    Xtr, Xte = tr[FEATURE_COLS].to_numpy(), te[FEATURE_COLS].to_numpy()
    ytr, yte = tr["time_to_overflow"].to_numpy(), te["time_to_overflow"].to_numpy()
    htr, hte = tr["hazard_within_h"].to_numpy(), te["hazard_within_h"].to_numpy()

    reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                        l2_regularization=1.0, random_state=random_state)
    reg.fit(Xtr, ytr)
    quantiles = {}
    for q, key in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                          learning_rate=0.06, random_state=random_state)
        m.fit(Xtr, ytr); quantiles[key] = m

    metrics = {"n_records": int(len(data)), "n_train": int(len(tr)), "n_test": int(len(te)),
               "reg_r2": float(r2_score(yte, reg.predict(Xte))),
               "reg_mae_h": float(mean_absolute_error(yte, reg.predict(Xte)))}

    clf = None
    if len(np.unique(htr)) > 1 and hte.sum() > 0:
        base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                              random_state=random_state)
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(Xtr, htr)
        p = clf.predict_proba(Xte)[:, 1]
        metrics.update({"hazard_roc_auc": float(roc_auc_score(hte, p)),
                        "hazard_avg_precision": float(average_precision_score(hte, p)),
                        "hazard_brier": float(brier_score_loss(hte, p))})

    bundle = {"regressor": reg, "features": FEATURE_COLS, "classifier": clf, **quantiles,
              "horizon_h": HORIZON_H, "tto_cap_h": TTO_CAP_H}
    meta = {"version": f"forward_hgb_{timezone.now().date().isoformat()}",
            "trained_at": timezone.now().isoformat(),
            "target": "time_to_overflow (h) + hazard_within_6h",
            "validation": "temporal split (earliest 80% train / latest 20% test)",
            "features": FEATURE_COLS, "metrics": metrics}
    save_forward_bundle(bundle, meta)
    return meta


def predict_forward(bundle: dict, feature_row: dict):
    """Return risk-aware outputs for one engineered feature row."""
    X = np.array([[feature_row.get(c, 0.0) for c in bundle["features"]]], dtype=float)
    out = {"time_to_overflow_h": float(bundle["regressor"].predict(X)[0]),
           "tto_p10_h": float(bundle["q10"].predict(X)[0]),
           "tto_p90_h": float(bundle["q90"].predict(X)[0])}
    if bundle.get("classifier") is not None:
        out["hazard_prob"] = float(bundle["classifier"].predict_proba(X)[0, 1])
    # risk-aware priority in [0,1]: closer overflow (P10) + hazard probability
    cap = bundle.get("tto_cap_h", TTO_CAP_H)
    urgency = 1.0 - min(1.0, max(0.0, out["tto_p10_h"] / cap))
    out["risk_priority"] = float(max(out.get("hazard_prob", 0.0), urgency))
    return out
