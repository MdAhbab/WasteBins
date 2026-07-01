"""
Experiment 1 (reviewer Q1, Q2 + model-improvement study for SCS).
=================================================================
Turns the manuscript's single circular-target Random Forest into a proper,
honest, cited model study:

  (A) CIRCULAR-TARGET PITFALL  -- reproduce the manuscript setup (RF regressing a
      deterministic formula of its own inputs) -> R^2 ~ 0.99 = self-consistency.

  (B) HONEST FORWARD TASK with a TEMPORAL split (first 120 d train / last 60 d
      test) emulating deployment, plus GroupKFold-by-bin.
        - Regression: time-to-overflow (hours).
        - Classification: hazard-within-6h.

  (C) MODEL BAKE-OFF (the actual improvement):
        Ridge (current-state) | Random Forest [Breiman 2001]
        | Histogram Gradient Boosting [Friedman 2001; Ke et al. 2017 (LightGBM)].

  (D) UNCERTAINTY QUANTIFICATION: quantile gradient boosting (P10/P50/P90)
        [Koenker & Bassett 1978; Meinshausen 2006] -> prediction intervals for
        risk-aware routing; we report empirical interval coverage.

  (E) PROBABILITY CALIBRATION of the hazard classifier (isotonic)
        [Platt 1999; Niculescu-Mizil & Caruana 2005] -> Brier score + reliability
        curve (directly answers the "missing calibration curve" comment).

  (F) INFERENCE LATENCY (the paper claims sub-second routing).

All numbers are written to results/model.json.
"""
from __future__ import annotations
import json, time, pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.linear_model import Ridge
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (r2_score, mean_absolute_error, roc_auc_score,
                             average_precision_score, brier_score_loss)
from sklearn.model_selection import GroupKFold

import dataset as D

RESULTS = D.RESULTS
SPLIT_HOUR = 120 * 24
SEED = 42
FCOLS = D.FEATURE_COLS


def circular_target(data: pd.DataFrame) -> np.ndarray:
    waste = data["waste"].to_numpy(); gas = data["gas"].to_numpy()
    temp = data["temp"].to_numpy(); hum = data["humidity"].to_numpy()
    temp_p = np.clip(np.abs(temp - 25.0) / 15.0, 0, 1)
    hum_p = np.clip((hum - 50.0) / 50.0, 0, 1)
    p = 0.35*np.clip(waste,0,1) + 0.25*np.clip(gas,0,1) + 0.10*temp_p + 0.05*hum_p
    return np.clip(p / 0.75, 0, 1)


def main():
    _, data = D.build()
    data = data.dropna(subset=FCOLS).reset_index(drop=True)
    tr = (data["t"] < SPLIT_HOUR).to_numpy()
    te = ~tr
    X = data[FCOLS].to_numpy()
    out = {"dataset": {"n_records": int(len(data)), "n_bins": D.N_BINS, "days": D.DAYS,
                       "n_train": int(tr.sum()), "n_test": int(te.sum()),
                       "hazard_rate_test": float(data.loc[te, "hazard_within_h"].mean())}}

    # ---------- (A) circular pitfall ----------
    yc = circular_target(data)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
    rf.fit(X[tr], yc[tr])
    out["A_circular_pitfall"] = {
        "target": "deterministic priority formula of the SAME inputs",
        "model": "RandomForest (manuscript config)",
        "r2_test": round(float(r2_score(yc[te], rf.predict(X[te]))), 4),
        "interpretation": "self-consistency, not predictive skill",
    }

    # ---------- (B+C) forward regression bake-off: time-to-overflow ----------
    yr = data["time_to_overflow"].to_numpy()
    reg_models = {
        "ridge_current_state": (Ridge(alpha=1.0), ["waste", "gas", "temp", "humidity"]),
        "random_forest": (RandomForestRegressor(n_estimators=300, max_depth=14,
                                                random_state=SEED, n_jobs=-1), FCOLS),
        "hist_grad_boost": (HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                                          max_depth=None, l2_regularization=1.0,
                                                          random_state=SEED), FCOLS),
    }
    reg = {}
    # naive baseline: predict train mean
    base = np.full(int(te.sum()), yr[tr].mean())
    reg["persistence_mean"] = {"r2": round(float(r2_score(yr[te], base)), 4),
                               "mae_h": round(float(mean_absolute_error(yr[te], base)), 3)}
    for name, (m, cols) in reg_models.items():
        Xtr = data.loc[tr, cols].to_numpy(); Xte = data.loc[te, cols].to_numpy()
        m.fit(Xtr, yr[tr]); pred = m.predict(Xte)
        reg[name] = {"r2": round(float(r2_score(yr[te], pred)), 4),
                     "mae_h": round(float(mean_absolute_error(yr[te], pred)), 3)}
    out["B_forward_regression_time_to_overflow"] = {"target": "hours until fill>=1.0",
                                                    "temporal_split": "first 120d train / last 60d test",
                                                    "models": reg}

    # ---------- (D) uncertainty: quantile gradient boosting ----------
    q_models = {}
    preds_q = {}
    for q in (0.1, 0.5, 0.9):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                          learning_rate=0.06, random_state=SEED)
        m.fit(data.loc[tr, FCOLS].to_numpy(), yr[tr])
        preds_q[q] = m.predict(data.loc[te, FCOLS].to_numpy())
    lo, hi = preds_q[0.1], preds_q[0.9]
    cover = float(np.mean((yr[te] >= lo) & (yr[te] <= hi)))
    out["D_uncertainty_quantile"] = {
        "method": "quantile gradient boosting (P10/P50/P90)",
        "nominal_interval": 0.80,
        "empirical_coverage_P10_P90": round(cover, 3),
        "median_r2": round(float(r2_score(yr[te], preds_q[0.5])), 4),
        "use": "P10 (soonest-overflow) drives risk-aware priority for critical bins",
    }

    # ---------- (B+C) forward classification: hazard within 6h ----------
    yh = data["hazard_within_h"].to_numpy()
    clf_out = {}
    rf_c = RandomForestClassifier(n_estimators=300, max_depth=14, random_state=SEED,
                                  n_jobs=-1, class_weight="balanced")
    rf_c.fit(X[tr], yh[tr]); p_rf = rf_c.predict_proba(X[te])[:, 1]
    hgb_c = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                           l2_regularization=1.0, random_state=SEED)
    hgb_c.fit(X[tr], yh[tr]); p_hgb = hgb_c.predict_proba(X[te])[:, 1]

    def cls_metrics(p):
        return {"roc_auc": round(float(roc_auc_score(yh[te], p)), 4),
                "avg_precision": round(float(average_precision_score(yh[te], p)), 4),
                "brier": round(float(brier_score_loss(yh[te], p)), 4)}

    clf_out["random_forest"] = cls_metrics(p_rf)
    clf_out["hist_grad_boost"] = cls_metrics(p_hgb)
    clf_out["baseline_current_fill"] = {"roc_auc": round(float(roc_auc_score(yh[te], data.loc[te, "waste"])), 4)}
    clf_out["baseline_current_gas"] = {"roc_auc": round(float(roc_auc_score(yh[te], data.loc[te, "gas"])), 4)}

    # ---------- (E) calibration ----------
    cal = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                                                random_state=SEED),
                                 method="isotonic", cv=3)
    cal.fit(X[tr], yh[tr]); p_cal = cal.predict_proba(X[te])[:, 1]
    frac_pos, mean_pred = calibration_curve(yh[te], p_cal, n_bins=10, strategy="quantile")
    clf_out["hist_grad_boost_calibrated"] = cls_metrics(p_cal)
    clf_out["reliability_curve_calibrated"] = {"mean_predicted": [round(float(x), 3) for x in mean_pred],
                                               "fraction_positive": [round(float(x), 3) for x in frac_pos]}
    out["C_forward_classification_hazard"] = clf_out

    # ---------- feature importance (permutation, honest) ----------
    best = reg_models["hist_grad_boost"][0]
    perm = permutation_importance(best, data.loc[te, FCOLS].to_numpy(), yr[te],
                                  n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = sorted(zip(FCOLS, perm.importances_mean), key=lambda kv: -kv[1])
    out["feature_importance_permutation"] = {k: round(float(v), 4) for k, v in imp}

    # ---------- (F) inference latency ----------
    Xte = data.loc[te, FCOLS].to_numpy()[:5000]
    t0 = time.perf_counter(); best.predict(Xte); dt = time.perf_counter() - t0
    out["F_inference_latency"] = {"model": "hist_grad_boost", "n": int(len(Xte)),
                                  "total_ms": round(dt * 1000, 2),
                                  "per_1000_ms": round(dt * 1000 / len(Xte) * 1000, 3)}

    # ---------- GroupKFold by bin (HGB regression) ----------
    gkf = GroupKFold(n_splits=5); groups = data["bin_id"].to_numpy()
    r2s = []
    for tri, tei in gkf.split(X, yr, groups):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=SEED)
        m.fit(X[tri], yr[tri]); r2s.append(r2_score(yr[tei], m.predict(X[tei])))
    out["group_kfold_by_bin_hgb"] = {"r2_mean": round(float(np.mean(r2s)), 4),
                                     "r2_std": round(float(np.std(r2s)), 4)}

    (RESULTS / "model.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
