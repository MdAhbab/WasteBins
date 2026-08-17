"""
Offline training of the forward-looking prediction bundle.

Run through ``manage.py train_forward``.  Nothing here is ever invoked from a
request: fitting a gradient-boosting ensemble plus a hyperparameter search is a
minutes-scale job, and the deployed application must stay responsive while it
happens.

What is trained
---------------
* ``regressor``   -- time to overflow in hours, right-censored at 24 h.
* ``q10``/``q90`` -- quantile heads giving a prediction interval, so the planner
  can act on a pessimistic bound rather than a point estimate.
* ``classifier``  -- calibrated probability of a hazard within the horizon.

Why the target is forward-looking
---------------------------------
The original model regressed a deterministic priority *formula* computed from
the very inputs it was given as features.  Its R^2 measured self-consistency,
not skill.  Here the labels come from the strictly future trajectory, which the
features cannot see, so the score is genuine forecasting performance.

Protocol
--------
Delegated to :mod:`wastebins_core.hpo`: a temporal outer hold-out, a
grouped-by-bin inner cross-validation, and a randomised search with a declared
space and fixed seed.  The full protocol, the search space and the winning
configuration are all written into the artefact metadata.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from django.utils import timezone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from wastebins_core import features as CORE_FEATURES
from wastebins_core import hpo as CORE_HPO
from wastebins_core import xai as CORE_XAI

from bins.models import ModelVersion, SensorReading
from bins.services import audit, models_registry

logger = logging.getLogger(__name__)

# Re-exported for backward compatibility with older imports.
FEATURE_COLS = CORE_FEATURES.FEATURE_COLS
ROLL = CORE_FEATURES.ROLL
HORIZON_H = CORE_FEATURES.HORIZON_H
TTO_CAP_H = CORE_FEATURES.TTO_CAP_H
GAS_DANGER = CORE_FEATURES.GAS_DANGER

MIN_ROWS = 200
MIN_ROWS_PER_BIN = 30


def _readings_frame() -> pd.DataFrame:
    rows = (SensorReading.objects
            .order_by("node_id", "timestamp")
            .values("node_id", "timestamp", "waste_level", "gas_level",
                    "temperature", "humidity"))
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    frame = frame.rename(columns={"waste_level": "waste", "gas_level": "gas",
                                  "temperature": "temp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def build_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """
    Assemble the supervised dataset from stored telemetry.

    Returns ``(X, y_tto, y_hazard, groups, timestamps)``, sorted by time so the
    temporal split is a simple prefix.
    """
    frame = _readings_frame()
    if frame.empty:
        raise ValueError("No sensor readings stored; ingest telemetry before training.")

    feature_blocks: List[np.ndarray] = []
    tto_blocks: List[np.ndarray] = []
    hazard_blocks: List[np.ndarray] = []
    group_blocks: List[np.ndarray] = []
    time_blocks: List[pd.Series] = []

    for node_id, group in frame.groupby("node_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < MIN_ROWS_PER_BIN:
            continue

        hours = ((group["timestamp"] - group["timestamp"].iloc[0])
                 .dt.total_seconds() / 3600.0).to_numpy()
        series = {
            "waste": group["waste"].astype(float).to_numpy(),
            "gas": group["gas"].astype(float).to_numpy(),
            "temp": group["temp"].astype(float).to_numpy(),
            "humidity": group["humidity"].astype(float).to_numpy(),
        }
        dow = group["timestamp"].dt.dayofweek.to_numpy()

        X = CORE_FEATURES.build_matrix(series, hours, dow=dow)
        hazard, tto = CORE_FEATURES.forward_labels(series["waste"], series["gas"], hours)

        # The tail of the record has no future to label against, so its labels
        # are censored by construction; dropping it prevents the model from
        # learning that "the end of the record" means "no hazard".
        #
        # The window must cover the *longest* label horizon, not the shortest.
        # It used to use HORIZON_H (6 h) while `forward_labels` looks TTO_CAP_H
        # (24 h) ahead for the time-to-overflow target, so the final 18 h of
        # every bin's record carried a time-to-overflow of exactly the cap
        # whenever the record simply ran out -- a fabricated "this bin is safe
        # for a full day" label on precisely the rows where the answer was
        # unknown, and one that biases the long-horizon regime the planner
        # depends on.
        label_horizon = max(CORE_FEATURES.HORIZON_H, CORE_FEATURES.TTO_CAP_H)
        cutoff = hours[-1] - label_horizon
        keep = hours <= cutoff
        if keep.sum() < MIN_ROWS_PER_BIN // 2:
            continue

        feature_blocks.append(X[keep])
        tto_blocks.append(tto[keep])
        hazard_blocks.append(hazard[keep])
        group_blocks.append(np.full(int(keep.sum()), node_id))
        time_blocks.append(group["timestamp"][keep])

    if not feature_blocks:
        raise ValueError(
            f"Not enough history: each bin needs at least {MIN_ROWS_PER_BIN} readings.")

    X = np.vstack(feature_blocks)
    y_tto = np.concatenate(tto_blocks)
    y_hazard = np.concatenate(hazard_blocks)
    groups = np.concatenate(group_blocks)
    timestamps = pd.concat(time_blocks, ignore_index=True)

    order = np.argsort(timestamps.to_numpy())
    return (X[order], y_tto[order], y_hazard[order], groups[order],
            timestamps.iloc[order].reset_index(drop=True))


def train_forward(test_frac: float = 0.2, random_state: int = 42,
                  n_iter: int = 25, n_splits: int = 5, quick: bool = False,
                  compute_importance: bool = True) -> Dict:
    """
    Train, evaluate and register the forward bundle.

    ``quick=True`` shrinks the search budget for smoke tests and CI; the
    metadata records which mode was used so a fast run is never mistaken for the
    tuned result reported in the paper.
    """
    X, y_tto, y_hazard, groups, timestamps = build_dataset()
    if len(y_tto) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} usable rows to train; have {len(y_tto)}.")

    if quick:
        n_iter, n_splits = 4, 3

    # Elapsed hours on a single global clock, for the purge gaps below.
    clock_h = ((timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600.0).to_numpy()
    label_horizon = max(CORE_FEATURES.HORIZON_H, CORE_FEATURES.TTO_CAP_H)

    # Purge one full label horizon out of the training side of the hold-out.  A
    # row's target is derived from the trajectory up to `label_horizon` hours
    # after it, so without the gap every training row near the cut is labelled
    # with test-period outcomes and the hold-out score is measuring leakage.
    train_idx, test_idx = CORE_HPO.temporal_split(
        len(y_tto), test_frac, hours=clock_h, embargo_h=label_horizon)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_tto[train_idx], y_tto[test_idx]
    h_train, h_test = y_hazard[train_idx], y_hazard[test_idx]
    g_train = groups[train_idx]

    # The inner cross-validation needs the same treatment, and it additionally
    # has to be ordered: GroupKFold-by-bin was free to train on a bin's future
    # and validate on its past, so the CV score it reported was not a forecast.
    inner_cv = CORE_HPO.purged_time_series_splits(
        clock_h[train_idx], n_splits=n_splits, embargo_h=label_horizon)

    # --- regressor ---------------------------------------------------
    search = CORE_HPO.search_regressor(X_train, y_train, g_train,
                                       n_iter=n_iter, n_splits=n_splits,
                                       seed=random_state, cv=inner_cv)
    regressor = search.estimator
    predictions = regressor.predict(X_test)
    metrics: Dict = {
        "n_records": int(len(y_tto)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_bins": int(len(np.unique(groups))),
        "reg_r2": round(float(r2_score(y_test, predictions)), 5),
        "reg_mae_h": round(float(mean_absolute_error(y_test, predictions)), 5),
        "reg_cv_r2_mean": round(float(search.best_cv_score), 5),
        "reg_cv_r2_std": round(float(search.cv_score_std), 5),
    }

    # A model that cannot beat predicting the training mean has learned nothing.
    baseline = np.full_like(y_test, float(np.mean(y_train)), dtype=float)
    metrics["baseline_mean_r2"] = round(float(r2_score(y_test, baseline)), 5)
    metrics["baseline_mean_mae_h"] = round(float(mean_absolute_error(y_test, baseline)), 5)

    # Censoring is the single most important caveat on the headline R^2 and it
    # has to be reported, not buried.  Most bins do not overflow inside the
    # 24 h window, so most labels are the cap itself -- a constant.  An R^2
    # computed over all rows is therefore substantially a score for recognising
    # "this one is not going to overflow today", which is easy.  The uncensored
    # subset is where the actual regression problem lives, and it scores well
    # below the pooled figure; both are recorded so neither can be quoted alone.
    censored = y_test >= (CORE_FEATURES.TTO_CAP_H - 1e-9)
    metrics["censored_share_train"] = round(
        float(np.mean(y_train >= CORE_FEATURES.TTO_CAP_H - 1e-9)), 4)
    metrics["censored_share_test"] = round(float(np.mean(censored)), 4)
    metrics["n_test_uncensored"] = int((~censored).sum())
    if (~censored).sum() > 1 and float(np.std(y_test[~censored])) > 0:
        metrics["reg_r2_uncensored"] = round(
            float(r2_score(y_test[~censored], predictions[~censored])), 5)
        metrics["reg_mae_h_uncensored"] = round(
            float(mean_absolute_error(y_test[~censored], predictions[~censored])), 5)
    metrics["censoring_note"] = (
        "time_to_overflow is right-censored at the cap and the regressor is fitted "
        "with a squared loss, which treats a censored label as an observed value. "
        "That biases long-horizon predictions downward; reg_r2_uncensored is the "
        "honest figure for the regression task itself.")

    # --- quantile heads ----------------------------------------------
    best = dict(search.best_params)
    quantiles = {}
    for q, key in ((0.1, "q10"), (0.5, "q50"), (0.9, "q90")):
        params = {k: v for k, v in best.items() if k != "max_features"}
        model = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                              random_state=random_state, **params)
        model.fit(X_train, y_train)
        quantiles[key] = model

    lo = quantiles["q10"].predict(X_test)
    hi = quantiles["q90"].predict(X_test)
    inside = np.mean((y_test >= np.minimum(lo, hi)) & (y_test <= np.maximum(lo, hi)))
    metrics["interval_coverage_p10_p90"] = round(float(inside), 4)
    metrics["interval_nominal_coverage"] = 0.80
    metrics["interval_mean_width_h"] = round(float(np.mean(np.abs(hi - lo))), 4)

    # --- hazard classifier -------------------------------------------
    classifier = None
    if len(np.unique(h_train)) > 1 and h_test.sum() > 0:
        clf_search = CORE_HPO.search_classifier(X_train, h_train, g_train,
                                                n_iter=max(4, n_iter // 2),
                                                n_splits=n_splits, seed=random_state,
                                                cv=inner_cv)
        # Calibrate on the same purged, ordered folds.  The default `cv=3` is a
        # stratified *random* split, which on an autocorrelated series with
        # overlapping label windows puts near-duplicate rows on both sides of
        # every fold; the isotonic map then looks well calibrated because it was
        # fitted on data it had effectively already seen.  Fall back to the
        # random split only when a purged fold has no positives to calibrate on,
        # and record which was used so the number is never misread.
        usable = [(tr, va) for tr, va in inner_cv
                  if len(np.unique(h_train[tr])) > 1 and len(np.unique(h_train[va])) > 1]
        calibration_cv = usable if len(usable) >= 2 else 3
        metrics["hazard_calibration_cv"] = (
            f"purged time-series ({len(usable)} folds)" if usable and len(usable) >= 2
            else "stratified k-fold fallback (too few positives per purged fold)")
        calibrated = CalibratedClassifierCV(clf_search.estimator, method="isotonic",
                                            cv=calibration_cv)
        calibrated.fit(X_train, h_train)
        classifier = calibrated
        # Namespace the classifier metrics so they cannot be confused with the
        # regressor's in a flat metrics dictionary.
        for key, value in CORE_HPO.classifier_report(calibrated, X_test, h_test).items():
            metrics[f"hazard_{key}"] = value
        metrics["hazard_cv_auc_mean"] = round(float(clf_search.best_cv_score), 5)
        metrics["hazard_cv_auc_std"] = round(float(clf_search.cv_score_std), 5)
        clf_params = clf_search.best_params
    else:
        clf_params = {}
        metrics["hazard_note"] = "insufficient hazard events in this dataset to fit a classifier"

    # --- global explainability ---------------------------------------
    importance: List[Dict] = []
    if compute_importance:
        sample = X_test if len(X_test) <= 2000 else X_test[:2000]
        target = y_test if len(y_test) <= 2000 else y_test[:2000]
        importance = CORE_XAI.permutation_importance(
            regressor.predict, sample, target, CORE_FEATURES.FEATURE_COLS,
            n_repeats=4 if quick else 8, seed=random_state)

    # A compact background sample travels with the bundle so local attribution
    # has a proper interventional reference at serving time.
    rng = np.random.default_rng(random_state)
    background = X_train[rng.choice(len(X_train), size=min(128, len(X_train)),
                                    replace=False)]

    bundle = {
        "regressor": regressor,
        "features": list(CORE_FEATURES.FEATURE_COLS),
        "classifier": classifier,
        "background": background,
        "horizon_h": CORE_FEATURES.HORIZON_H,
        "tto_cap_h": CORE_FEATURES.TTO_CAP_H,
        **quantiles,
    }

    version = f"forward_hgb_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
    meta = {
        "version": version,
        "trained_at": timezone.now().isoformat(),
        "target": ("time_to_overflow (hours, censored at 24 h) + "
                   f"hazard_within_{CORE_FEATURES.HORIZON_H:g}h"),
        "features": list(CORE_FEATURES.FEATURE_COLS),
        "feature_contract": CORE_FEATURES.describe(),
        "validation": ("temporal outer hold-out (latest {:.0%}) with a {:g} h purge gap, "
                       "and purged expanding-window inner cross-validation on the "
                       "same gap").format(test_frac, label_horizon),
        "label_horizon_h": label_horizon,
        "purge_gap_h": label_horizon,
        "hpo_protocol": CORE_HPO.describe_protocol(),
        "hyperparameters": {"regressor": best, "classifier": clf_params},
        "search": {"regressor": search.as_dict()},
        "metrics": metrics,
        "training_rows": int(len(y_tto)),
        "global_importance": importance,
        "quick_mode": bool(quick),
    }

    models_registry.save_forward_bundle(bundle, meta)
    meta["artifact_sha256"] = models_registry.load_forward_meta().get("artifact_sha256", "")

    ModelVersion.objects.filter(kind="forward_bundle", is_active=True).update(is_active=False)
    ModelVersion.objects.update_or_create(
        kind="forward_bundle", version=version,
        defaults={
            "name": "Forward-looking overflow and hazard bundle",
            "artifact_path": str(models_registry.forward_path()),
            "artifact_sha256": meta.get("artifact_sha256", ""),
            "hyperparameters": meta["hyperparameters"],
            "search_space": CORE_HPO.describe_protocol().get("spaces", {}),
            "metrics": metrics,
            "training_rows": int(len(y_tto)),
            "validation_scheme": meta["validation"],
            "is_active": True,
            "trained_at": timezone.now(),
        },
    )

    # A newly trained base model invalidates a corrector fitted to the old one.
    models_registry.reset_continual_learner()

    audit.append("model", {
        "version": version,
        "metrics": metrics,
        "hyperparameters": meta["hyperparameters"],
        "validation": meta["validation"],
        "artifact_sha256": meta.get("artifact_sha256", ""),
    }, actor="trainer")

    return meta


def predict_forward(bundle: Dict, feature_row: Dict) -> Dict:
    """Score one engineered feature row against a loaded bundle."""
    return models_registry.predict_batch([feature_row], use_continual=False)[0] \
        if bundle is not None else {}
