"""
Does a survival objective beat the censored squared loss here? Measured: no.
=============================================================================

Time to overflow is right-censored at 24 h and 60 percent of labels sit at that
cap. Fitting a squared loss against such labels treats "at least 24 h" as
"exactly 24 h", which biases long horizons downward. The standard remedy is an
accelerated failure time model, which is given the interval [t, infinity) for a
censored row instead of the point t.

This script runs that comparison so the conclusion can be checked rather than
taken on trust. Three candidates on one temporal split with the same 24 h purge
gap:

  * the pooled squared-loss model that is actually served,
  * a conditional squared-loss model fitted on uncensored rows only,
  * XGBoost `survival:aft` across three distributions and three scales.

Metrics are chosen to survive censoring. Harrell's C-index scores only the
ordering and excludes pairs whose earlier time is censored, so it needs no
uncensored label. Mean absolute error and R squared are reported on uncensored
rows, where a timing label exists at all. Mean absolute error over every row is
also shown, and it is the one to distrust: with most labels at the cap, a model
that simply predicts the cap scores well on it.

Result, measured on the seeded network:

    model                          C-index   MAE(unc)  R2(unc)
    pooled squared loss (served)    0.8995      3.271   0.6466
    conditional squared loss        0.8681      2.393   0.7974
    AFT normal, scale 0.5           0.8518      3.451   0.5530

The survival objective is worse at ranking and worse at timing. Every AFT
variant tried came out below the squared-loss baseline on both.

Why, stated as a hypothesis rather than a result: the censoring here is
administrative and Type I. Every record is cut at the same known 24 h horizon,
because that is where the label stops being computed, not because bins leave the
study at varying times. Accelerated failure time models assume a parametric
survival distribution and are designed for censoring that varies across
subjects. With one fixed cap and a distribution that fits none of the normal,
logistic or extreme-value families well, the parametric assumption costs more
than the correct censoring treatment gains.

The practical conclusion the manuscript draws: serve the pooled model, which
ranks best and ranking is what the planner consumes, and report the conditional
model as the unbiased timing estimator for the rows where timing is defined.

Requires xgboost, which the deployed service does not use.
Run:  python exp_censoring.py
"""
from __future__ import annotations

import os
import pathlib
import sys

# Resolve the repository from this file's own location.  These two lines used to
# carry absolute paths from one developer's machine, which meant the script ran
# nowhere else and the reproducibility claim the paper makes was false for every
# other reader.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "waste_manager"))
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "waste_manager.settings")

import django  # noqa: E402

django.setup()

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

from bins.utils.ai.train_forward import (  # noqa: E402
    _concordance_index, build_dataset)
from wastebins_core import features as CORE_FEATURES  # noqa: E402
from wastebins_core import hpo as CORE_HPO  # noqa: E402

CAP = CORE_FEATURES.TTO_CAP_H

X, y_tto, y_haz, hours, groups = build_dataset()
print(f"{len(y_tto)} rows, {X.shape[1]} features, cap {CAP} h")

train_idx, test_idx = CORE_HPO.temporal_split(
    len(y_tto), test_frac=0.2, hours=hours,
    embargo_h=CORE_FEATURES.TTO_CAP_H)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_tto[train_idx], y_tto[test_idx]

censored_train = y_train >= CAP - 1e-9
censored_test = y_test >= CAP - 1e-9
observed_test = ~censored_test
print(f"train {len(y_train)} ({censored_train.mean():.1%} censored), "
      f"test {len(y_test)} ({censored_test.mean():.1%} censored)")
print()


def report(name, predictions):
    c = _concordance_index(y_test, predictions, event=observed_test)
    mae = mean_absolute_error(y_test[observed_test], predictions[observed_test])
    r2 = r2_score(y_test[observed_test], predictions[observed_test])
    mae_all = mean_absolute_error(y_test, predictions)
    print(f"  {name:<34}{c:>9.4f}{mae:>11.3f}{r2:>9.4f}{mae_all:>11.3f}")
    return c, mae, r2


print(f"  {'model':<34}{'C-index':>9}{'MAE(unc)':>11}{'R2(unc)':>9}{'MAE(all)':>11}")
print("  " + "-" * 74)

# 1. pooled squared loss, as served
pooled = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04,
                                       max_depth=4, max_leaf_nodes=31,
                                       min_samples_leaf=20, random_state=42)
pooled.fit(X_train, y_train)
report("pooled squared loss (served)", pooled.predict(X_test))

# 2. conditional squared loss, uncensored rows only
obs_train = ~censored_train
conditional = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04,
                                            max_depth=4, max_leaf_nodes=31,
                                            min_samples_leaf=20, random_state=42)
conditional.fit(X_train[obs_train], y_train[obs_train])
report("conditional squared loss", conditional.predict(X_test))

# 3. accelerated failure time
# A right-censored row is [t, inf): the overflow happens at or after t.
lower = np.where(censored_train, y_train, y_train)
upper = np.where(censored_train, np.inf, y_train)
dtrain = xgb.DMatrix(X_train)
dtrain.set_float_info("label_lower_bound", lower)
dtrain.set_float_info("label_upper_bound", upper)
dtest = xgb.DMatrix(X_test)

for dist in ("normal", "logistic", "extreme"):
    for scale in (0.5, 1.0, 1.5):
        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": dist,
            "aft_loss_distribution_scale": scale,
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "seed": 42,
        }
        booster = xgb.train(params, dtrain, num_boost_round=400, verbose_eval=False)
        # AFT predicts the survival time directly.
        predictions = np.clip(booster.predict(dtest), 0.0, CAP)
        report(f"AFT {dist}, scale {scale}", predictions)
