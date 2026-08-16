"""
Hyperparameter optimisation with a declared, auditable protocol.
================================================================

The first submission reported R^2 and AUC with no statement of how the models
were configured.  This module makes the whole procedure explicit and
reproducible, and records it in the artefact metadata so a reader can check it.

Protocol
--------
1. **Outer split -- temporal.**  The most recent ``test_frac`` of the record
   stream is held out and never touched during search.  Waste generation has
   strong weekly and seasonal structure, so a random split would leak the future.

2. **Inner split -- grouped by bin.**  Within the training portion, model
   selection uses ``GroupKFold`` with the bin identifier as the group, so a model
   is never tuned on the same bin it is scored on.  Without this, a model can
   memorise per-bin idiosyncrasies and report skill it does not have on a newly
   installed bin.

3. **Search -- randomised, fixed budget.**  ``RandomizedSearchCV`` over the
   declared space with a fixed ``n_iter`` and a fixed seed.  Randomised search is
   preferred to a grid because the important axes (learning rate, depth) are
   continuous and unequally influential.

4. **Refit and report.**  The winning configuration is refitted on the full
   training portion and reported on the untouched temporal test set.  Search
   scores are reported separately from test scores, and never conflated.

Everything -- the space, the number of draws, the seed, the fold count, the
winning configuration and both score sets -- is returned in the result and
persisted alongside the model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             mean_absolute_error, r2_score, roc_auc_score)


# ---------------------------------------------------------------------------
# Declared search spaces
# ---------------------------------------------------------------------------
HGB_REGRESSOR_SPACE: Dict[str, List] = {
    "max_iter": [200, 300, 400, 600, 800],
    "learning_rate": [0.02, 0.04, 0.06, 0.08, 0.12, 0.16],
    "max_depth": [None, 3, 4, 6, 8],
    "max_leaf_nodes": [15, 31, 63, 127],
    "min_samples_leaf": [10, 20, 40, 80],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0, 3.0],
    "max_features": [0.6, 0.8, 1.0],
}

HGB_CLASSIFIER_SPACE: Dict[str, List] = {
    "max_iter": [200, 300, 400, 600],
    "learning_rate": [0.02, 0.04, 0.06, 0.10, 0.15],
    "max_depth": [None, 3, 4, 6],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 40, 80],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0],
}

RF_SPACE: Dict[str, List] = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 8, 12, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", 0.5, 0.8, 1.0],
}

RIDGE_SPACE: Dict[str, List] = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}


@dataclass
class SearchResult:
    estimator_name: str
    best_params: Dict
    best_cv_score: float
    cv_score_std: float
    n_candidates: int
    n_splits: int
    search_seconds: float
    scoring: str
    space: Dict = field(default_factory=dict)
    cv_scheme: str = "GroupKFold(groups=bin_id) inside a temporal outer split"
    estimator: object = None

    def as_dict(self) -> Dict:
        return {
            "estimator": self.estimator_name,
            "best_params": self.best_params,
            "best_cv_score": round(float(self.best_cv_score), 5),
            "cv_score_std": round(float(self.cv_score_std), 5),
            "n_candidates": self.n_candidates,
            "n_splits": self.n_splits,
            "search_seconds": round(self.search_seconds, 2),
            "scoring": self.scoring,
            "cv_scheme": self.cv_scheme,
            "search_space": {k: list(map(_jsonable, v)) for k, v in self.space.items()},
        }


def _jsonable(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def temporal_split(n: int, test_frac: float = 0.2):
    """Indices for an ordered temporal hold-out; assumes rows are time-sorted."""
    cut = int(round(n * (1.0 - max(0.0, min(0.9, test_frac)))))
    cut = max(1, min(n - 1, cut))
    return np.arange(cut), np.arange(cut, n)


def _n_splits_for(groups: Sequence, requested: int) -> int:
    """GroupKFold cannot use more folds than there are distinct groups."""
    distinct = len(set(np.asarray(groups).tolist()))
    return max(2, min(int(requested), distinct))


def search_regressor(X, y, groups, space: Optional[Dict] = None,
                     estimator=None, n_iter: int = 30, n_splits: int = 5,
                     seed: int = 42, scoring: str = "r2",
                     n_jobs: int = -1) -> SearchResult:
    """Randomised search for the time-to-overflow regressor."""
    space = space if space is not None else HGB_REGRESSOR_SPACE
    estimator = estimator if estimator is not None else \
        HistGradientBoostingRegressor(random_state=seed)
    name = type(estimator).__name__
    splits = _n_splits_for(groups, n_splits)

    started = time.perf_counter()
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=int(n_iter),
        scoring=scoring,
        cv=GroupKFold(n_splits=splits),
        random_state=seed,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y, groups=groups)
    elapsed = time.perf_counter() - started

    idx = int(search.best_index_)
    return SearchResult(
        estimator_name=name,
        best_params=dict(search.best_params_),
        best_cv_score=float(search.best_score_),
        cv_score_std=float(search.cv_results_["std_test_score"][idx]),
        n_candidates=int(n_iter),
        n_splits=splits,
        search_seconds=elapsed,
        scoring=scoring,
        space=space,
        estimator=search.best_estimator_,
    )


def search_classifier(X, y, groups, space: Optional[Dict] = None,
                      estimator=None, n_iter: int = 25, n_splits: int = 5,
                      seed: int = 42, scoring: str = "roc_auc",
                      n_jobs: int = -1) -> SearchResult:
    """Randomised search for the hazard classifier."""
    space = space if space is not None else HGB_CLASSIFIER_SPACE
    estimator = estimator if estimator is not None else \
        HistGradientBoostingClassifier(random_state=seed)
    name = type(estimator).__name__
    splits = _n_splits_for(groups, n_splits)

    started = time.perf_counter()
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=int(n_iter),
        scoring=scoring,
        cv=GroupKFold(n_splits=splits),
        random_state=seed,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y, groups=groups)
    elapsed = time.perf_counter() - started

    idx = int(search.best_index_)
    return SearchResult(
        estimator_name=name,
        best_params=dict(search.best_params_),
        best_cv_score=float(search.best_score_),
        cv_score_std=float(search.cv_results_["std_test_score"][idx]),
        n_candidates=int(n_iter),
        n_splits=splits,
        search_seconds=elapsed,
        scoring=scoring,
        space=space,
        estimator=search.best_estimator_,
    )


# ---------------------------------------------------------------------------
# Model bake-off under the identical protocol
# ---------------------------------------------------------------------------
def regressor_candidates(seed: int = 42) -> Dict[str, Dict]:
    """The comparator set, each with its own declared space."""
    return {
        "ridge": {"estimator": Ridge(random_state=None), "space": RIDGE_SPACE, "n_iter": 5},
        "random_forest": {"estimator": RandomForestRegressor(random_state=seed, n_jobs=-1),
                          "space": RF_SPACE, "n_iter": 12},
        "hist_gradient_boosting": {"estimator": HistGradientBoostingRegressor(random_state=seed),
                                   "space": HGB_REGRESSOR_SPACE, "n_iter": 30},
    }


def bake_off(X_train, y_train, groups_train, X_test, y_test,
             seed: int = 42, n_splits: int = 5, n_jobs: int = -1) -> Dict:
    """
    Tune and evaluate every candidate under the same protocol.

    Reporting a single model's score says nothing about whether the architecture
    mattered.  Tuning each competitor with the same budget and the same folds is
    the only way that comparison is meaningful.
    """
    results = {}
    for name, cfg in regressor_candidates(seed).items():
        res = search_regressor(
            X_train, y_train, groups_train,
            space=cfg["space"], estimator=clone(cfg["estimator"]),
            n_iter=cfg["n_iter"], n_splits=n_splits, seed=seed, n_jobs=n_jobs,
        )
        pred = res.estimator.predict(X_test)
        results[name] = {
            **res.as_dict(),
            "test_r2": round(float(r2_score(y_test, pred)), 5),
            "test_mae_h": round(float(mean_absolute_error(y_test, pred)), 5),
        }
    # Naive baselines make the learned numbers interpretable.
    persistence = np.full_like(np.asarray(y_test, dtype=float), float(np.mean(y_train)))
    results["baseline_mean"] = {
        "estimator": "mean predictor",
        "test_r2": round(float(r2_score(y_test, persistence)), 5),
        "test_mae_h": round(float(mean_absolute_error(y_test, persistence)), 5),
    }
    return results


def classifier_report(estimator, X_test, y_test) -> Dict:
    """Discrimination *and* calibration -- an AUC alone hides a miscalibrated model."""
    proba = estimator.predict_proba(X_test)[:, 1]
    out = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 5),
        "average_precision": round(float(average_precision_score(y_test, proba)), 5),
        "brier": round(float(brier_score_loss(y_test, proba)), 5),
        "positive_rate": round(float(np.mean(y_test)), 5),
    }
    # A Brier score is only impressive relative to always predicting the base rate.
    base = float(np.mean(y_test))
    baseline_brier = float(np.mean((np.full_like(proba, base) - y_test) ** 2))
    out["brier_skill_score"] = round(
        float(1.0 - out["brier"] / baseline_brier) if baseline_brier > 1e-12 else 0.0, 5)
    return out


def describe_protocol() -> Dict:
    """The methodology statement, embedded in every trained artefact."""
    return {
        "outer_split": "temporal hold-out; the most recent 20% of records, never used in search",
        "inner_cv": "GroupKFold with bin identifier as the group",
        "search": "RandomizedSearchCV with a fixed draw budget and fixed seed",
        "refit": "best configuration refitted on the full training portion",
        "reported": "search score and untouched test score reported separately",
        "rationale": {
            "temporal_outer": "waste generation has weekly and seasonal structure; a random "
                              "split leaks the future into training",
            "grouped_inner": "prevents tuning on the same bin used for scoring, so the score "
                             "reflects generalisation to a newly installed bin",
            "randomised": "the influential axes are continuous and unequally important, which "
                          "favours random draws over a grid at equal budget",
        },
        "spaces": {
            "hist_gradient_boosting_regressor": {k: list(map(_jsonable, v))
                                                 for k, v in HGB_REGRESSOR_SPACE.items()},
            "hist_gradient_boosting_classifier": {k: list(map(_jsonable, v))
                                                  for k, v in HGB_CLASSIFIER_SPACE.items()},
            "random_forest": {k: list(map(_jsonable, v)) for k, v in RF_SPACE.items()},
            "ridge": {k: list(map(_jsonable, v)) for k, v in RIDGE_SPACE.items()},
        },
    }
