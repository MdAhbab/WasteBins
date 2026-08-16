"""
Model artefacts: loading, caching, prediction and the continual corrector.

Two hard rules are enforced here, because both were requirements from the
operator side:

1. **No batch training on the request path.**  ``train_forward`` is a management
   command.  Nothing in this module fits a gradient-boosting model; the most it
   ever does online is one bounded gradient step on a small linear corrector.
2. **Predictions are attributable.**  Every artefact is registered with its
   version, hyperparameters, validation scheme and metrics, so an audit can
   establish which model produced a given dispatch decision.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from django.conf import settings
from joblib import dump, load

from wastebins_core import continual as CORE_CONTINUAL
from wastebins_core import features as CORE_FEATURES
from wastebins_core import xai as CORE_XAI

_LOCK = threading.RLock()
_CACHE: Dict[str, object] = {}
_CACHE_MTIME: Dict[str, float] = {}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def forward_path() -> Path:
    return Path(settings.FORWARD_MODEL_FILENAME)


def forward_meta_path() -> Path:
    return Path(settings.FORWARD_MODEL_META_FILENAME)


def continual_path() -> Path:
    return Path(settings.CONTINUAL_MODEL_FILENAME)


def continual_meta_path() -> Path:
    return Path(settings.CONTINUAL_META_FILENAME)


def file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Forward bundle
# ---------------------------------------------------------------------------
def save_forward_bundle(bundle: Dict, meta: Dict) -> None:
    path = forward_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(bundle, path)
    meta = dict(meta)
    meta["artifact_sha256"] = file_sha256(path)
    forward_meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with _LOCK:
        _CACHE.pop("forward", None)
        _CACHE_MTIME.pop("forward", None)


def load_forward_bundle() -> Optional[Dict]:
    """Load and cache the bundle, reloading automatically when the file changes."""
    path = forward_path()
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    with _LOCK:
        if _CACHE.get("forward") is not None and _CACHE_MTIME.get("forward") == mtime:
            return _CACHE["forward"]
    try:
        bundle = load(path)
    except Exception:
        return None
    with _LOCK:
        _CACHE["forward"] = bundle
        _CACHE_MTIME["forward"] = mtime
    return bundle


def load_forward_meta() -> Dict:
    path = forward_meta_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_model_version() -> str:
    meta = load_forward_meta()
    return meta.get("version", "untrained")


# ---------------------------------------------------------------------------
# Continual corrector
# ---------------------------------------------------------------------------
def _budget() -> CORE_CONTINUAL.Budget:
    return CORE_CONTINUAL.Budget.from_settings(getattr(settings, "ML_SERVING", {}))


def get_continual_learner(n_features: Optional[int] = None
                          ) -> CORE_CONTINUAL.ContinualResidualLearner:
    """Process-wide corrector, restored from disk on first use."""
    n_features = int(n_features or len(CORE_FEATURES.FEATURE_COLS))
    with _LOCK:
        learner = _CACHE.get("continual")
        if isinstance(learner, CORE_CONTINUAL.ContinualResidualLearner) \
                and learner.n_features == n_features:
            return learner
        learner = CORE_CONTINUAL.ContinualResidualLearner(n_features, budget=_budget())
        path = continual_path()
        if path.exists():
            try:
                learner.load_state_dict(load(path))
            except Exception:
                pass
        _CACHE["continual"] = learner
        return learner


def save_continual_learner() -> bool:
    with _LOCK:
        learner = _CACHE.get("continual")
    if not isinstance(learner, CORE_CONTINUAL.ContinualResidualLearner):
        return False
    path = continual_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(learner.state_dict(), path)
    continual_meta_path().write_text(json.dumps(learner.status(), indent=2), encoding="utf-8")
    return True


def reset_continual_learner() -> None:
    with _LOCK:
        _CACHE.pop("continual", None)
    for path in (continual_path(), continual_meta_path()):
        if path.exists():
            path.unlink()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _as_matrix(rows: Sequence[Dict[str, float]], columns: Sequence[str]) -> np.ndarray:
    return np.array([[float(r.get(c, 0.0)) for c in columns] for r in rows], dtype=float)


def predict_batch(feature_rows: Sequence[Dict[str, float]],
                  use_continual: bool = True) -> List[Dict]:
    """
    Score a batch of feature rows.

    Batched deliberately: the gradient-boosting predict call has a fixed
    per-invocation overhead that dominates for single rows, so scoring 30 bins
    one at a time costs far more than scoring them together.
    """
    bundle = load_forward_bundle()
    if bundle is None or not feature_rows:
        return [{} for _ in feature_rows]

    columns = bundle.get("features", CORE_FEATURES.FEATURE_COLS)
    X = _as_matrix(feature_rows, columns)

    regressor = bundle["regressor"]
    tto = np.asarray(regressor.predict(X), dtype=float)

    if use_continual and getattr(settings, "ML_SERVING", {}).get("CONTINUAL_ENABLED", True):
        learner = get_continual_learner(X.shape[1])
        if learner.active:
            tto = np.array([learner.predict(X[i], tto[i]) for i in range(X.shape[0])],
                           dtype=float)

    cap = float(bundle.get("tto_cap_h", CORE_FEATURES.TTO_CAP_H))
    tto = np.clip(tto, 0.0, cap)

    p10 = np.asarray(bundle["q10"].predict(X), dtype=float) if bundle.get("q10") is not None else tto
    p90 = np.asarray(bundle["q90"].predict(X), dtype=float) if bundle.get("q90") is not None else tto
    p10 = np.clip(p10, 0.0, cap)
    p90 = np.clip(p90, 0.0, cap)
    # Quantile heads are fitted independently and can cross; enforce the order
    # rather than emitting an interval whose lower bound exceeds its upper.
    lo = np.minimum(p10, p90)
    hi = np.maximum(p10, p90)

    classifier = bundle.get("classifier")
    hazard = (np.asarray(classifier.predict_proba(X)[:, 1], dtype=float)
              if classifier is not None else np.zeros(X.shape[0]))

    out: List[Dict] = []
    for i in range(X.shape[0]):
        urgency = 1.0 - min(1.0, max(0.0, float(lo[i]) / cap))
        out.append({
            "time_to_overflow_h": round(float(tto[i]), 3),
            "tto_p10_h": round(float(lo[i]), 3),
            "tto_p90_h": round(float(hi[i]), 3),
            "hazard_prob": round(float(hazard[i]), 4),
            "risk_priority": round(float(max(hazard[i], urgency)), 4),
            "model_version": load_forward_meta().get("version", "unknown"),
        })
    return out


def predict_one(feature_row: Dict[str, float], use_continual: bool = True) -> Dict:
    result = predict_batch([feature_row], use_continual=use_continual)
    return result[0] if result else {}


def observe_outcomes(samples: Sequence[Tuple[Dict[str, float], float]]) -> Dict:
    """
    Feed realised outcomes to the continual corrector under its budget.

    ``samples`` are ``(feature_row, observed_time_to_overflow_hours)``.  Called
    from the ingestion path, never from a page render.
    """
    bundle = load_forward_bundle()
    if bundle is None or not samples:
        return {"applied": False, "reason": "no_model"}

    columns = bundle.get("features", CORE_FEATURES.FEATURE_COLS)
    X = _as_matrix([row for row, _ in samples], columns)
    base = np.asarray(bundle["regressor"].predict(X), dtype=float)

    learner = get_continual_learner(X.shape[1])
    batch = [(X[i], float(samples[i][1]), float(base[i])) for i in range(X.shape[0])]
    report = learner.update(batch)
    if report.applied:
        save_continual_learner()
    return report.as_dict()


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------
def background_matrix(bundle: Dict, n: int = 128) -> np.ndarray:
    """Background reference for attribution, cached with the bundle."""
    stored = bundle.get("background")
    if stored is not None:
        arr = np.asarray(stored, dtype=float)
        if arr.ndim == 2 and arr.shape[0] > 0:
            return arr[:n]
    return np.zeros((1, len(bundle.get("features", CORE_FEATURES.FEATURE_COLS))))


def explain_prediction(feature_row: Dict[str, float], head: str = "tto",
                       prefer: str = "auto", top_k: int = 8) -> Dict:
    """
    Attribute one prediction.

    ``head`` selects ``"tto"`` (hours to overflow) or ``"hazard"`` (probability).
    When the continual corrector is active the *composite* function is
    explained, so the attribution describes what actually ran.
    """
    bundle = load_forward_bundle()
    if bundle is None:
        return {"error": "no trained model available"}

    columns = list(bundle.get("features", CORE_FEATURES.FEATURE_COLS))
    x = np.array([float(feature_row.get(c, 0.0)) for c in columns], dtype=float)
    background = background_matrix(bundle)

    if head == "hazard" and bundle.get("classifier") is not None:
        model = bundle["classifier"]

        def predict(matrix):
            return model.predict_proba(np.atleast_2d(matrix))[:, 1]

        target, units = "hazard probability", ""
    else:
        model = bundle["regressor"]
        learner = get_continual_learner(len(columns))
        use_corrector = learner.active and \
            getattr(settings, "ML_SERVING", {}).get("CONTINUAL_ENABLED", True)

        def predict(matrix):
            matrix = np.atleast_2d(matrix)
            base = np.asarray(model.predict(matrix), dtype=float)
            if not use_corrector:
                return base
            return np.array([learner.predict(matrix[i], base[i])
                             for i in range(matrix.shape[0])], dtype=float)

        target, units = "time to overflow", "h"

    budget = getattr(settings, "ML_SERVING", {}).get("EXPLAIN_MAX_PERTURBATIONS", 1000)
    started = time.perf_counter()
    explanation = CORE_XAI.explain(
        model, x, background, columns, predict=predict, prefer=prefer,
        top_k=top_k, n_samples=int(budget), target=target, units=units,
    )
    payload = explanation.as_dict()
    payload["head"] = head
    payload["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    payload["model_version"] = load_forward_meta().get("version", "unknown")
    return payload


def global_importance() -> List[Dict]:
    """Permutation importance stored at training time; empty until trained."""
    return load_forward_meta().get("global_importance", [])


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def status() -> Dict:
    bundle = load_forward_bundle()
    meta = load_forward_meta()
    learner = get_continual_learner()
    return {
        "forward_model": {
            "available": bundle is not None,
            "version": meta.get("version", "untrained"),
            "trained_at": meta.get("trained_at"),
            "metrics": meta.get("metrics", {}),
            "hyperparameters": meta.get("hyperparameters", {}),
            "validation": meta.get("validation"),
            "n_features": len(bundle.get("features", [])) if bundle else 0,
            "artifact_sha256": meta.get("artifact_sha256", ""),
            "training_rows": meta.get("training_rows", 0),
        },
        "continual": learner.status(),
        "serving_policy": {
            "inline_batch_training": bool(
                getattr(settings, "ML_SERVING", {}).get("ALLOW_INLINE_BATCH_TRAINING", False)),
            "note": "Batch training runs only via `manage.py train_forward`. The request "
                    "path performs at most one bounded gradient step on a small linear "
                    "residual corrector.",
        },
        "explainability": {
            "local_default": "kernel_shap",
            "treeshap_available": CORE_XAI.shap_available(),
            "global_importance_available": bool(meta.get("global_importance")),
        },
    }
