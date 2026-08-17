"""
Continual learning under non-stationarity.
==========================================

A deployed bin network does not hold still: sensors age, collection schedules
change, neighbourhoods gentrify, and monsoon season changes waste composition.
A model frozen at training time decays. Retraining the ensemble continuously
would fix that but is far too expensive to sit on the request path — and the
operator must not pay a training cost while using the application.

The design separates the two timescales, and this experiment tests whether the
separation actually works:

* a **slow path** trained offline, frozen at serving time;
* a **fast path** — a small linear model learning the residual of the frozen
  ensemble — bounded to a few hundred samples and a few tens of milliseconds per
  update.

Three properties are measured, and all three matter:

1. **It helps when the world changes.** Prequential error after an abrupt or
   gradual shift, against the frozen model.
2. **It does no harm when the world does not.** On a stationary stream the
   corrector must stay dormant. A continual learner that degrades a healthy
   model is worse than none, and this is the property such systems most often
   fail.
3. **It is cheap enough to run on the request path.** Measured per-prediction
   and per-update latency against the declared budget.

Protocol: prequential (test-then-train). Every sample is predicted *before* it
is learned from, so the reported error is what the deployed system would
actually have achieved as the stream arrived.

Run:  python exp_continual.py
Out:  results/continual.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import continual as CL   # noqa: E402
from wastebins_core import features as FT    # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
N_TRAIN = 1500
N_STREAM = 3000
SEEDS = (0, 1, 2, 3, 4)


def synthetic_task(rng: np.random.Generator, n: int, n_features: int):
    """
    A non-linear regression standing in for time-to-overflow.

    Deliberately includes an interaction and a quadratic term so a linear model
    cannot solve it outright — otherwise the linear corrector would look far
    better than it deserves.
    """
    X = rng.normal(0, 1, (n, n_features))
    y = (3.0 * X[:, 0]
         - 2.0 * X[:, 5]
         + 0.8 * X[:, 10] ** 2
         + 1.2 * X[:, 1] * X[:, 2]
         + rng.normal(0, 0.4, n))
    return X, y


DRIFT_SCENARIOS = {
    "stationary": lambda y, t, n: y,
    "abrupt_small": lambda y, t, n: y + np.where(t >= n // 2, 1.5, 0.0),
    "abrupt_large": lambda y, t, n: y + np.where(t >= n // 2, 6.0, 0.0),
    "gradual": lambda y, t, n: y + 5.0 * (t / n),
    "seasonal": lambda y, t, n: y + 3.0 * np.sin(2 * np.pi * t / (n / 3.0)),
    "transient": lambda y, t, n: y + np.where((t >= n // 3) & (t < 2 * n // 3), 4.0, 0.0),
    "recurring": lambda y, t, n: y + np.where((t // (n // 6)) % 2 == 1, 3.0, 0.0),
}


def run_scenario(name: str, seed: int, n_features: int) -> Dict:
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(seed)
    X_train, y_train = synthetic_task(rng, N_TRAIN, n_features)
    base = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06,
                                         random_state=seed).fit(X_train, y_train)

    X_stream, y_clean = synthetic_task(rng, N_STREAM, n_features)
    t = np.arange(N_STREAM)
    y_stream = DRIFT_SCENARIOS[name](y_clean, t, N_STREAM)

    base_predictions = base.predict(X_stream)
    learner = CL.ContinualResidualLearner(n_features, warmup_samples=100, seed=seed)

    started = time.perf_counter()
    report = CL.prequential_evaluation(learner, X_stream, y_stream,
                                       base_predictions, report_every=0)
    report["wall_clock_s"] = round(time.perf_counter() - started, 3)
    report["scenario"] = name
    report["seed"] = seed
    report["needs_retrain"] = bool(learner.needs_retrain)
    report["corrector_active_at_end"] = bool(learner.active)
    return report


def latency_study(n_features: int) -> Dict:
    """Per-prediction and per-update cost against the declared budget."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(SEED)
    X_train, y_train = synthetic_task(rng, N_TRAIN, n_features)
    base = HistGradientBoostingRegressor(max_iter=200, random_state=SEED).fit(X_train, y_train)

    X, y = synthetic_task(rng, 2000, n_features)
    predictions = base.predict(X)
    learner = CL.ContinualResidualLearner(n_features, warmup_samples=50, seed=SEED)

    # Warm the corrector so the measured path is the active one.
    for i in range(600):
        learner.observe(X[i], y[i] + 4.0, predictions[i])

    started = time.perf_counter()
    for i in range(1000):
        learner.predict(X[i], predictions[i])
    per_prediction_us = (time.perf_counter() - started) / 1000 * 1e6

    batch = [(X[i], float(y[i]), float(predictions[i])) for i in range(256)]
    learner.last_update_ts = 0.0            # bypass the rate limit for measurement
    update = learner.update(batch)

    return {
        "per_corrected_prediction_us": round(per_prediction_us, 2),
        "bounded_update_ms": round(update.elapsed_ms, 3),
        "samples_in_update": update.n_samples,
        "budget_ms": learner.budget.max_ms_per_update,
        "budget_samples": learner.budget.max_samples_per_update,
        "within_budget": update.elapsed_ms <= learner.budget.max_ms_per_update,
        "note": "Batch training never runs on the request path. This is the entire online "
                "cost the deployed service pays.",
    }


def main() -> None:
    n_features = len(FT.FEATURE_COLS)
    print(f"Continual learning study — {n_features} features, "
          f"{N_STREAM} streamed samples, {len(SEEDS)} seeds")
    print("=" * 84)

    scenarios: Dict[str, Dict] = {}
    print(f"\n{'scenario':<16}{'base MAE':>11}{'continual MAE':>16}"
          f"{'improvement':>14}{'drifts':>9}{'retrain?':>10}")
    print("-" * 84)

    for name in DRIFT_SCENARIOS:
        runs = [run_scenario(name, seed, n_features) for seed in SEEDS]
        base_mae = float(np.mean([r["prequential_mae_base"] for r in runs]))
        cont_mae = float(np.mean([r["prequential_mae_continual"] for r in runs]))
        improvement = float(np.mean([r["improvement_pct"] for r in runs]))
        drifts = float(np.mean([r["drift_detections"] for r in runs]))
        retrain = sum(1 for r in runs if r["needs_retrain"])

        scenarios[name] = {
            "prequential_mae_base": round(base_mae, 4),
            "prequential_mae_continual": round(cont_mae, 4),
            "improvement_pct": round(improvement, 3),
            "improvement_pct_std": round(
                float(np.std([r["improvement_pct"] for r in runs])), 3),
            "mean_drift_detections": round(drifts, 2),
            "seeds_flagging_retrain": retrain,
            "n_seeds": len(SEEDS),
            "per_seed_improvement_pct": [round(r["improvement_pct"], 3) for r in runs],
        }
        print(f"{name:<16}{base_mae:>11.4f}{cont_mae:>16.4f}"
              f"{improvement:>13.2f}%{drifts:>9.1f}{retrain:>7}/{len(SEEDS)}")

    latency = latency_study(n_features)
    print(f"\nServing cost")
    print(f"  corrected prediction : {latency['per_corrected_prediction_us']:.2f} µs")
    print(f"  bounded update       : {latency['bounded_update_ms']:.2f} ms "
          f"for {latency['samples_in_update']} samples "
          f"(budget {latency['budget_ms']:.0f} ms) "
          f"— {'within budget' if latency['within_budget'] else 'OVER BUDGET'}")

    stationary = scenarios["stationary"]
    harmless = abs(stationary["improvement_pct"]) < 1.0
    print(f"\nDo-no-harm check on a stationary stream: "
          f"{stationary['improvement_pct']:+.2f}% "
          f"({'PASS — corrector stayed dormant' if harmless else 'FAIL — corrector interfered'})")

    payload = {
        "protocol": {
            "evaluation": "prequential (test-then-train); every sample predicted before it "
                          "is learned from",
            "n_features": n_features,
            "n_train": N_TRAIN,
            "n_stream": N_STREAM,
            "seeds": list(SEEDS),
            "base_model": "HistGradientBoostingRegressor, frozen at training time",
            "online_model": "linear residual corrector, bounded SGD with ADWIN2 + "
                            "scale-aware Page-Hinkley drift detection and reservoir replay",
            "activation_rule": "the corrector only affects predictions while its running "
                               "prequential error beats the frozen model by a margin",
        },
        "scenarios": scenarios,
        "latency": latency,
        "do_no_harm": {
            "stationary_improvement_pct": stationary["improvement_pct"],
            "passes": harmless,
            "criterion": "|improvement| < 1% on a stationary stream",
        },
    }
    (RESULTS / "continual.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {RESULTS / 'continual.json'}")


if __name__ == "__main__":
    main()
