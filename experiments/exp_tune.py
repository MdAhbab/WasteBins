"""
Choosing solver settings and the prize weight, on containers kept for that.
===========================================================================

Two things in the study were chosen by looking at results, and both are chosen
here on containers that the primary evaluation sample does not contain.

``solvers``
    Each budgeted solver is run under a small grid of its own settings on the
    tuning set ``T160``, at the time the insertion planner takes on the same
    instance.  The setting with the lowest mean objective is kept.  Every solver
    gets the same number of instances and the same time per run, so none is
    tuned harder than another.  The grid is small and is stated in full; this is
    a configuration choice, not an automated search.

``lambda``
    The prize weight is swept on ``T60``.  The value used in the study was fixed
    before any comparison was run, so this does not choose it.  It checks that
    the value still sits where the trade-off between coverage and distance
    turns, on containers that played no part in fixing it.

Run:  python -m experiments.exp_tune --what solvers --shard 0 --of 3
      python -m experiments.exp_tune --what lambda
      python -m experiments.exp_tune --select
Out:  results/raw/tune-*.jsonl, results/tuning.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import vrp as VRP             # noqa: E402
from experiments import exp_fleet as EF           # noqa: E402
from experiments import instances as IN           # noqa: E402
from experiments import policies as PO            # noqa: E402
from experiments import store as ST               # noqa: E402
from experiments.exp_compare import IMPROVE_S, REFERENCE  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

TUNING_SNAPSHOTS = 3
LAMBDAS = (10.0, 25.0, 45.0, 70.0, 100.0, 150.0)
LAMBDA_SNAPSHOTS = 25


def _grid(**axes) -> List[Dict]:
    names = list(axes)
    return [dict(zip(names, values)) for values in itertools.product(*axes.values())]


#: The settings tried for each solver.  The first entry of each is the setting
#: the first version of the study used.
GRIDS: Dict[str, List[Dict]] = {
    "ortools": [dict(PO.DEFAULT_CONFIG["ortools"], **g) for g in _grid(
        first_solution=("PATH_CHEAPEST_ARC", "PARALLEL_CHEAPEST_INSERTION"),
        metaheuristic=("GUIDED_LOCAL_SEARCH", "TABU_SEARCH", "SIMULATED_ANNEALING"))],
    "aco": [dict(PO.DEFAULT_CONFIG["aco"], **g) for g in _grid(
        n_ants=(16, 8), beta=(2.5, 4.0), polish_fraction=(0.0, 0.3))],
    "genetic": [dict(PO.DEFAULT_CONFIG["genetic"], **g) for g in _grid(
        population_size=(40, 20), mutation_rate=(0.20, 0.50),
        polish_fraction=(0.0, 0.3))],
}


def _tag(config: Dict) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def tune_solvers(shard: int, of: int) -> None:
    study = "tune-solvers"
    done = ST.index(study)
    weights = VRP.ObjectiveWeights()
    for index in ST.mine(range(TUNING_SNAPSHOTS), shard, of):
        snapshot = IN.snapshots("T160", TUNING_SNAPSHOTS, 6)[index]
        travel = EF.travel_for(snapshot, IN.WHEN)
        fleet = EF.build_fleet(snapshot)
        canonical = PO.canonical_tasks(snapshot)

        ref_key = f"{study}|T160|{index}|insertion|full"
        if ref_key not in done:
            record = PO.solve_and_record(*REFERENCE, snapshot, travel, fleet, weights,
                                         budget_s=None, improve_s=IMPROVE_S,
                                         canonical=canonical)
            record.update({"key": ref_key, "study": study, "snapshot": index})
            ST.append(study, shard, record)
            done[ref_key] = record
        matched = float(done[ref_key]["elapsed_s"])
        print(f"T160 snapshot {index}: insertion {done[ref_key]['objective']:.1f} "
              f"in {matched:.0f}s", flush=True)

        for solver, grid in GRIDS.items():
            for config in grid:
                key = f"{study}|T160|{index}|{solver}|{_tag(config)}"
                if key in done:
                    continue
                wall = time.perf_counter()
                plan = PO.plan_with(solver, PO.RULES["full"], snapshot, travel, fleet,
                                    weights, budget_s=matched, improve_s=IMPROVE_S,
                                    config={solver: config})
                scored = PO.score(plan, canonical, travel, weights)
                record = {
                    "key": key, "study": study, "snapshot": index,
                    "planner": solver, "config": config,
                    "objective": round(scored["objective"], 4),
                    "served": scored["metrics"]["bins_served"],
                    "km": scored["metrics"]["distance_km"],
                    "elapsed_s": round(time.perf_counter() - wall, 3),
                    "budget_s": round(matched, 3),
                    "iterations": plan.metrics.get("iterations",
                                                   plan.metrics.get("generations")),
                }
                ST.append(study, shard, record)
                done[key] = record
                print(f"  {solver:<8}{record['objective']:>9.1f}  {_tag(config)}",
                      flush=True)


def tune_lambda(shard: int, of: int) -> None:
    study = "tune-lambda"
    done = ST.index(study)
    for index in ST.mine(range(LAMBDA_SNAPSHOTS), shard, of):
        snapshot = IN.snapshots("T60", LAMBDA_SNAPSHOTS, 4)[index]
        travel = EF.travel_for(snapshot, IN.WHEN)
        fleet = EF.build_fleet(snapshot)
        for value in LAMBDAS:
            key = f"{study}|T60|{index}|{value:g}"
            if key in done:
                continue
            weights = VRP.ObjectiveWeights(lambda_prize=value)
            plan = PO.plan_with("insertion", PO.RULES["full"], snapshot, travel, fleet,
                                weights, budget_s=None, improve_s=3.0)
            record = {
                "key": key, "study": study, "snapshot": index, "lambda_prize": value,
                "served": plan.metrics["bins_served"],
                "km": plan.metrics["distance_km"],
                "missed": plan.metrics["missed_overflow"],
                "elapsed_s": round(plan.compute_ms / 1000.0, 3),
            }
            ST.append(study, shard, record)
            done[key] = record
        print(f"T60 snapshot {index} done", flush=True)


def select() -> Dict:
    """Keep, for each solver, the setting with the lowest mean objective."""
    records = [r for r in ST.read("tune-solvers") if "config" in r]
    reference = [r for r in ST.read("tune-solvers") if r.get("planner") == "insertion"]
    table, selected = {}, {}
    for solver, grid in GRIDS.items():
        rows = []
        for config in grid:
            values = [r["objective"] for r in records
                      if r["planner"] == solver and _tag(r["config"]) == _tag(config)]
            if len(values) < TUNING_SNAPSHOTS:
                continue
            rows.append({"config": config, "mean_objective": float(np.mean(values)),
                         "objectives": values})
        if not rows:
            continue
        rows.sort(key=lambda row: row["mean_objective"])
        table[solver] = rows
        selected[solver] = rows[0]["config"]
    payload = {
        "tuning_set": "T160", "snapshots": TUNING_SNAPSHOTS,
        "matched_budget_s": [r["elapsed_s"] for r in reference],
        "insertion_objective": [r["objective"] for r in reference],
        "selected": selected, "grid": table,
        "note": "each solver was run under every setting of its grid on the same "
                "tuning instances at the same matched time; the setting with the "
                "lowest mean objective is kept",
    }
    (RESULTS / "tuning.json").write_text(json.dumps(payload, indent=2))
    for solver, rows in table.items():
        print(solver)
        for row in rows:
            print(f"  {row['mean_objective']:>9.1f}  {_tag(row['config'])}")
    print(f"\nwrote {RESULTS / 'tuning.json'}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", choices=["solvers", "lambda"], default=None)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    args = parser.parse_args()
    ST.keep_awake()
    if args.what == "solvers":
        tune_solvers(args.shard, args.of)
    elif args.what == "lambda":
        tune_lambda(args.shard, args.of)
    if args.select:
        select()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
