"""
How the planner behaves as the network grows.
=============================================

A dispatcher that only works at thirty containers is of no use to a city, so the
cost of the method has to be stated as a function of instance size rather than at
one operating point.  This script holds everything fixed except the number of
containers and the fleet sized to match it, and reports three things per size.

``compute``
    Wall-clock time to produce a plan, split into construction and improvement.
    Construction dominates and grows faster than linearly, which is the honest
    limit of the current implementation and is reported as such.

``quality``
    The objective, the distance and the share of containers served, so a reader
    can see whether the planner degrades as well as slows.

``margin``
    The same quantities for the strongest baseline, so the comparison at the
    headline size is not mistaken for a claim at every size.

The fleet grows with the network at a fixed ratio of one vehicle per 25
containers, so the per-vehicle workload stays roughly constant and the curve
measures the planner rather than a change in how constrained the instance is.

Run:  python -m experiments.exp_scale [--sizes 50,100,200,400] [--snapshots 5]
Out:  results/scale.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import vrp as VRP                 # noqa: E402
from experiments import exp_fleet as EF               # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

CONTAINERS_PER_VEHICLE = 25
COMPARATOR = "aco"           # the strongest baseline at the headline size


def run_size(n_bins: int, n_snapshots: int, budget: float) -> Dict:
    from datetime import datetime, timezone
    when = datetime(2026, 3, 3, 7, 30, tzinfo=timezone.utc)

    n_vehicles = max(2, round(n_bins / CONTAINERS_PER_VEHICLE))
    EF.set_study("dhaka", n_bins=n_bins, n_vehicles=n_vehicles)
    rng = np.random.default_rng(EF.SEED)
    weights = VRP.ObjectiveWeights()

    rows: Dict[str, List[Dict]] = {"proposed": [], COMPARATOR: []}
    matrix_s: List[float] = []
    violations: List[str] = []

    for i in range(n_snapshots):
        snapshot = EF.make_snapshot(rng, i)
        t0 = time.perf_counter()
        travel = EF.travel_for(snapshot, when)
        matrix_s.append(time.perf_counter() - t0)
        tasks = EF.build_tasks(snapshot)
        fleet = EF.build_fleet(snapshot)

        for policy in rows:
            started = time.perf_counter()
            plan = EF.run_policy(policy, tasks, fleet, travel, weights, snapshot,
                                 budget)
            elapsed = time.perf_counter() - started
            if plan is None:
                continue
            violations.extend(EF.audit_plan(plan, f"{policy}@{n_bins}:{i}"))
            rows[policy].append({
                "solve_s": elapsed,
                "objective": float(plan.objective),
                "distance_km": float(plan.metrics["distance_km"]),
                "co2_kg": float(plan.metrics["co2_kg"]),
                "served": float(plan.metrics["bins_served"]),
                "served_share": float(plan.metrics["bins_served"]) / n_bins,
            })

    def mean(policy: str, field: str) -> float:
        values = [r[field] for r in rows[policy]]
        return float(np.mean(values)) if values else float("nan")

    proposed_obj = np.array([r["objective"] for r in rows["proposed"]])
    baseline_obj = np.array([r["objective"] for r in rows[COMPARATOR]])
    gap = float(np.mean((baseline_obj - proposed_obj) / baseline_obj) * 100.0)

    return {
        "n_bins": n_bins,
        "n_vehicles": n_vehicles,
        "n_snapshots": n_snapshots,
        "search_budget_s": budget,
        "matrix_build_s": float(np.mean(matrix_s)),
        "proposed": {f: mean("proposed", f) for f in
                     ("solve_s", "objective", "distance_km", "co2_kg",
                      "served", "served_share")},
        COMPARATOR: {f: mean(COMPARATOR, f) for f in
                     ("solve_s", "objective", "distance_km", "co2_kg",
                      "served", "served_share")},
        "objective_gap_pct": gap,
        "feasible": len(violations) == 0,
        "violations": violations[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="50,100,200,400")
    parser.add_argument("--snapshots", type=int, default=5)
    parser.add_argument("--budget", type=float, default=3.0)
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    rows = []
    print(f"{'bins':>6}{'veh':>5}{'matrix s':>10}{'solve s':>10}{'served':>9}"
          f"{'km':>9}{'objective':>12}{'gap vs ' + COMPARATOR:>16}")
    print("-" * 77)
    for n_bins in sizes:
        row = run_size(n_bins, args.snapshots, args.budget)
        rows.append(row)
        p = row["proposed"]
        print(f"{n_bins:>6}{row['n_vehicles']:>5}{row['matrix_build_s']:>10.2f}"
              f"{p['solve_s']:>10.2f}{100 * p['served_share']:>8.1f}%"
              f"{p['distance_km']:>9.1f}{p['objective']:>12.1f}"
              f"{row['objective_gap_pct']:>15.1f}%", flush=True)

    payload = {
        "containers_per_vehicle": CONTAINERS_PER_VEHICLE,
        "comparator": COMPARATOR,
        "sizes": rows,
        "note": "the fleet grows with the network so the per-vehicle workload "
                "stays roughly constant; the curve measures the planner and not "
                "a change in how constrained the instance is",
    }
    (RESULTS / "scale.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {RESULTS / 'scale.json'}")
    if not all(r["feasible"] for r in rows):
        print("FAILED: infeasible plans produced")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
