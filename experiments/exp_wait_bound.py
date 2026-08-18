"""
Is the worst-case wait bound tight, or merely true?
===================================================

A bound that is never approached tells an operator very little.  This experiment
asks the sharper question: for a container the planner finds expensive to reach,
is it collected at exactly the hour `aging.worst_case_wait_bound` predicts?

The setup is deliberately the smallest one that can answer it.  One depot, one
container, one vehicle with a shift long enough that time is never the binding
constraint.  The container's distance from the depot is swept, which sweeps its
marginal insertion cost, and at each distance its wait is advanced one dispatch
cycle at a time until the planner collects it.  Nothing else varies.

This is the experiment that would have caught the defect it now guards.  Pricing
the skip on the bounded ordering score `w/(tau + w)` capped the penalty at
`lambda_prize * overdue_multiplier`, or 180 at the default weights.  Every
container whose insertion cost exceeded that was skipped at every wait, including
a wait of a million hours, while the tier ordering worked perfectly throughout.
Run this file against that version and the `served_at_h` column reads `never`
from about 33 km outward.

Out: results/wait_bound.json
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wastebins_core import aging as AG      # noqa: E402
from wastebins_core import scenario as SC    # noqa: E402
from wastebins_core import vrp as VRP        # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

DISTANCES_KM = (10.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 100.0)
CYCLE_H = 12.0
MAX_WAIT_H = 4800.0          # 400 cycles, far past any predicted collection
SPEED_KMH = 20.0


def one_task(wait_h: float, tau_h: float = AG.DEFAULT_TAU_H) -> VRP.BinTask:
    """
    Build the container through the production path, not by hand.

    Constructing a `BinTask` directly would let this experiment set
    `overdue_pressure` itself, which is exactly the field under test.  Going
    through `effective_priorities` and `make_tasks` means the experiment measures
    the wiring the service uses rather than a convenient reconstruction of it.
    """
    tiered = AG.effective_priorities({1: 0.30}, {1: wait_h}, {1: 0.0}, tau_h=tau_h)
    return SC.make_tasks(
        [1], {1: 0.5}, {1: tiered[1].score}, index_of={1: 1},
        hazards={1: False},
        tiers={1: tiered[1].tier},
        overdue_pressures={1: tiered[1].pressure},
        densities={1: 220.0}, capacities_l={1: 1100.0},
    )[0]


def travel_for(km: float) -> VRP.TravelModel:
    matrix = np.array([[0.0, km * 1000.0], [km * 1000.0, 0.0]])
    return VRP.TravelModel(matrix, default_speed_kmh=SPEED_KMH)


def sweep(weights: VRP.ObjectiveWeights, tau_h: float = AG.DEFAULT_TAU_H) -> list:
    rows = []
    for km in DISTANCES_KM:
        travel = travel_for(km)
        vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0, shift_minutes=1200.0)

        evaluated = VRP.evaluate_route([one_task(tau_h, tau_h)], vehicle, travel)
        insertion_cost = (VRP.route_cost(evaluated, weights)
                          if evaluated is not None else math.inf)

        predicted = AG.worst_case_wait_bound(
            tau_h=tau_h, cycle_h=CYCLE_H, max_overdue=1,
            served_overdue_per_cycle=1, max_insertion_cost=insertion_cost,
            lambda_prize=weights.lambda_prize,
            overdue_multiplier=weights.overdue_multiplier)

        served_at = None
        wait = tau_h
        while wait <= MAX_WAIT_H:
            plan = VRP.solve([one_task(wait, tau_h)], [vehicle], travel, weights,
                             time_budget_s=0.4)
            if any(route.stops for route in plan.routes):
                served_at = wait
                break
            wait += CYCLE_H

        rows.append({
            "distance_km": km,
            "insertion_cost": round(float(insertion_cost), 3),
            "predicted_bound_h": None if not math.isfinite(predicted) else round(predicted, 1),
            "served_at_h": served_at,
            "tight": served_at is not None and abs(served_at - predicted) < 1e-9,
            "holds": served_at is not None and served_at <= predicted + 1e-9,
        })
    return rows


def main() -> None:
    weights = VRP.ObjectiveWeights()
    rows = sweep(weights)

    print("Is the wait bound tight?")
    print("=" * 72)
    print(f"  one container, one vehicle, cycle {CYCLE_H:.0f} h, "
          f"tau {AG.DEFAULT_TAU_H:.0f} h")
    print(f"  penalty ceiling under the retired pricing: "
          f"{weights.lambda_prize * weights.overdue_multiplier:.0f}")
    print()
    print(f"{'km':>6}{'insertion cost':>16}{'predicted h':>14}"
          f"{'served at h':>14}{'verdict':>10}")
    print("-" * 60)
    for r in rows:
        served = "never" if r["served_at_h"] is None else f"{r['served_at_h']:.0f}"
        verdict = "tight" if r["tight"] else ("holds" if r["holds"] else "BREACH")
        print(f"{r['distance_km']:>6.0f}{r['insertion_cost']:>16.2f}"
              f"{r['predicted_bound_h']:>14.0f}{served:>14}{verdict:>10}")

    n_tight = sum(1 for r in rows if r["tight"])
    n_holds = sum(1 for r in rows if r["holds"])
    print("-" * 60)
    print(f"  {n_holds} of {len(rows)} respect the bound, "
          f"{n_tight} of {len(rows)} attain it exactly")

    payload = {
        "protocol": {
            "cycle_h": CYCLE_H,
            "tau_h": AG.DEFAULT_TAU_H,
            "speed_kmh": SPEED_KMH,
            "max_wait_searched_h": MAX_WAIT_H,
            "weights": {
                "lambda_prize": weights.lambda_prize,
                "overdue_multiplier": weights.overdue_multiplier,
            },
            "retired_penalty_ceiling":
                weights.lambda_prize * weights.overdue_multiplier,
            "note": "the container is built through effective_priorities and "
                    "make_tasks so the wiring under test is the one the service "
                    "uses, not a reconstruction of it",
        },
        "sweep": rows,
        "n_holds": n_holds,
        "n_tight": n_tight,
        "n_cases": len(rows),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "wait_bound.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {RESULTS / 'wait_bound.json'}")


if __name__ == "__main__":
    main()
