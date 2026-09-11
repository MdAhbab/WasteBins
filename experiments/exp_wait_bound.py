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

# Waits used to show that the retired pricing saturates.  A million hours is not
# a realistic wait; it is the point of the demonstration.  Under the retired
# scheme the penalty converges to lambda_prize * overdue_multiplier from below and
# never reaches it, so a bin costing more than that to insert is skipped at every
# wait, and no finite horizon can be searched to prove it.  The analytic statement
# and the measurement have to agree, and this is where that is checked.
SATURATION_WAITS_H = (48.0, 96.0, 240.0, 1e3, 1e4, 1e6)


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
            # The reservation rule is switched off here, and it has to be.
            # This sweep asks a question about the *price*: at what wait does
            # the escalating skip penalty of `eq:skip` first exceed the cost of
            # a dedicated round trip? There is one container in the instance, so
            # the rule would mark it unskippable and the planner would collect
            # it the moment it was promoted. The sweep would then report the
            # promotion threshold in every row and appear to confirm a bound it
            # had stopped testing. The two mechanisms are measured separately:
            # this one covers the affordability term, and the rollout in
            # `exp_fleet` covers the queueing term the rule delivers.
            plan = VRP.solve([one_task(wait, tau_h)], [vehicle], travel, weights,
                             time_budget_s=0.4, reserve_overdue=False)
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


def retired_pricing_saturation(weights: VRP.ObjectiveWeights,
                               tau_h: float = AG.DEFAULT_TAU_H) -> dict:
    """
    Show that the retired pricing has a ceiling, and what it costs.

    The retired scheme priced skipping on the ordering score `w/(tau + w)`, which
    is bounded above by 1, so the penalty is bounded by
    `lambda_prize * overdue_multiplier`.  This evaluates that penalty directly at
    waits up to a million hours and reports the supremum it approaches, together
    with the distance at which a bin becomes too expensive to be worth serving
    under it.  Both are stated in the paper and neither was reproducible before.
    """
    ceiling = weights.lambda_prize * weights.overdue_multiplier
    retired = []
    for wait in SATURATION_WAITS_H:
        score = AG.overdue_score(wait, tau_h)          # the retired price basis
        retired.append({
            "wait_h": wait,
            "ordering_score": round(float(score), 9),
            "retired_skip_cost": round(float(weights.lambda_prize * score
                                             * weights.overdue_multiplier), 6),
            "current_skip_cost": round(float(weights.lambda_prize
                                             * AG.overdue_pressure(wait, tau_h)
                                             * weights.overdue_multiplier), 6),
        })

    # The distance at which a dedicated round trip costs more than the ceiling.
    first_unaffordable = None
    for km in [float(k) for k in range(5, 101)]:
        matrix = np.array([[0.0, km * 1000.0], [km * 1000.0, 0.0]])
        travel = VRP.TravelModel(matrix, default_speed_kmh=SPEED_KMH)
        vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0, shift_minutes=1200.0)
        evaluated = VRP.evaluate_route([one_task(tau_h, tau_h)], vehicle, travel)
        if evaluated is None:
            continue
        if VRP.route_cost(evaluated, weights) > ceiling:
            first_unaffordable = km
            break

    return {
        "ceiling": ceiling,
        "note": "the retired skip cost approaches the ceiling from below and never "
                "reaches it, so a bin whose insertion cost exceeds the ceiling is "
                "skipped at every wait, however large; this is an analytic "
                "property, and the table below evaluates it to 1e6 hours",
        "by_wait": retired,
        "first_unaffordable_km": first_unaffordable,
    }


def main() -> None:
    weights = VRP.ObjectiveWeights()
    rows = sweep(weights)
    saturation = retired_pricing_saturation(weights)

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

    print()
    print("Why the retired pricing could not do this")
    print("=" * 72)
    print(f"  ceiling = lambda_prize * overdue_multiplier = {saturation['ceiling']:.0f}")
    print(f"  a dedicated round trip first exceeds it at "
          f"{saturation['first_unaffordable_km']} km from the depot")
    print(f"{'wait h':>12}{'retired cost':>16}{'current cost':>16}")
    print("-" * 44)
    for r in saturation["by_wait"]:
        print(f"{r['wait_h']:>12.0f}{r['retired_skip_cost']:>16.4f}"
              f"{r['current_skip_cost']:>16.2f}")
    print("-" * 44)
    print(f"  retired cost never reaches {saturation['ceiling']:.0f}; "
          f"current cost grows without limit")

    payload = {
        "retired_pricing_saturation": saturation,
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
