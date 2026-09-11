"""
What a planner loses by optimising on approximate distances.
============================================================

Running the routing study twice, once on the street graph and once on
great-circle distance times a constant, compares two different plans each
measured in its own metric. That answers a narrow question: what an operator is
told. It does not answer the question an operator cares about, which is what the
vehicle actually drives.

This script answers that one. For each snapshot it plans twice and scores three
ways:

``believed``
    The round the constant-factor planner produces, measured with constant-factor
    distances. This is the number that planner reports, and it is the number in
    the two-study comparison.

``actual``
    The same plan, the same sequence of containers in the same order, measured on
    the street graph. This is what the vehicle drives if it follows that plan.
    The gap against ``believed`` is the operator's surprise.

``road_planner``
    The round a planner that knew the street graph produces, measured on the
    street graph. The gap against ``actual`` is what optimising on the wrong
    distances costs, as opposed to merely reporting them wrongly.

Separating the two matters because they have different remedies. Reporting error
is fixed by measuring the finished plan properly. Optimisation error is not: the
planner has already chosen a worse sequence, and no amount of re-measurement
recovers it.

Run:  python -m experiments.exp_distance_model [--snapshots 25]
Out:  results/distance_model.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import stats as ST                # noqa: E402
from wastebins_core import traffic as TR              # noqa: E402
from wastebins_core import vrp as VRP                 # noqa: E402
from wastebins_core.geo import dist_matrix            # noqa: E402
from wastebins_core import roadnet as RN              # noqa: E402
from experiments import exp_fleet as EF               # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

WHEN = datetime(2026, 3, 3, 7, 30, tzinfo=timezone.utc)
CONSTANT_DETOUR = 1.30


def travel_models(coords, net):
    """A street-graph model and a constant-factor model over the same stops."""
    road, freeflow, _snap = net.matrices(coords)
    approx = dist_matrix(coords, detour_factor=CONSTANT_DETOUR)
    provider = TR.SyntheticTrafficProvider(seed=EF.SEED)
    road_model = VRP.TravelModel(
        road, travel_context=TR.build_travel_context(
            coords, road, provider, WHEN, TR.FREEFLOW_KMH,
            freeflow_matrix=freeflow),
        default_speed_kmh=20.0, freeflow_kmh=TR.FREEFLOW_KMH)
    approx_model = VRP.TravelModel(
        approx, travel_context=TR.build_travel_context(
            coords, approx, provider, WHEN, TR.FREEFLOW_KMH),
        default_speed_kmh=20.0, freeflow_kmh=TR.FREEFLOW_KMH)
    return road_model, approx_model


def measure(plan: VRP.FleetPlan, travel: VRP.TravelModel) -> float:
    """
    Kilometres the plan's sequences cover under ``travel``.

    The routes are re-simulated rather than re-summed, because arrival times and
    loads change with the distances and a route that was feasible under one
    model need not be under the other. A route that becomes infeasible is
    reported rather than skipped: it is the sharpest form of the error.
    """
    total = 0.0
    infeasible = 0
    for route in plan.routes:
        order = [stop.task for stop in route.stops]
        if not order:
            continue
        evaluated = VRP.evaluate_route(order, route.vehicle, travel)
        if evaluated is None:
            infeasible += 1
            # Charge the sequence's raw length, which is a lower bound on what
            # a driver would cover before discovering the plan does not fit.
            total += sum(travel.distance(a.index, b.index)
                         for a, b in zip(order[:-1], order[1:])) / 1000.0
            continue
        total += evaluated.distance_m / 1000.0
    return total, infeasible


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=int, default=25)
    parser.add_argument("--budget", type=float, default=3.0)
    args = parser.parse_args()

    EF.set_study("dhaka")
    net = RN.load("dhaka")
    rng = np.random.default_rng(EF.SEED)
    weights = VRP.ObjectiveWeights()

    believed: List[float] = []
    actual: List[float] = []
    road_planner: List[float] = []
    infeasible_total = 0

    print(f"{'snap':>5}{'believed km':>13}{'actually km':>13}{'surprise':>11}"
          f"{'road planner':>14}{'cost of error':>15}")
    print("-" * 71)

    for i in range(args.snapshots):
        snapshot = EF.make_snapshot(rng, i)
        road_model, approx_model = travel_models(snapshot["coords"], net)
        tasks = EF.build_tasks(snapshot)
        fleet = EF.build_fleet(snapshot)

        plan_approx = VRP.solve(tasks, fleet, approx_model, weights,
                                time_budget_s=args.budget)
        plan_road = VRP.solve(tasks, fleet, road_model, weights,
                              time_budget_s=args.budget)

        told, _ = measure(plan_approx, approx_model)
        driven, bad = measure(plan_approx, road_model)
        proper, _ = measure(plan_road, road_model)

        believed.append(told)
        actual.append(driven)
        road_planner.append(proper)
        infeasible_total += bad

        print(f"{i:>5}{told:>13.1f}{driven:>13.1f}"
              f"{100 * (driven - told) / told:>10.1f}%"
              f"{proper:>14.1f}{100 * (driven - proper) / proper:>14.1f}%",
              flush=True)

    believed_a = np.array(believed)
    actual_a = np.array(actual)
    road_a = np.array(road_planner)

    surprise = ST.compare_paired(list(believed_a), list(actual_a),
                                 label="believed vs actually driven",
                                 lower_is_better=True)
    penalty = ST.compare_paired(list(road_a), list(actual_a),
                                label="street-graph planner vs approximate planner",
                                lower_is_better=True)

    payload = {
        "protocol": {
            "n_snapshots": args.snapshots,
            "n_bins": EF.STUDY["n_bins"],
            "n_vehicles": EF.STUDY["n_vehicles"],
            "constant_detour_factor": CONSTANT_DETOUR,
            "search_budget_s": args.budget,
            "note": "the same containers, fills, fleet and seeds throughout; "
                    "only the distances the planner optimises on differ",
        },
        "believed_km": {"mean": float(believed_a.mean())},
        "actually_driven_km": {"mean": float(actual_a.mean())},
        "road_planner_km": {"mean": float(road_a.mean())},
        "reporting_error_pct": float(
            100 * np.mean((actual_a - believed_a) / believed_a)),
        "optimisation_error_pct": float(
            100 * np.mean((actual_a - road_a) / road_a)),
        "infeasible_routes_under_road_model": infeasible_total,
        "paired_reporting": surprise,
        "paired_optimisation": penalty,
    }
    (RESULTS / "distance_model.json").write_text(json.dumps(payload, indent=2))

    print()
    print(f"  the approximate planner reports   {believed_a.mean():.1f} km")
    print(f"  it would actually drive           {actual_a.mean():.1f} km"
          f"   ({payload['reporting_error_pct']:+.1f} percent, reporting error)")
    print(f"  a street-graph planner drives     {road_a.mean():.1f} km"
          f"   ({payload['optimisation_error_pct']:+.1f} percent worse for the "
          f"approximate planner, optimisation error)")
    print(f"  routes infeasible once measured on the street graph: "
          f"{infeasible_total}")
    print(f"\nSaved {RESULTS / 'distance_model.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
