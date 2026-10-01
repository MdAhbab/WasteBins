"""
What a planner loses by optimising on approximate distances.
============================================================

Running the routing study twice, once on the street graph and once on
great-circle distance times a constant, compares two different plans each
measured in its own metric. That answers a narrow question: what an operator is
told. It does not answer the question an operator cares about, which is what the
vehicle actually drives.

This script answers that one. Each snapshot is planned on constant-factor
distances and the plan is then measured three ways.

``reported``
    The length of the constant-factor plan in constant-factor distances. This is
    the number that planner reports.

``driven``
    The same plan, the same containers in the same order, measured on the street
    graph. This is what the vehicle drives if it follows that plan. The gap
    against ``reported`` is the reporting error.

``street``
    The length of a plan built on the street graph by the same planner at the
    same matched time, read from the main comparison. The gap between ``driven``
    and ``street`` is the optimisation error: what choosing the sequence on the
    wrong distances costs, as opposed to merely reporting it wrongly.

The driven length is the full length. A plan built on understated distances can
overrun the shift once it is driven, and the first version of this experiment
charged such a route only the legs between its containers, which left out the
legs to and from the depot and made the driven total a lower bound. Here the
sequence is simulated to the end with the shift limit recorded rather than
enforced, so every leg is counted and the overrun is reported in minutes.

Run:  python -m experiments.exp_distance_model --shard 0 --of 4
Out:  results/raw/distance.sNN.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import traffic as TR              # noqa: E402
from wastebins_core import vrp as VRP                 # noqa: E402
from wastebins_core.geo import dist_matrix            # noqa: E402
from experiments import exp_fleet as EF               # noqa: E402
from experiments import instances as IN               # noqa: E402
from experiments import policies as PO                # noqa: E402
from experiments import store as ST                   # noqa: E402
from experiments.exp_compare import IMPROVE_S         # noqa: E402

STUDY = "distance"
CONSTANT_DETOUR = 1.30
SNAPSHOTS = 25
PLANNERS = ("insertion", "ortools", "aco")


def constant_factor_travel(coords) -> VRP.TravelModel:
    """Great-circle distance times the constant, under the same congestion surface."""
    approx = dist_matrix(coords, detour_factor=CONSTANT_DETOUR)
    provider = TR.SyntheticTrafficProvider(seed=EF.SEED)
    return VRP.TravelModel(
        approx, travel_context=TR.build_travel_context(
            coords, approx, provider, IN.WHEN, TR.FREEFLOW_KMH),
        default_speed_kmh=20.0, freeflow_kmh=TR.FREEFLOW_KMH)


def measure(plan: VRP.FleetPlan, travel: VRP.TravelModel) -> Dict:
    """
    Length and timing of the plan's sequences under ``travel``, limits recorded.

    Loads are re-simulated, so a tipping trip is inserted wherever the body
    fills, exactly as a driver would have to make it.
    """
    km, over_routes, overrun, worst, late, routes = 0.0, 0, 0.0, 0.0, 0, 0
    for route in plan.routes:
        order = [stop.task for stop in route.stops]
        if not order:
            continue
        routes += 1
        driven = VRP.evaluate_route(order, route.vehicle, travel, enforce_time=False)
        km += driven.distance_m / 1000.0
        late += driven.late_arrivals
        if driven.overrun_min > 1e-6:
            over_routes += 1
            overrun += driven.overrun_min
            worst = max(worst, driven.overrun_min)
    return {"km": km, "routes": routes, "routes_over_shift": over_routes,
            "overrun_min_total": overrun, "overrun_min_worst": worst,
            "late_arrivals": late}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOTS)
    args = parser.parse_args()

    ST.keep_awake()
    ST.RAW.mkdir(parents=True, exist_ok=True)
    (ST.RAW / f"{STUDY}.meta.json").write_text(json.dumps({
        "study": STUDY, "constant_detour_factor": CONSTANT_DETOUR,
        "network": "S0", "snapshots": args.snapshots, "planners": PLANNERS,
        "improve_s": IMPROVE_S, "solver_config": PO.tuned_config(),
        "environment": ST.environment(),
    }, indent=2, default=str))

    done = ST.index(STUDY)
    weights = VRP.ObjectiveWeights()
    for index in ST.mine(range(args.snapshots), args.shard, args.of):
        if all(f"{STUDY}|S0|{index}|{p}" in done for p in PLANNERS):
            continue
        snapshot = IN.snapshots("S0", SNAPSHOTS, 6)[index]
        street = EF.travel_for(snapshot, IN.WHEN)
        approx = constant_factor_travel(snapshot["coords"])
        fleet = EF.build_fleet(snapshot)
        shift = float(fleet[0].shift_minutes)
        matched = None
        for planner in PLANNERS:
            key = f"{STUDY}|S0|{index}|{planner}"
            if key in done:
                if planner == "insertion":
                    matched = done[key]["elapsed_s"]
                continue
            wall = time.perf_counter()
            plan = PO.plan_with(planner, PO.RULES["full"], snapshot, approx, fleet,
                                weights, budget_s=matched, improve_s=IMPROVE_S)
            elapsed = time.perf_counter() - wall
            if planner == "insertion":
                matched = elapsed
            reported = measure(plan, approx)
            driven = measure(plan, street)
            record = {
                "key": key, "study": STUDY, "network": "S0", "snapshot": index,
                "planner": planner, "elapsed_s": round(elapsed, 3),
                "budget_s": None if planner == "insertion" else round(matched, 3),
                "served": int(plan.metrics["bins_served"]),
                "reported_km": round(reported["km"], 4),
                "driven_km": round(driven["km"], 4),
                "routes": driven["routes"],
                "routes_over_shift": driven["routes_over_shift"],
                "overrun_min_total": round(driven["overrun_min_total"], 3),
                "overrun_min_worst": round(driven["overrun_min_worst"], 3),
                "late_arrivals": driven["late_arrivals"],
                "shift_min": shift,
                "reported_routes_over_shift": reported["routes_over_shift"],
            }
            ST.append(STUDY, args.shard, record)
            done[key] = record
            gap = 100.0 * (record["driven_km"] - record["reported_km"]) / record["driven_km"]
            print(f"  S0 {index:>2} {planner:<10} reported {record['reported_km']:7.1f} km  "
                  f"driven {record['driven_km']:7.1f} km  ({gap:+.1f} percent)  "
                  f"over shift {record['routes_over_shift']} of {record['routes']}, "
                  f"worst {record['overrun_min_worst']:.0f} min", flush=True)
    print(f"[{STUDY}] shard {args.shard} finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
