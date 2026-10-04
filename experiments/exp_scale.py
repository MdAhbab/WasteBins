"""
How the planners behave as the network grows.
=============================================

A dispatcher that only works at thirty containers is of no use to a city, so the
cost of the method has to be stated as a function of instance size rather than at
one operating point.  This script holds everything fixed except the number of
containers and the fleet sized to match it.

For each size it records, per snapshot:

* the construction time and the improvement time of the insertion planner,
  separately, because the first grows with the instance and the second is a
  fixed allowance;
* the objective, distance and share of containers served by that planner;
* the same for OR-Tools given the insertion planner's total time.

The fleet grows with the network at one vehicle per 25 containers, so the
workload per vehicle stays roughly constant and the curve measures the planner
rather than a change in how constrained the instance is.

The growth exponent is fitted by `analyze.py` to the construction time over all
sizes.  The earlier version fitted the whole solve time above 40 containers,
because below that the solve was bounded by its budget; construction is not
bounded by anything, so every size is a valid point.

Run:  python -m experiments.exp_scale --shard 0 --of 4
Out:  results/raw/scale.sNN.jsonl
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import vrp as VRP                 # noqa: E402
from experiments import exp_fleet as EF               # noqa: E402
from experiments import instances as IN               # noqa: E402
from experiments import policies as PO                # noqa: E402
from experiments import store as ST                   # noqa: E402
from experiments.exp_compare import IMPROVE_S         # noqa: E402

STUDY = "scale"
CONTAINERS_PER_VEHICLE = 25
SIZES = (40, 60, 80, 120, 160, 240, 320)
SNAPSHOTS = 3


def instance(n_bins: int, index: int):
    n_vehicles = max(2, round(n_bins / CONTAINERS_PER_VEHICLE))
    EF.set_study("dhaka", n_bins=n_bins, n_vehicles=n_vehicles)
    rng = np.random.default_rng(EF.SEED)
    snapshot = None
    for i in range(index + 1):
        snapshot = EF.make_snapshot(rng, i)
    t0 = time.perf_counter()
    travel = EF.travel_for(snapshot, IN.WHEN)
    matrix_s = time.perf_counter() - t0
    return snapshot, travel, EF.build_fleet(snapshot), matrix_s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    parser.add_argument("--snapshots", type=int, default=SNAPSHOTS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    args = parser.parse_args()

    ST.keep_awake()
    sizes = [int(s) for s in args.sizes.split(",")]
    # Largest first, so the shards finish at about the same time.
    units: List[Tuple[int, int]] = [(n, i) for n in sorted(sizes, reverse=True)
                                    for i in range(args.snapshots)]
    ST.RAW.mkdir(parents=True, exist_ok=True)
    ST.write_meta(STUDY, {
        "study": STUDY, "sizes": sizes, "snapshots": args.snapshots,
        "containers_per_vehicle": CONTAINERS_PER_VEHICLE,
        "improve_s": IMPROVE_S, "solver_config": PO.tuned_config(),
        "environment": ST.environment(),
    })

    done = ST.index(STUDY)
    weights = VRP.ObjectiveWeights()
    for n_bins, index in ST.mine(units, args.shard, args.of):
        keys = {p: f"{STUDY}|{n_bins}|{index}|{p}" for p in ("insertion", "ortools")}
        if all(k in done for k in keys.values()):
            continue
        snapshot, travel, fleet, matrix_s = instance(n_bins, index)
        canonical = PO.canonical_tasks(snapshot)
        if keys["insertion"] not in done:
            record = PO.solve_and_record("insertion", "full", snapshot, travel, fleet,
                                         weights, budget_s=None, improve_s=IMPROVE_S,
                                         canonical=canonical)
            record.update({"key": keys["insertion"], "study": STUDY,
                           "n_containers": n_bins, "n_vehicles": len(fleet),
                           "snapshot": index, "matrix_s": round(matrix_s, 3)})
            record.pop("served_ids", None)
            ST.append(STUDY, args.shard, record)
            done[keys["insertion"]] = record
        reference = done[keys["insertion"]]
        if keys["ortools"] not in done:
            record = PO.solve_and_record("ortools", "full", snapshot, travel, fleet,
                                         weights, budget_s=reference["elapsed_s"],
                                         improve_s=IMPROVE_S, canonical=canonical)
            record.update({"key": keys["ortools"], "study": STUDY,
                           "n_containers": n_bins, "n_vehicles": len(fleet),
                           "snapshot": index, "matrix_s": round(matrix_s, 3)})
            record.pop("served_ids", None)
            ST.append(STUDY, args.shard, record)
            done[keys["ortools"]] = record
        other = done[keys["ortools"]]
        print(f"  n={n_bins:>4} snapshot {index}: construction "
              f"{reference['search']['construct_s']:7.1f}s  insertion "
              f"{reference['objective']:9.1f}  ortools {other['objective']:9.1f}",
              flush=True)
    print(f"[{STUDY}] shard {args.shard} finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
