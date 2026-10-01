"""
Single-cycle comparisons, at matched time and on named instances.
=================================================================

Each study here plans the same dispatch snapshots with several planners and
writes one raw record per solve to ``results/raw/``.  Nothing is summarised in
this file; `analyze.py` does that from the records.

Time is matched per instance, not set once for all.  The insertion planner runs
first: its construction runs to completion and its improvement runs for a fixed
allowance.  Whatever that took in total, on that instance and on that machine at
that moment, is then the whole budget of every budgeted planner on the same
instance.  A fixed number of seconds would not do this, because the construction
takes two minutes on one instance and one second on another, and takes longer
when other work shares the processor.

Studies
-------
``main``        Five Dhaka samples of 160 containers, every planner.
``wyndham``     The 32 observed Wyndham containers, every planner.
``ablation``    Ten snapshots of the primary sample, the insertion planner under
                each rule.
``weights-*``   Eight snapshots under alternative objective weights.
``depot-*``     Eight snapshots from other transfer stations, and ten Wyndham
                days from other depot positions.
``budget-*``    Six snapshots with the budgeted planners given a multiple of the
                matched time.

The supporting studies use fewer snapshots than the main comparison.  Each of
their units costs the same solves as a unit of the main comparison, and they
answer whether a conclusion moves, which a paired difference over six to ten
instances shows.

Run:  python -m experiments.exp_compare --study main --shard 0 --of 4
Out:  results/raw/<study>.sNN.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import vrp as VRP             # noqa: E402
from experiments import exp_fleet as EF           # noqa: E402
from experiments import instances as IN           # noqa: E402
from experiments import policies as PO            # noqa: E402
from experiments import store as ST               # noqa: E402

#: Improvement time of the insertion planner, after its construction, seconds.
IMPROVE_S = 30.0

REFERENCE = ("insertion", "full")
PLANNERS_FULL = [("ortools", "full"), ("aco", "full"), ("genetic", "full"),
                 ("risk_graph", "tier_unbounded"),
                 ("static_sweep", "full"), ("threshold", "full")]
ABLATION = [("insertion", "urgency"), ("insertion", "ageing"),
            ("insertion", "tier_bounded"), ("insertion", "tier_unbounded"),
            ("insertion", "reserve_bounded"), ("insertion", "full_ageing"),
            ("insertion_cheapest", "full"), ("insertion_construct", "full")]

#: Alternative objective weights.  Each changes one weight by a factor a
#: municipality could defend and leaves the others at their defaults.
WEIGHT_VARIANTS: Dict[str, Dict[str, float]] = {
    "co2x0": {"co2_kg": 0.0},
    "co2x3": {"co2_kg": 6.0},
    "timex0.5": {"hours": 6.0},
    "timex2": {"hours": 24.0},
    "overflowx0.5": {"missed_overflow_penalty": 40.0},
    "overflowx2": {"missed_overflow_penalty": 160.0},
    "lambdax0.5": {"lambda_prize": 22.5},
    "lambdax2": {"lambda_prize": 90.0},
}

#: Transfer stations used as the depot, by rank of centrality (0 is the main one).
DEPOT_RANKS = (4, 9, 18)


def study_config(name: str) -> Dict:
    base = dict(area="dhaka", n_vehicles=6, improve_s=IMPROVE_S,
                weights={}, depot=None, budget_factor=1.0, flat=False,
                reference_from=None)
    if name == "main":
        return dict(base, flat=True,
                    networks=[("S0", 25), ("S1", 10), ("S2", 10), ("S3", 10), ("S4", 10)],
                    specs=PLANNERS_FULL)
    if name == "wyndham":
        return dict(base, area="wyndham", n_vehicles=3, flat=True,
                    networks=[("W", 25)], specs=PLANNERS_FULL)
    if name == "ablation":
        return dict(base, networks=[("S0", 10)], specs=ABLATION,
                    reference_from="main")
    if name.startswith("weights-"):
        tag = name.split("-", 1)[1]
        return dict(base, networks=[("S0", 8)], weights=WEIGHT_VARIANTS[tag],
                    specs=[("ortools", "full"), ("aco", "full")])
    if name.startswith("depot-"):
        tag = name.split("-", 1)[1]
        if tag in IN.wyndham_depots():
            return dict(base, area="wyndham", n_vehicles=3, networks=[("W", 10)],
                        depot=IN.wyndham_depots()[tag],
                        specs=[("ortools", "full"), ("aco", "full")])
        return dict(base, networks=[("S0", 8)], depot=IN.depots()[int(tag)],
                    specs=[("ortools", "full"), ("aco", "full")])
    if name.startswith("budget-"):
        factor = float(name.split("-", 1)[1])
        return dict(base, networks=[("S0", 6)], budget_factor=factor,
                    reference_from="main",
                    specs=[("ortools", "full"), ("aco", "full"), ("genetic", "full")])
    raise KeyError(f"unknown study {name!r}")


def key_of(study: str, network: str, snapshot: int, planner: str, rule: str) -> str:
    return f"{study}|{network}|{snapshot}|{planner}|{rule}"


def load_instance(cfg: Dict, network: str, index: int):
    """Snapshot, travel model and fleet for one unit."""
    count = dict(cfg["networks"])[network]
    if cfg["area"] == "wyndham":
        snapshot = IN.wyndham_snapshots(count, cfg["n_vehicles"],
                                        depot=cfg["depot"])[index]
    else:
        snapshot = IN.snapshots(network, count, cfg["n_vehicles"],
                                depot=cfg["depot"])[index]
    travel = EF.travel_for(snapshot, IN.WHEN)
    fleet = EF.build_fleet(snapshot)
    return snapshot, travel, fleet


def run_unit(study: str, cfg: Dict, network: str, index: int, shard: int,
             done: Dict[str, Dict], reference_done: Dict[str, Dict]) -> int:
    """Run whatever this unit still lacks.  Returns the number of solves made."""
    ref_key = key_of(study, network, index, *REFERENCE)
    wanted = [(p, r) for p, r in cfg["specs"]
              if key_of(study, network, index, p, r) not in done]
    borrowed = None
    if cfg["reference_from"]:
        borrowed = reference_done.get(
            key_of(cfg["reference_from"], network, index, *REFERENCE))
    need_reference = ref_key not in done and borrowed is None
    if not wanted and not need_reference:
        return 0

    weights = VRP.ObjectiveWeights(**cfg["weights"])
    snapshot, travel, fleet = load_instance(cfg, network, index)
    canonical = PO.canonical_tasks(snapshot)
    solved = 0

    def stamp(record: Dict, planner: str, rule: str) -> Dict:
        record.update({
            "key": key_of(study, network, index, planner, rule),
            "study": study, "network": network, "snapshot": index,
            "n_containers": len(snapshot["node_ids"]),
            "n_vehicles": len(fleet),
            "weights": cfg["weights"],
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return record

    if need_reference:
        record = PO.solve_and_record(*REFERENCE, snapshot, travel, fleet, weights,
                                     budget_s=None, improve_s=cfg["improve_s"],
                                     flat_emissions=cfg["flat"], canonical=canonical)
        ST.append(study, shard, stamp(record, *REFERENCE))
        done[ref_key] = record
        solved += 1
    reference = done.get(ref_key) or borrowed
    matched = float(reference["elapsed_s"]) * float(cfg["budget_factor"])

    for planner, rule in wanted:
        record = PO.solve_and_record(planner, rule, snapshot, travel, fleet, weights,
                                     budget_s=matched, improve_s=cfg["improve_s"],
                                     flat_emissions=cfg["flat"], canonical=canonical)
        record["reference_elapsed_s"] = reference["elapsed_s"]
        ST.append(study, shard, stamp(record, planner, rule))
        done[record["key"]] = record
        solved += 1
        print(f"    {planner:<20}{rule:<16}obj {record.get('objective', float('nan')):>9.1f}"
              f"  served {record.get('metrics', {}).get('bins_served', 0):>3}"
              f"  {record.get('elapsed_s', 0):>6.1f}s", flush=True)
    return solved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--max-units", type=int, default=None,
                        help="stop after this many units, for a smoke run")
    parser.add_argument("--snapshots", type=int, default=None,
                        help="cap the snapshots per network, for a smoke run")
    args = parser.parse_args()

    ST.keep_awake()
    cfg = study_config(args.study)
    if args.snapshots is not None:
        cfg["networks"] = [(n, min(c, args.snapshots)) for n, c in cfg["networks"]]
    units: List[Tuple[str, int]] = [(network, i) for network, count in cfg["networks"]
                                    for i in range(count)]
    (ST.RAW).mkdir(parents=True, exist_ok=True)
    (ST.RAW / f"{args.study}.meta.json").write_text(json.dumps({
        "study": args.study,
        "config": {k: v for k, v in cfg.items()},
        "improve_s": cfg["improve_s"],
        "solver_config": PO.tuned_config(),
        "environment": ST.environment(),
    }, indent=2, default=str))

    done = ST.index(args.study)
    reference_done = ST.index(cfg["reference_from"]) if cfg["reference_from"] else {}
    mine = list(ST.mine(units, args.shard, args.of))
    print(f"[{args.study}] shard {args.shard}/{args.of}: {len(mine)} of {len(units)} units",
          flush=True)
    started = time.perf_counter()
    for count, (network, index) in enumerate(mine, start=1):
        if args.max_units is not None and count > args.max_units:
            break
        t0 = time.perf_counter()
        print(f"  {network} snapshot {index}", flush=True)
        solved = run_unit(args.study, cfg, network, index, args.shard, done,
                          reference_done)
        print(f"  {network} snapshot {index}: {solved} solves, "
              f"{time.perf_counter() - t0:.0f}s "
              f"(total {(time.perf_counter() - started) / 60:.1f} min)", flush=True)
    print(f"[{args.study}] shard {args.shard} finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
