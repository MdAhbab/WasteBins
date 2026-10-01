"""
Dispatch rules, planners, and the one way a plan is scored.
===========================================================

The method under study is a *dispatch rule*: which containers are overdue, what
it costs to skip one, and which of them the planner may not skip.  A *planner*
then builds routes under that rule.  The two are separate here because the
evidence has to keep them separate.  A waiting-time guarantee is a property of
the rule.  How short the routes are is a property of the planner.

Rules
-----
``urgency``
    Priority alone.  No ageing, no overdue tier, no reservation.  This is
    sensor-driven dispatch as it is usually described, and the failure the paper
    starts from.
``ageing``
    Priority blended with a saturating ageing term.  The common remedy.
``tier_bounded``
    Overdue tier, skip penalty on the bounded ranking score.
``tier_unbounded``
    Overdue tier, skip penalty growing without limit.
``reserve_bounded``
    Overdue tier, bounded penalty, one reserved queue head.
``full``
    Overdue tier, unbounded penalty, one reserved queue head.  ``full_rK``
    reserves up to ``K``, and ``full_ageing`` adds the ageing blend.

Scoring
-------
A rule changes the prices a planner sees, so two plans made under two rules
report objectives in two currencies.  Every plan is therefore re-scored once,
under the prices of the full rule, by `score`.  That number is what tables
compare.  The plan's own objective is recorded beside it and is not compared
across rules.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import aging as AG            # noqa: E402
from wastebins_core import metaheuristics as MH   # noqa: E402
from wastebins_core import scenario as SC         # noqa: E402
from wastebins_core import vrp as VRP             # noqa: E402
from experiments import exp_fleet as EF           # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class Rule:
    name: str
    overdue_tier: bool = True
    escalating_price: bool = True
    #: Weight of the saturating ageing term blended into the ranking score of a
    #: container that is not yet overdue.  The full rule does not use it: on the
    #: pilot networks it changed no longest wait, and the rule without it served
    #: more containers with fewer overflows.  ``ageing`` keeps it because that
    #: blend, with no tier behind it, is the common remedy the rule is set
    #: against, and ``full_ageing`` measures what adding it to the full rule does.
    gamma: float = 0.0
    reserve: int = 1


RULES: Dict[str, Rule] = {r.name: r for r in (
    Rule("urgency", overdue_tier=False, escalating_price=False, reserve=0),
    Rule("ageing", overdue_tier=False, escalating_price=False,
         gamma=AG.DEFAULT_GAMMA, reserve=0),
    Rule("tier_bounded", escalating_price=False, reserve=0),
    Rule("tier_unbounded", reserve=0),
    Rule("reserve_bounded", escalating_price=False, reserve=1),
    Rule("full", reserve=1),
    Rule("full_r2", reserve=2),
    Rule("full_r4", reserve=4),
    Rule("full_r8", reserve=8),
    Rule("full_ageing", gamma=AG.DEFAULT_GAMMA, reserve=1),
)}

#: Planners whose search is limited by a wall-clock budget they are handed.
BUDGETED = ("ortools", "aco", "genetic")
#: Planners built on the insertion construction, limited by an improvement time.
INSERTION = ("insertion", "insertion_cheapest")

#: Solver settings.  The defaults are the settings of the first version of the
#: study.  `exp_tune` replaces them with the configuration that did best on the
#: tuning set and writes that choice to ``results/tuning.json``, which is read
#: here, so the configuration an experiment used is a file and not a memory.
DEFAULT_CONFIG: Dict[str, Dict] = {
    "ortools": {"first_solution": "PATH_CHEAPEST_ARC",
                "metaheuristic": "GUIDED_LOCAL_SEARCH",
                "emission_model": "mean_load"},
    "aco": {"n_ants": 16, "alpha": 1.0, "beta": 2.5, "rho": 0.10, "q0": 0.25,
            "polish_fraction": 0.0},
    "genetic": {"population_size": 40, "crossover_rate": 0.85,
                "mutation_rate": 0.20, "elite": 4, "tournament": 3,
                "polish_fraction": 0.0},
}


def tuned_config() -> Dict[str, Dict]:
    config = {k: dict(v) for k, v in DEFAULT_CONFIG.items()}
    path = RESULTS / "tuning.json"
    if path.exists():
        selected = json.loads(path.read_text()).get("selected", {})
        for name, values in selected.items():
            config.setdefault(name, {}).update(values)
    return config


def build_tasks(snapshot: Dict, rule: Rule,
                tau_h: float = AG.DEFAULT_TAU_H) -> List[VRP.BinTask]:
    return EF.build_tasks(snapshot, gamma=rule.gamma, tau_h=tau_h,
                          overdue_tier=rule.overdue_tier,
                          escalating_price=rule.escalating_price)


def plan_with(planner: str, rule: Rule, snapshot: Dict, travel, fleet,
              weights: VRP.ObjectiveWeights, budget_s: float,
              improve_s: float, seed: int = EF.SEED,
              config: Optional[Dict] = None,
              tau_h: float = AG.DEFAULT_TAU_H) -> Optional[VRP.FleetPlan]:
    """
    One plan, by one planner, under one rule.

    ``improve_s`` is the improvement time of the insertion planner, counted
    after its construction.  ``budget_s`` is the whole wall-clock allowance of a
    budgeted planner.  A comparison sets the second to what the first took.
    """
    tasks = build_tasks(snapshot, rule, tau_h)
    if planner in INSERTION:
        return VRP.solve(tasks, fleet, travel, weights,
                         improve_budget_s=improve_s,
                         reserve_overdue=rule.reserve,
                         construction=("cheapest" if planner == "insertion_cheapest"
                                       else "regret2"))
    if planner == "insertion_construct":
        return VRP.solve(tasks, fleet, travel, weights, improve=False,
                         reserve_overdue=rule.reserve)
    if planner in BUDGETED:
        chosen = dict((config or tuned_config()).get(planner, {}))
        return MH.solve_with(planner, tasks, fleet, travel, weights, seed=seed,
                             time_budget_s=budget_s, config=chosen,
                             reserve=rule.reserve)
    if planner == "risk_graph":
        return MH.solve_with("risk_graph", tasks, fleet, travel, weights,
                             reserve=rule.reserve)
    if planner == "static_sweep":
        # Traditional fixed schedule: a round of every bin in angular order
        # about the depot, using no priority information at all.  A rollout
        # passes where the previous shift's round stopped.
        order = SC.sweep_order(snapshot["coords"][1:], snapshot["coords"][0])
        by_index = {t.index: t for t in tasks}
        swept = [by_index[i + 1] for i in order if (i + 1) in by_index]
        return SC.static_sweep_plan(swept, fleet, travel, weights,
                                    start_node=snapshot.get("round_start"))
    if planner == "threshold":
        return SC.threshold_plan(tasks, fleet, travel, fill_threshold=0.70,
                                 fills=snapshot["fills"], weights=weights,
                                 improve_budget_s=improve_s)
    raise ValueError(f"unknown planner {planner!r}")


def canonical_tasks(snapshot: Dict,
                    tau_h: float = AG.DEFAULT_TAU_H) -> Dict[int, VRP.BinTask]:
    """The containers priced under the full rule, by id."""
    return {t.node_id: t for t in build_tasks(snapshot, RULES["full"], tau_h)}


def score(plan: VRP.FleetPlan, canonical: Dict[int, VRP.BinTask], travel,
          weights: VRP.ObjectiveWeights) -> Dict:
    """
    The plan's cost under the full rule's prices, with the terms it is made of.

    The sequences are simulated again with the canonical containers.  A rule
    changes prices and never loads, windows or service times, so a sequence that
    was feasible stays feasible, and the count of sequences that did not is
    reported so that this is checked rather than assumed.
    """
    routes, infeasible = [], 0
    for route in plan.routes:
        order = [canonical[s.task.node_id] for s in route.stops]
        evaluated = VRP.evaluate_route(order, route.vehicle, travel)
        if evaluated is None:
            infeasible += 1
            continue
        routes.append(evaluated)
    return score_routes(routes, canonical, weights, infeasible)


def score_routes(routes: List[VRP.VehicleRoute], canonical: Dict[int, VRP.BinTask],
                 weights: VRP.ObjectiveWeights, infeasible: int = 0) -> Dict:
    """
    Cost of routes already simulated on the canonical containers.

    `score` simulates a plan and calls this.  The weather study calls it
    directly, with the routes as they were driven in the actual conditions.
    """
    served_set = {s.task.node_id for r in routes for s in r.stops}
    unserved = [t for nid, t in canonical.items() if nid not in served_set]

    km = sum(r.distance_m for r in routes) / 1000.0
    co2 = sum(r.co2_kg for r in routes)
    hours = sum(r.duration_min for r in routes) / 60.0
    late = sum(1 for r in routes for s in r.stops
               if math.isfinite(s.task.time_to_overflow_h)
               and s.start_service_min / 60.0 > s.task.time_to_overflow_h)
    skip = VRP.unserved_cost(unserved, weights)
    terms = {
        "distance": weights.distance_km * km,
        "co2": weights.co2_kg * co2,
        "time": weights.hours * hours,
        "late": weights.missed_overflow_penalty * late,
        "skip": skip,
    }
    metrics = VRP.summarise(routes, unserved, list(canonical.values()), weights)
    overdue_left = [t for t in unserved if t.overdue]
    return {
        "objective": float(sum(terms.values())),
        "terms": {k: round(float(v), 4) for k, v in terms.items()},
        "metrics": metrics,
        "served_ids": sorted(served_set),
        # The two ways a container overflows are kept apart.  Reached after it
        # overflowed, and left out although it overflows within the horizon.
        "late_served": int(late),
        "overflow_unserved": int(sum(1 for t in unserved
                                     if VRP._overflows_within_horizon(t, weights))),
        "unserved": len(unserved),
        "unserved_by_tier": {str(k): int(sum(1 for t in unserved if t.tier == k))
                             for k in (AG.TIER_HAZARD, AG.TIER_OVERDUE, AG.TIER_NORMAL)},
        "overdue_total": int(sum(1 for t in canonical.values() if t.overdue)),
        "overdue_unserved": len(overdue_left),
        "longest_wait_unserved_h": float(max([t.wait_hours for t in unserved] or [0.0])),
        "longest_wait_served_h": float(max(
            [canonical[n].wait_hours for n in served_set] or [0.0])),
        "infeasible_on_rescoring": int(infeasible),
    }


#: Fields of `plan.metrics` that describe the search, kept with each record.
SEARCH_FIELDS = ("generations", "population", "iterations", "ants",
                 "ortools_status", "ortools_solved", "decode_dropped",
                 "ortools_objective", "first_solution", "metaheuristic",
                 "emission_model", "overdue_total", "heads_requested",
                 "heads_reserved", "heads_served", "head_repairs",
                 "unservable_overdue", "head_ids", "construct_s", "improve_s",
                 "descents", "ruin_rounds", "ruin_accepted", "construction",
                 "trips", "vehicles_used")


def solve_and_record(planner: str, rule_name: str, snapshot: Dict, travel, fleet,
                     weights: VRP.ObjectiveWeights, budget_s: float,
                     improve_s: float, config: Optional[Dict] = None,
                     flat_emissions: bool = False,
                     tau_h: float = AG.DEFAULT_TAU_H,
                     canonical: Optional[Dict[int, VRP.BinTask]] = None) -> Dict:
    """Run one planner under one rule and return the record that is stored."""
    rule = RULES[rule_name]
    wall, cpu = time.perf_counter(), time.process_time()
    plan = plan_with(planner, rule, snapshot, travel, fleet, weights,
                     budget_s=budget_s, improve_s=improve_s, config=config,
                     tau_h=tau_h)
    elapsed, cpu_used = time.perf_counter() - wall, time.process_time() - cpu
    if plan is None:
        return {"planner": planner, "rule": rule_name, "available": False}

    canonical = canonical or canonical_tasks(snapshot, tau_h)
    scored = score(plan, canonical, travel, weights)
    record = {
        "planner": planner,
        "rule": rule_name,
        "available": True,
        "budget_s": None if budget_s is None else round(float(budget_s), 3),
        "improve_s_allowed": round(float(improve_s), 3),
        "elapsed_s": round(elapsed, 3),
        "cpu_s": round(cpu_used, 3),
        "objective": round(scored["objective"], 4),
        "objective_own_prices": round(float(plan.objective), 4),
        "violations": EF.audit_plan(plan, f"{planner}/{rule_name}"),
        "search": {k: plan.metrics[k] for k in SEARCH_FIELDS if k in plan.metrics},
    }
    record.update({k: v for k, v in scored.items() if k != "objective"})
    if flat_emissions:
        nominal = MH.EM_NOMINAL_CO2_PER_KM(fleet[0].profile)
        flat = score(plan, canonical, MH.FlatEmissionTravel(travel, nominal), weights)
        record["flat_emission"] = {"objective": round(flat["objective"], 4),
                                   "co2_term": flat["terms"]["co2"],
                                   "co2_per_km": round(nominal, 5)}
    return record
