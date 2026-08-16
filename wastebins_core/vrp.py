"""
Fleet routing: prize-collecting CVRP with time windows and shift limits.
========================================================================

The first submission routed a *single* vehicle with a distance budget standing
in for every real constraint.  Reviewers objected on both counts, and rightly:
a distance budget cannot express a truck that fills up halfway through its
round, a depot that must be returned to for tipping, a crew whose shift ends, or
a bin that may only be serviced outside school hours.

This module models the real problem:

* **Capacity** -- each vehicle carries at most ``capacity_kg``; the load a bin
  contributes is its fill fraction times its physical mass when full, divided by
  the on-board compaction ratio.
* **Depot return** -- every route starts and ends at its vehicle's depot.  A
  vehicle may perform several *trips* within one shift, returning to tip when
  full, which is what real rounds do.
* **Shift limit** -- total elapsed time (travel + service + tipping) may not
  exceed ``shift_minutes``.
* **Time windows** -- each bin has ``[window_start, window_end]``; arriving early
  incurs waiting, arriving late is infeasible.
* **Prize collecting** -- a bin may be *skipped*.  Every bin carries a prize
  equal to its effective priority, and the objective trades collected prize
  against travel cost, so the fleet serves the urgent subset rather than
  sweeping the network.
* **Stream compatibility** -- a hazardous-waste bin is only served by a vehicle
  licensed for that stream.
* **Heterogeneous fleet** -- vehicles differ in capacity, speed, shift and
  emissions profile.

Algorithm
---------
Construction uses a **regret-2 insertion** heuristic driven by prize density,
which handles time windows far better than nearest-neighbour.  Improvement uses
a deterministic local search over four neighbourhoods -- intra-route 2-opt and
Or-opt, inter-route relocate and swap -- each move accepted only if it is
feasible *and* improves the objective.  Everything is deterministic given the
inputs, so results are reproducible without a seed.

Objective
---------
    minimise   travel_cost + unserved_penalty
    where      travel_cost      = w_dist * km + w_co2 * kg_CO2 + w_time * hours
               unserved_penalty = lambda_prize * sum(prize of skipped bins)

``lambda_prize`` converts urgency into the same currency as distance, and the
sensitivity of every reported result to it is swept in the experiments.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import emissions as EM


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
@dataclass
class BinTask:
    """One serviceable bin."""

    node_id: int
    index: int                     # row/column in the distance matrix
    prize: float = 0.0             # effective priority in [0, 1]
    load_kg: float = 0.0           # mass to be collected
    service_minutes: float = 4.0
    window_start_min: float = 0.0  # minutes after shift start
    window_end_min: float = 1440.0
    stream: str = "general"
    hazard: bool = False
    tier: int = 1                  # 0 = hazard override (must be served first)
    time_to_overflow_h: float = math.inf


@dataclass
class VehicleSpec:
    """One truck and the constraints bounding its shift."""

    vehicle_id: int
    name: str = "truck"
    depot_index: int = 0
    capacity_kg: float = 6000.0
    shift_minutes: float = 480.0
    shift_start_minute: int = 360
    avg_speed_kmh: float = 20.0
    compaction_ratio: float = 2.5
    tipping_minutes: float = 15.0   # time to discharge at the depot
    accepts_streams: Tuple[str, ...] = ()
    profile: EM.VehicleProfile = field(default_factory=EM.VehicleProfile)

    def accepts(self, stream: str) -> bool:
        return (not self.accepts_streams) or (stream in self.accepts_streams)


@dataclass
class ObjectiveWeights:
    distance_km: float = 1.0
    co2_kg: float = 2.0
    hours: float = 12.0
    lambda_prize: float = 45.0      # cost charged per unit of unserved prize
    hazard_multiplier: float = 6.0  # extra penalty for skipping a hazard bin
    missed_overflow_penalty: float = 80.0


@dataclass
class Stop:
    task: BinTask
    arrival_min: float
    start_service_min: float
    departure_min: float
    wait_min: float
    leg_distance_m: float
    leg_co2_kg: float
    load_after_kg: float
    trip_index: int


@dataclass
class VehicleRoute:
    vehicle: VehicleSpec
    stops: List[Stop] = field(default_factory=list)
    distance_m: float = 0.0
    co2_kg: float = 0.0
    duration_min: float = 0.0
    load_kg: float = 0.0
    trips: int = 1

    @property
    def node_ids(self) -> List[int]:
        return [s.task.node_id for s in self.stops]

    @property
    def indices(self) -> List[int]:
        return [s.task.index for s in self.stops]


@dataclass
class FleetPlan:
    routes: List[VehicleRoute]
    unserved: List[BinTask]
    objective: float
    metrics: Dict
    compute_ms: float = 0.0
    algorithm: str = "regret2_ls"

    def served_node_ids(self) -> List[int]:
        return [nid for r in self.routes for nid in r.node_ids]


# ---------------------------------------------------------------------------
# Travel-time abstraction
# ---------------------------------------------------------------------------
class TravelModel:
    """
    Converts a distance-matrix leg into duration and emissions.

    Wrapping this makes the planner agnostic to whether speeds come from a flat
    constant (the original manuscript), the synthetic congestion surface, or a
    live traffic feed -- all three satisfy the same interface.
    """

    def __init__(self, distance_m: np.ndarray,
                 travel_context=None,
                 default_speed_kmh: float = 20.0,
                 freeflow_kmh: float = 34.0):
        self.distance_m = np.asarray(distance_m, dtype=float)
        self.ctx = travel_context
        self.default_speed_kmh = float(default_speed_kmh)
        self.freeflow_kmh = float(freeflow_kmh)

    def distance(self, i: int, j: int) -> float:
        return float(self.distance_m[i][j])

    def speed_kmh(self, i: int, j: int) -> float:
        if self.ctx is not None:
            return float(self.ctx.speed_kmh(i, j))
        return self.default_speed_kmh

    def minutes(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        km = self.distance(i, j) / 1000.0
        return 60.0 * km / max(self.speed_kmh(i, j), 1e-6)

    def friction(self, i: int, j: int) -> float:
        if self.ctx is not None:
            return float(self.ctx.friction(i, j))
        return 0.0

    def leg_co2(self, i: int, j: int, payload_kg: float,
                profile: EM.VehicleProfile, idle_minutes: float = 0.0,
                lifts: int = 0, lifted_kg: float = 0.0) -> float:
        leg = EM.leg_emissions(
            distance_m=self.distance(i, j),
            speed_kmh=self.speed_kmh(i, j),
            payload_kg=payload_kg,
            friction=self.friction(i, j),
            idle_minutes=idle_minutes,
            lifts=lifts,
            lifted_kg=lifted_kg,
            profile=profile,
            freeflow_kmh=self.freeflow_kmh,
        )
        return leg.co2_kg(profile)


# ---------------------------------------------------------------------------
# Feasibility evaluation of one vehicle's ordered task list
# ---------------------------------------------------------------------------
def evaluate_route(order: Sequence[BinTask], vehicle: VehicleSpec,
                   travel: TravelModel,
                   allow_multi_trip: bool = True) -> Optional[VehicleRoute]:
    """
    Simulate a vehicle serving ``order`` and return the resulting route, or
    ``None`` when the sequence violates capacity, a time window or the shift.

    The simulation is exact: it tracks clock time (including waiting for a bin's
    window to open), on-board load, and inserts a depot tipping trip whenever the
    next bin would overflow the body.
    """
    depot = vehicle.depot_index
    route = VehicleRoute(vehicle=vehicle, trips=1)

    clock = 0.0            # minutes after shift start
    load = 0.0
    position = depot
    trip_index = 0

    for task in order:
        if not vehicle.accepts(task.stream):
            return None

        effective_load = task.load_kg / max(vehicle.compaction_ratio, 1e-6)
        if effective_load > vehicle.capacity_kg + 1e-9:
            return None                     # a single bin exceeds the body

        # --- tip at the depot when the next bin would overflow -----------
        if load + effective_load > vehicle.capacity_kg + 1e-9:
            if not allow_multi_trip:
                return None
            back_min = travel.minutes(position, depot)
            back_dist = travel.distance(position, depot)
            back_co2 = travel.leg_co2(position, depot, load, vehicle.profile)
            clock += back_min + vehicle.tipping_minutes
            route.distance_m += back_dist
            route.co2_kg += back_co2
            load = 0.0
            position = depot
            trip_index += 1
            route.trips = trip_index + 1
            if clock > vehicle.shift_minutes:
                return None

        # --- drive to the bin -------------------------------------------
        leg_min = travel.minutes(position, task.index)
        leg_dist = travel.distance(position, task.index)
        arrival = clock + leg_min

        if arrival > task.window_end_min + 1e-9:
            return None                     # too late for this bin

        wait = max(0.0, task.window_start_min - arrival)
        start_service = arrival + wait
        departure = start_service + task.service_minutes

        # Emissions charge the *current* payload for the leg plus the dwell.
        leg_co2 = travel.leg_co2(
            position, task.index, load, vehicle.profile,
            idle_minutes=task.service_minutes,
            lifts=1, lifted_kg=effective_load,
        )

        # --- shift feasibility must include getting home ------------------
        return_min = travel.minutes(task.index, depot)
        if departure + return_min > vehicle.shift_minutes + 1e-9:
            return None

        load += effective_load
        route.distance_m += leg_dist
        route.co2_kg += leg_co2
        route.stops.append(Stop(
            task=task,
            arrival_min=arrival,
            start_service_min=start_service,
            departure_min=departure,
            wait_min=wait,
            leg_distance_m=leg_dist,
            leg_co2_kg=leg_co2,
            load_after_kg=load,
            trip_index=trip_index,
        ))
        clock = departure
        position = task.index

    # --- final return to the depot --------------------------------------
    if route.stops:
        back_min = travel.minutes(position, depot)
        route.distance_m += travel.distance(position, depot)
        route.co2_kg += travel.leg_co2(position, depot, load, vehicle.profile)
        clock += back_min + vehicle.tipping_minutes
        if clock > vehicle.shift_minutes + 1e-9:
            return None

    route.duration_min = clock
    route.load_kg = sum(s.task.load_kg for s in route.stops)
    return route


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def route_cost(route: VehicleRoute, weights: ObjectiveWeights) -> float:
    """
    Cost of one vehicle's route.

    Besides travel, this charges for bins reached *after* they were predicted to
    overflow.  Without that term the planner would be optimising a different
    objective from the one the results report, and could trade a missed overflow
    for a shorter tour while still appearing to improve.
    """
    late = 0
    for stop in route.stops:
        tto = stop.task.time_to_overflow_h
        if math.isfinite(tto) and (stop.start_service_min / 60.0) > tto:
            late += 1
    return (weights.distance_km * (route.distance_m / 1000.0)
            + weights.co2_kg * route.co2_kg
            + weights.hours * (route.duration_min / 60.0)
            + weights.missed_overflow_penalty * late)


def unserved_cost(tasks: Sequence[BinTask], weights: ObjectiveWeights) -> float:
    """Prize forgone by skipping bins, plus the overflows that skipping causes."""
    total = 0.0
    for t in tasks:
        penalty = weights.lambda_prize * max(0.0, t.prize)
        if t.hazard or t.tier == 0:
            penalty *= weights.hazard_multiplier
        if math.isfinite(t.time_to_overflow_h):
            penalty += weights.missed_overflow_penalty
        total += penalty
    return total


def plan_objective(routes: Sequence[VehicleRoute], unserved: Sequence[BinTask],
                   weights: ObjectiveWeights) -> float:
    return sum(route_cost(r, weights) for r in routes) + unserved_cost(unserved, weights)


# ---------------------------------------------------------------------------
# Construction: regret-2 insertion
# ---------------------------------------------------------------------------
def _best_insertion(task: BinTask, order: List[BinTask], vehicle: VehicleSpec,
                    travel: TravelModel, weights: ObjectiveWeights,
                    base_cost: float) -> Optional[Tuple[float, int]]:
    """Cheapest feasible position for ``task`` in ``order``; ``(delta, position)``."""
    best: Optional[Tuple[float, int]] = None
    for pos in range(len(order) + 1):
        trial = order[:pos] + [task] + order[pos:]
        evaluated = evaluate_route(trial, vehicle, travel)
        if evaluated is None:
            continue
        delta = route_cost(evaluated, weights) - base_cost
        if best is None or delta < best[0]:
            best = (delta, pos)
    return best


def construct_regret2(tasks: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
                      travel: TravelModel, weights: ObjectiveWeights
                      ) -> Tuple[Dict[int, List[BinTask]], List[BinTask]]:
    """
    Regret-2 insertion.

    At each step every unrouted bin is costed into every vehicle.  The bin
    chosen is the one with the largest *regret* -- the gap between its best and
    second-best insertion cost, scaled by its prize -- because that is the bin
    which will become most expensive if it is left for later.  Plain cheapest
    insertion is myopic in exactly the way that produces the detours the
    reviewers noticed in the original greedy tour.
    """
    orders: Dict[int, List[BinTask]] = {v.vehicle_id: [] for v in vehicles}
    by_id = {v.vehicle_id: v for v in vehicles}
    pending = list(tasks)
    unserved: List[BinTask] = []

    # Hazard-tier bins are inserted first so they never lose a tie-break.
    pending.sort(key=lambda t: (t.tier, -t.prize))

    base_costs = {v.vehicle_id: 0.0 for v in vehicles}

    while pending:
        best_choice = None       # (regret_key, task, vehicle_id, position, delta)

        for task in pending:
            options: List[Tuple[float, int, int]] = []   # (delta, vehicle_id, pos)
            for v in vehicles:
                if not v.accepts(task.stream):
                    continue
                found = _best_insertion(task, orders[v.vehicle_id], v, travel,
                                        weights, base_costs[v.vehicle_id])
                if found is not None:
                    options.append((found[0], v.vehicle_id, found[1]))
            if not options:
                continue
            options.sort(key=lambda o: o[0])
            best_delta, vid, pos = options[0]
            second = options[1][0] if len(options) > 1 else best_delta + weights.lambda_prize
            regret = second - best_delta

            # Skipping costs lambda * prize; only insert when that is worth paying.
            skip_cost = weights.lambda_prize * task.prize
            if task.hazard or task.tier == 0:
                skip_cost *= weights.hazard_multiplier
            if best_delta > skip_cost:
                continue

            # Prefer hazard tier, then high regret, then high prize.
            key = (task.tier, -(regret + skip_cost - best_delta))
            if best_choice is None or key < best_choice[0]:
                best_choice = (key, task, vid, pos, best_delta)

        if best_choice is None:
            unserved.extend(pending)
            break

        _, task, vid, pos, delta = best_choice
        orders[vid] = orders[vid][:pos] + [task] + orders[vid][pos:]
        base_costs[vid] += delta
        pending.remove(task)

    return orders, unserved


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------
def _route_or_none(order: List[BinTask], vehicle: VehicleSpec, travel: TravelModel):
    return evaluate_route(order, vehicle, travel) if order else VehicleRoute(vehicle=vehicle)


def local_search(orders: Dict[int, List[BinTask]], vehicles: Sequence[VehicleSpec],
                 travel: TravelModel, weights: ObjectiveWeights,
                 unserved: List[BinTask], max_rounds: int = 12,
                 time_budget_s: float = 6.0) -> Tuple[Dict[int, List[BinTask]], List[BinTask]]:
    """
    Deterministic descent over four neighbourhoods.

    ``intra_2opt``      reverse a segment inside one route.
    ``intra_oropt``     relocate a run of 1-3 bins inside one route.
    ``inter_relocate``  move one bin to another vehicle.
    ``inter_swap``      exchange two bins between vehicles.
    ``reinsert``        try to bring a currently-skipped bin into service.

    Only feasible, strictly improving moves are accepted, so the objective is
    monotone non-increasing and the procedure always terminates.
    """
    by_id = {v.vehicle_id: v for v in vehicles}
    deadline = time.perf_counter() + max(0.1, time_budget_s)

    def vehicle_cost(vid: int, order: List[BinTask]) -> Optional[float]:
        """Cost of one vehicle's order, or None when infeasible."""
        if not order:
            return 0.0
        r = evaluate_route(order, by_id[vid], travel)
        return None if r is None else route_cost(r, weights)

    # Running per-vehicle costs, so a move only re-evaluates the routes it
    # actually touches instead of the whole fleet.
    costs: Dict[int, float] = {}
    for vid, order in orders.items():
        c = vehicle_cost(vid, order)
        if c is None:
            return orders, unserved
        costs[vid] = c

    skip_cost = unserved_cost(unserved, weights)
    best_cost = sum(costs.values()) + skip_cost

    def try_move(changes: Dict[int, List[BinTask]],
                 new_unserved: Optional[List[BinTask]] = None) -> bool:
        """Apply ``changes`` if every touched route stays feasible and the
        objective strictly improves.  Mutates the enclosing state on success."""
        nonlocal best_cost, skip_cost, unserved
        new_costs: Dict[int, float] = {}
        for vid, order in changes.items():
            c = vehicle_cost(vid, order)
            if c is None:
                return False
            new_costs[vid] = c
        candidate_skip = (unserved_cost(new_unserved, weights)
                          if new_unserved is not None else skip_cost)
        delta = (sum(new_costs.values()) - sum(costs[vid] for vid in changes)
                 + candidate_skip - skip_cost)
        if delta >= -1e-9:
            return False
        orders.update(changes)
        costs.update(new_costs)
        skip_cost = candidate_skip
        if new_unserved is not None:
            unserved = new_unserved
        best_cost += delta
        return True

    for _ in range(max_rounds):
        improved = False

        # --- intra-route: 2-opt -----------------------------------------
        for vid in list(orders.keys()):
            n = len(orders[vid])
            for i in range(n - 1):
                for k in range(i + 2, n):
                    if time.perf_counter() > deadline:
                        return orders, unserved
                    order = orders[vid]
                    if k >= len(order):
                        break
                    trial = order[:i] + order[i:k + 1][::-1] + order[k + 1:]
                    if try_move({vid: trial}):
                        improved = True

        # --- intra-route: Or-opt ----------------------------------------
        for vid in list(orders.keys()):
            for seg in (1, 2, 3):
                i = 0
                while i <= len(orders[vid]) - seg:
                    order = orders[vid]
                    block = order[i:i + seg]
                    rest = order[:i] + order[i + seg:]
                    moved = False
                    for j in range(len(rest) + 1):
                        if j == i:
                            continue
                        if time.perf_counter() > deadline:
                            return orders, unserved
                        if try_move({vid: rest[:j] + block + rest[j:]}):
                            improved = moved = True
                            break
                    if not moved:
                        i += 1

        # --- inter-route: relocate --------------------------------------
        vids = list(orders.keys())
        for a in vids:
            for b in vids:
                if a == b:
                    continue
                i = 0
                while i < len(orders[a]):
                    task = orders[a][i]
                    if not by_id[b].accepts(task.stream):
                        i += 1
                        continue
                    src = orders[a][:i] + orders[a][i + 1:]
                    moved = False
                    for j in range(len(orders[b]) + 1):
                        if time.perf_counter() > deadline:
                            return orders, unserved
                        dst = orders[b][:j] + [task] + orders[b][j:]
                        if try_move({a: src, b: dst}):
                            improved = moved = True
                            break
                    if not moved:
                        i += 1

        # --- inter-route: swap ------------------------------------------
        for a in vids:
            for b in vids:
                if a >= b:
                    continue
                for i in range(len(orders[a])):
                    for j in range(len(orders[b])):
                        if time.perf_counter() > deadline:
                            return orders, unserved
                        if i >= len(orders[a]) or j >= len(orders[b]):
                            break
                        ta, tb = orders[a][i], orders[b][j]
                        if not (by_id[b].accepts(ta.stream) and by_id[a].accepts(tb.stream)):
                            continue
                        sa = orders[a][:i] + [tb] + orders[a][i + 1:]
                        sb = orders[b][:j] + [ta] + orders[b][j + 1:]
                        if try_move({a: sa, b: sb}):
                            improved = True

        # --- bring a skipped bin back into service ----------------------
        for task in list(unserved):
            placed = False
            for vid in vids:
                if not by_id[vid].accepts(task.stream):
                    continue
                remaining = [t for t in unserved if t is not task]
                for j in range(len(orders[vid]) + 1):
                    if time.perf_counter() > deadline:
                        return orders, unserved
                    trial = orders[vid][:j] + [task] + orders[vid][j:]
                    if try_move({vid: trial}, new_unserved=remaining):
                        improved = placed = True
                        break
                if placed:
                    break

        if not improved:
            break

    return orders, unserved


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------
def solve(tasks: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
          travel: TravelModel, weights: Optional[ObjectiveWeights] = None,
          improve: bool = True, time_budget_s: float = 6.0,
          algorithm: str = "regret2_ls") -> FleetPlan:
    """Construct, then improve, then report a fully evaluated fleet plan."""
    weights = weights or ObjectiveWeights()
    started = time.perf_counter()

    orders, unserved = construct_regret2(tasks, vehicles, travel, weights)
    if improve:
        orders, unserved = local_search(orders, vehicles, travel, weights,
                                        unserved, time_budget_s=time_budget_s)

    by_id = {v.vehicle_id: v for v in vehicles}
    routes: List[VehicleRoute] = []
    for vid, order in orders.items():
        if not order:
            continue
        evaluated = evaluate_route(order, by_id[vid], travel)
        if evaluated is None:
            # Defensive: never emit an infeasible plan; drop back to the prefix
            # that is feasible and mark the remainder unserved.
            feasible: List[BinTask] = []
            for task in order:
                trial = feasible + [task]
                if evaluate_route(trial, by_id[vid], travel) is not None:
                    feasible = trial
                else:
                    unserved.append(task)
            evaluated = evaluate_route(feasible, by_id[vid], travel) if feasible else None
            if evaluated is None:
                unserved.extend(t for t in order if t not in unserved)
                continue
        routes.append(evaluated)

    objective = plan_objective(routes, unserved, weights)
    plan = FleetPlan(
        routes=routes,
        unserved=unserved,
        objective=objective,
        metrics=summarise(routes, unserved, tasks),
        compute_ms=(time.perf_counter() - started) * 1000.0,
        algorithm=algorithm,
    )
    return plan


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarise(routes: Sequence[VehicleRoute], unserved: Sequence[BinTask],
              all_tasks: Sequence[BinTask]) -> Dict:
    """Service, sustainability and equity metrics for a completed plan."""
    total_km = sum(r.distance_m for r in routes) / 1000.0
    total_co2 = sum(r.co2_kg for r in routes)
    total_min = sum(r.duration_min for r in routes)
    served = [s for r in routes for s in r.stops]

    hazard_tasks = [t for t in all_tasks if t.hazard or t.tier == 0]
    hazard_served = [s for s in served if s.task.hazard or s.task.tier == 0]
    hazard_times = [s.start_service_min / 60.0 for s in hazard_served]

    missed = 0
    for s in served:
        if math.isfinite(s.task.time_to_overflow_h) and \
                (s.start_service_min / 60.0) > s.task.time_to_overflow_h:
            missed += 1
    # A skipped bin that would overflow inside the horizon is also a miss.
    for t in unserved:
        if math.isfinite(t.time_to_overflow_h):
            missed += 1

    total_prize = sum(max(0.0, t.prize) for t in all_tasks)
    collected = sum(max(0.0, s.task.prize) for s in served)

    return {
        "vehicles_used": len(routes),
        "bins_served": len(served),
        "bins_unserved": len(unserved),
        "distance_km": round(total_km, 3),
        "co2_kg": round(total_co2, 3),
        "co2_kg_per_km": round(total_co2 / total_km, 4) if total_km > 1e-9 else 0.0,
        "duration_min": round(total_min, 2),
        "load_kg": round(sum(r.load_kg for r in routes), 1),
        "trips": sum(r.trips for r in routes),
        "hazard_total": len(hazard_tasks),
        "hazard_served": len(hazard_served),
        "hazard_coverage_pct": round(100.0 * len(hazard_served) / len(hazard_tasks), 2)
        if hazard_tasks else 100.0,
        "mean_hazard_response_h": round(float(np.mean(hazard_times)), 3) if hazard_times else 0.0,
        "worst_hazard_response_h": round(float(np.max(hazard_times)), 3) if hazard_times else 0.0,
        "missed_overflow": missed,
        "prize_collected_pct": round(100.0 * collected / total_prize, 2)
        if total_prize > 1e-9 else 100.0,
        "capacity_utilisation_pct": round(
            100.0 * sum(r.load_kg for r in routes)
            / max(1e-9, sum(r.vehicle.capacity_kg * r.trips for r in routes)), 2),
    }


def plan_to_dict(plan: FleetPlan) -> Dict:
    """JSON-serialisable representation for the API and the audit ledger."""
    return {
        "algorithm": plan.algorithm,
        "objective": round(plan.objective, 4),
        "compute_ms": round(plan.compute_ms, 2),
        "metrics": plan.metrics,
        "routes": [
            {
                "vehicle_id": r.vehicle.vehicle_id,
                "vehicle_name": r.vehicle.name,
                "trips": r.trips,
                "distance_km": round(r.distance_m / 1000.0, 3),
                "co2_kg": round(r.co2_kg, 3),
                "duration_min": round(r.duration_min, 2),
                "load_kg": round(r.load_kg, 1),
                "capacity_kg": r.vehicle.capacity_kg,
                "stops": [
                    {
                        "node_id": s.task.node_id,
                        "sequence": i,
                        "trip_index": s.trip_index,
                        "arrival_min": round(s.arrival_min, 2),
                        "start_service_min": round(s.start_service_min, 2),
                        "departure_min": round(s.departure_min, 2),
                        "wait_min": round(s.wait_min, 2),
                        "leg_distance_m": round(s.leg_distance_m, 1),
                        "leg_co2_kg": round(s.leg_co2_kg, 4),
                        "load_after_kg": round(s.load_after_kg, 1),
                        "prize": round(s.task.prize, 4),
                        "hazard": bool(s.task.hazard),
                    }
                    for i, s in enumerate(r.stops)
                ],
            }
            for r in plan.routes
        ],
        "unserved": [
            {"node_id": t.node_id, "prize": round(t.prize, 4), "hazard": bool(t.hazard)}
            for t in plan.unserved
        ],
    }
