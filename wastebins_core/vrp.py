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
a deterministic local search over six neighbourhoods -- intra-route 2-opt and
Or-opt, inter-route relocate and swap, and reinsert/eject to move bins in and
out of service -- each move accepted only if it is feasible *and* improves the
objective.  Everything is deterministic given the inputs, so results are
reproducible without a seed.

Objective
---------
    minimise   travel_cost + unserved_penalty
    where      travel_cost      = w_dist * km + w_co2 * kg_CO2 + w_time * hours
                                  + missed_overflow_penalty * (bins served late)
               unserved_penalty = sum over skipped bins of `skip_cost`

``lambda_prize`` converts urgency into the same currency as distance, and the
sensitivity of every reported result to it is swept in the experiments.

Every decision about whether a bin is worth serving -- in construction, in local
search and in the final score -- goes through :func:`skip_cost`.  Keeping a
second, subtly different copy of that rule inside the constructor is exactly the
kind of drift that makes a planner optimise something other than what is
reported: the copy omitted the overflow term, so construction systematically
abandoned bins that the objective said were worth 80 units to save.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import emissions as EM
from .aging import TIER_HAZARD, TIER_NORMAL, TIER_OVERDUE


def _finite(value: float, default: float = 0.0) -> float:
    """
    Coerce a non-finite input to a usable number.

    A NaN prize or load propagates silently through every comparison in this
    module -- ``nan > capacity`` is ``False``, so a NaN load passes the capacity
    check and produces a route that is infeasible in reality -- and then leaks
    into the JSON the API serves, where ``NaN`` is not even valid.  Bad inputs
    are neutralised at the boundary instead.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _exceeds(mass_kg: float, volume_m3: float, vehicle: "VehicleSpec",
             tol: float = 1e-9) -> bool:
    """
    Whether a load breaches either capacity limit of ``vehicle``.

    A refuse body is bounded twice, and the two limits are not interchangeable.
    Mass is bounded by the axle rating, and compaction does not reduce it.
    Volume is bounded by the body, and compaction is exactly what reduces it.

    Which one binds depends on density, and it is worth being exact rather than
    repeating the usual claim that waste is light and bulky.  The two limits meet
    at a density of ``capacity_kg / (body_volume_m3 * compaction_ratio)``.  For
    the default 6000 kg body of 16 cubic metres at a compaction ratio of 2.5 that
    crossover is 150 kg per cubic metre.  The densities configured here run from
    180 to 260, all above it, so **mass binds first in this deployment**: the
    rating is reached with about 10.9 of the 16 cubic metres filled.  Volume
    binds only for lighter material, such as uncompacted dry recyclables.

    Both are therefore checked, because a fleet sees both regimes and the wrong
    one silently permits infeasible routes.  The volume check is skipped when the
    instance carries no density information, so mass-only problems still work.
    """
    if _finite(mass_kg) > float(vehicle.capacity_kg) + tol:
        return True
    body = float(vehicle.body_volume_m3)
    if body > 0.0 and _finite(volume_m3) > body + tol:
        return True
    return False


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
    tier: int = TIER_NORMAL        # 0 hazard, 1 overdue, 2 normal
    # Pricing escalation for an overdue bin, `wait / (2 tau)`, from
    # `aging.overdue_pressure`.  `prize` orders bins inside a tier and is bounded
    # by construction; this prices them and is not bounded, which is the whole
    # point.  See `skip_cost` for why the two cannot be the same number.  Zero
    # means the caller supplied no wait, and pricing falls back to the prize.
    overdue_pressure: float = 0.0
    # The head of the overdue queue, which the planner may not decline.
    #
    # Every other bin is governed by a price, and a price is a preference: the
    # search weighs it against travel and drops the bin when the arithmetic
    # says so. That is correct for all of them and fatal for one. Measured on
    # the rollout, one bin was passed over for forty consecutive cycles, 480 h,
    # while the fleet cleared other overdue bins every cycle. No price fixes
    # that, because the bin loses the arithmetic every time it is evaluated.
    #
    # `skip_cost` charges a mandatory bin a cost no route can outweigh, which
    # turns serving it into a constraint the search optimises around rather than
    # a preference it can trade away. See `reserve_overdue_head` for the repair
    # that catches the case where it is dropped anyway.
    mandatory: bool = False
    time_to_overflow_h: float = math.inf
    # Loose density of this bin's contents, in kilograms per cubic metre, used to
    # convert the collected mass into the volume it occupies in the body.  Zero
    # means unknown, and the volume constraint is then skipped for this bin.
    density_kg_per_m3: float = 0.0

    @property
    def loose_volume_m3(self) -> float:
        """Volume this load occupies before compaction."""
        rho = float(self.density_kg_per_m3)
        if rho <= 0.0:
            return 0.0
        return max(0.0, _finite(self.load_kg)) / rho


@dataclass
class VehicleSpec:
    """One truck and the constraints bounding its shift."""

    vehicle_id: int
    name: str = "truck"
    depot_index: int = 0
    # Payload mass limit, in kilograms.  Mass is conserved: compacting waste does
    # not make it lighter, so this figure is never divided by the compaction
    # ratio.  An earlier version did exactly that, which is dimensionally wrong.
    capacity_kg: float = 6000.0
    shift_minutes: float = 480.0
    shift_start_minute: int = 360
    avg_speed_kmh: float = 20.0
    # Compaction reduces the *volume* the load occupies, which is why a refuse
    # body carries far more waste than its loose volume suggests.
    compaction_ratio: float = 2.5
    # Usable body volume in cubic metres.  Whether this or the mass limit binds
    # first depends on density; see `_exceeds` for the crossover, which is
    # 150 kg/m3 at the defaults and below every density configured here, so mass
    # binds in this deployment.  Left at zero the volume constraint is not
    # enforced, which keeps instances carrying no density information working as
    # pure mass problems.
    body_volume_m3: float = 0.0
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
    # Extra penalty for skipping a bin that has passed its equity deadline.  The
    # overdue tier orders such bins first, but ordering alone only decides which
    # bin the planner *tries* first: a prize-collecting objective can still drop
    # one when the detour costs more than the forgone prize.  Skipping has to be
    # the expensive option, or the wait bound is an ordering claim with no effect
    # on what actually gets collected.
    overdue_multiplier: float = 4.0
    missed_overflow_penalty: float = 80.0
    # A deadline beyond this horizon is not this shift's problem.  Without the
    # bound, a bin predicted to overflow in a fortnight attracts exactly the same
    # penalty as one overflowing this afternoon, which is both wrong and
    # inconsistent with `route_cost`, where a served bin is only ever charged for
    # being *actually* late.
    overflow_horizon_h: float = 24.0


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
    load_kg: float = 0.0            # raw mass collected, before compaction
    trips: int = 1

    @property
    def payload_kg(self) -> float:
        """
        Mass carried, which is what ``capacity_kg`` bounds.

        Identical to ``load_kg``: mass is conserved under compaction.  This
        property previously divided by the compaction ratio, which treated
        compacting the load as making it lighter.
        """
        return self.load_kg

    @property
    def compacted_volume_m3(self) -> float:
        """Body volume the load occupies after compaction."""
        loose = sum(s.task.loose_volume_m3 for s in self.stops)
        return loose / max(self.vehicle.compaction_ratio, 1e-9)

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
    load = 0.0             # mass on board, kilograms
    volume = 0.0           # compacted volume on board, cubic metres
    position = depot
    trip_index = 0

    for task in order:
        if not vehicle.accepts(task.stream):
            return None

        # A non-finite load would pass every comparison below (``nan > x`` is
        # False), so the route would look feasible while overflowing the body.
        # Mass is not divided by the compaction ratio: compacting waste changes
        # the volume it occupies, not its mass.
        added_mass = _finite(task.load_kg)
        added_volume = task.loose_volume_m3 / max(vehicle.compaction_ratio, 1e-9)
        if _exceeds(added_mass, added_volume, vehicle):
            return None                     # a single bin exceeds the body

        # --- tip at the depot when the next bin would overflow -----------
        if _exceeds(load + added_mass, volume + added_volume, vehicle):
            if not allow_multi_trip:
                return None
            back_min = travel.minutes(position, depot)
            back_dist = travel.distance(position, depot)
            back_co2 = travel.leg_co2(position, depot, load, vehicle.profile)
            clock += back_min + vehicle.tipping_minutes
            route.distance_m += back_dist
            route.co2_kg += back_co2
            load = 0.0
            volume = 0.0
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
            lifts=1, lifted_kg=added_mass,
        )

        # --- shift feasibility must include getting home ------------------
        return_min = travel.minutes(task.index, depot)
        if departure + return_min > vehicle.shift_minutes + 1e-9:
            return None

        load += added_mass
        volume += added_volume
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
    route.load_kg = sum(_finite(s.task.load_kg) for s in route.stops)
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


def skip_cost(task: BinTask, weights: ObjectiveWeights) -> float:
    """
    Cost of leaving one bin unserved.

    This is *the* definition, and everything that reasons about skipping must go
    through it.  The construction heuristic used to carry its own copy that
    omitted the overflow term, so a bin predicted to overflow was valued at
    ``lambda * prize`` while the objective the plan is scored against charged
    ``lambda * prize + missed_overflow_penalty``.  With the default weights that
    is an 80-unit blind spot aimed squarely at the bins the system exists to
    catch: any bin whose insertion cost fell in the gap was dropped during
    construction and only recovered if the local search happened to have budget
    left over to reinsert it.
    """
    # A mandatory bin is the head of the overdue queue. Charging it a finite
    # price, however large, still lets a large enough detour outweigh it, and
    # "large enough" is exactly the situation that starved a bin for 480 h. The
    # figure below cannot be outweighed by any route this problem admits, and it
    # stays finite so that objective arithmetic does not produce NaN.
    if task.mandatory:
        return MANDATORY_SKIP_COST
    if task.hazard or task.tier == TIER_HAZARD:
        penalty = (weights.lambda_prize * max(0.0, _finite(task.prize))
                   * weights.hazard_multiplier)
    elif task.tier == TIER_OVERDUE:
        # Charge the escalating pressure, not the bounded ordering score.  The
        # prize for an overdue bin is `w/(tau + w)`, which approaches 1 but never
        # reaches it, so pricing off the prize capped this penalty at
        # `lambda_prize * overdue_multiplier`, 180 at the default weights.  Any
        # bin whose marginal insertion cost exceeded 180, meaning roughly 33 km
        # from the depot, was then skipped at every wait from 48 h to a million
        # hours, and the overdue tier's ordering guarantee bought nothing: being
        # first in a queue nobody is served from is not service.  The `max`
        # keeps the old behaviour for a caller that supplied no wait, and is
        # never the binding term once one is supplied, because
        # `w/(2 tau) >= w/(tau + w)` for every `w >= tau`.
        pressure = max(max(0.0, _finite(task.prize)),
                       max(0.0, _finite(task.overdue_pressure)))
        penalty = weights.lambda_prize * pressure * weights.overdue_multiplier
    else:
        penalty = weights.lambda_prize * max(0.0, _finite(task.prize))
    if _overflows_within_horizon(task, weights):
        penalty += weights.missed_overflow_penalty
    return penalty


def _overflows_within_horizon(task: BinTask, weights: ObjectiveWeights) -> bool:
    """Whether skipping this bin means an overflow this planning horizon."""
    tto = task.time_to_overflow_h
    return math.isfinite(tto) and tto <= weights.overflow_horizon_h


def unserved_cost(tasks: Sequence[BinTask], weights: ObjectiveWeights) -> float:
    """Prize forgone by skipping bins, plus the overflows that skipping causes."""
    return float(sum(skip_cost(t, weights) for t in tasks))


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
                      travel: TravelModel, weights: ObjectiveWeights,
                      use_cache: bool = True
                      ) -> Tuple[Dict[int, List[BinTask]], List[BinTask]]:
    """
    Regret-2 insertion.

    At each step every unrouted bin is costed into every vehicle.  The bin
    chosen is the one with the largest *regret* -- the gap between its best and
    second-best insertion cost, scaled by its prize -- because that is the bin
    which will become most expensive if it is left for later.  Plain cheapest
    insertion is myopic in exactly the way that produces the detours the
    reviewers noticed in the original greedy tour.

    ``use_cache`` exists only so a test can run the same instance with the
    insertion table rebuilt from scratch every iteration and assert that the two
    produce the identical sequence of insertions.  Leave it on.
    """
    orders: Dict[int, List[BinTask]] = {v.vehicle_id: [] for v in vehicles}
    by_id = {v.vehicle_id: v for v in vehicles}
    pending = list(tasks)
    unserved: List[BinTask] = []

    # Hazard-tier bins are inserted first so they never lose a tie-break.  Inside
    # the overdue tier this is longest-wait-first, and the selection key below
    # keeps that order rather than overriding it with regret.
    pending.sort(key=lambda t: (t.tier, -t.prize))

    base_costs = {v.vehicle_id: 0.0 for v in vehicles}

    # Cheapest insertion of each pending bin into each vehicle, kept between
    # iterations.  Inserting a bin changes exactly one vehicle's route, so every
    # other (bin, vehicle) pairing is costed against a route that did not move
    # and its cached answer is still the answer.  Only the touched vehicle's
    # column is discarded.  This is memoisation and not an approximation: the
    # sequence of insertions is identical to recomputing the whole table each
    # iteration, which a test pins by comparing the two on the same instance.
    cache: Dict[Tuple[int, int], Optional[Tuple[float, int]]] = {}

    while pending:
        best_choice = None       # (regret_key, task, vehicle_id, position, delta)

        for task in pending:
            options: List[Tuple[float, int, int]] = []   # (delta, vehicle_id, pos)
            for v in vehicles:
                if not v.accepts(task.stream):
                    continue
                key_cache = (task.node_id, v.vehicle_id)
                if use_cache and key_cache in cache:
                    found = cache[key_cache]
                else:
                    found = _best_insertion(task, orders[v.vehicle_id], v, travel,
                                            weights, base_costs[v.vehicle_id])
                    cache[key_cache] = found
                if found is not None:
                    options.append((found[0], v.vehicle_id, found[1]))
            if not options:
                continue
            options.sort(key=lambda o: o[0])
            best_delta, vid, pos = options[0]
            second = options[1][0] if len(options) > 1 else best_delta + weights.lambda_prize
            regret = second - best_delta

            # Only insert when serving is cheaper than the penalty for skipping.
            # This must be the *same* quantity the objective charges, so it comes
            # from `skip_cost` rather than being recomputed here.
            forgone = skip_cost(task, weights)
            if best_delta > forgone:
                continue

            # Tier first, then it depends on the tier.
            #
            # Inside the overdue tier the service order *is* the guarantee, so it
            # is fixed by waiting time and regret only breaks ties.  `prize` for
            # an overdue bin is `overdue_score(w) = w/(tau + w)`, which is
            # strictly increasing in the wait, so ordering by descending prize is
            # ordering longest-wait-first.  Without this the constructor selected
            # by regret inside the tier, which let a bin promoted this cycle
            # overtake one promoted earlier.  The queueing term of the wait bound
            # assumes it cannot, so the proof was describing an order the code did
            # not implement.
            #
            # Outside the overdue tier there is no ordering claim to keep, and
            # regret is the right criterion: it picks the bin that will become
            # most expensive if it is left for later.
            if task.tier == TIER_OVERDUE:
                key = (task.tier, -_finite(task.prize),
                       -(regret + forgone - best_delta))
            else:
                key = (task.tier, 0.0, -(regret + forgone - best_delta))
            if best_choice is None or key < best_choice[0]:
                best_choice = (key, task, vid, pos, best_delta)

        if best_choice is None:
            unserved.extend(pending)
            break

        _, task, vid, pos, delta = best_choice
        orders[vid] = orders[vid][:pos] + [task] + orders[vid][pos:]
        base_costs[vid] += delta
        pending.remove(task)
        for cached_key in [k for k in cache if k[1] == vid]:
            del cache[cached_key]

    return orders, unserved


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------
def _bulk_insertion(vehicle_id: int, vehicle: VehicleSpec, order: List[BinTask],
                    pool: Sequence[BinTask], travel: TravelModel,
                    weights: ObjectiveWeights, base_cost: float,
                    deadline: float
                    ) -> Optional[Tuple[float, List[BinTask], List[BinTask]]]:
    """
    Greedily insert as many pooled bins into one vehicle as pays off *in total*.

    Bins are added by cheapest insertion, one at a time, but the acceptance test
    is applied to the running total rather than to each bin: after `k` additions
    the score is ``(route cost now - route cost before) - (skip cost of all k)``.
    The prefix minimising that total is returned.

    This is the whole point.  The marginal cost of the first bin onto an empty
    vehicle carries the entire fixed cost of opening a route, so it is dominated
    by its own prize and gets rejected; every subsequent bin in the same cluster
    would then have been nearly free.  A greedy test that never looks past the
    first bin cannot see that, and the result is stranded clusters next to idle
    trucks.

    Returns ``(total_delta, new_order, tasks_taken)`` or ``None`` when nothing
    can be inserted.  The caller still applies the usual strict-improvement test.
    """
    candidates = [t for t in pool if vehicle.accepts(t.stream)]
    if not candidates:
        return None

    current = list(order)
    remaining = list(candidates)
    taken: List[BinTask] = []
    forgone = 0.0
    best: Optional[Tuple[float, List[BinTask], List[BinTask]]] = None

    while remaining:
        if time.perf_counter() > deadline:
            break
        evaluated = evaluate_route(current, vehicle, travel) if current else None
        current_cost = route_cost(evaluated, weights) if evaluated is not None else 0.0

        choice: Optional[Tuple[float, int, BinTask]] = None
        for task in remaining:
            found = _best_insertion(task, current, vehicle, travel, weights,
                                    current_cost)
            if found is None:
                continue
            if choice is None or found[0] < choice[0]:
                choice = (found[0], found[1], task)
        if choice is None:
            break

        _, position, task = choice
        current = current[:position] + [task] + current[position:]
        taken.append(task)
        remaining = [t for t in remaining if t is not task]
        forgone += skip_cost(task, weights)

        evaluated = evaluate_route(current, vehicle, travel)
        if evaluated is None:                  # should not happen; stay safe
            break
        total_delta = route_cost(evaluated, weights) - base_cost - forgone
        if best is None or total_delta < best[0]:
            best = (total_delta, list(current), list(taken))

    return best


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
    ``eject``           drop a bin whose prize no longer pays for its detour.

    ``eject`` is the mirror of ``reinsert`` and the neighbourhood is incomplete
    without it: this is a *prize-collecting* problem, so the set of served bins
    is a decision variable in both directions.  A bin that was worth inserting
    into the route the constructor built can easily stop being worth it once
    relocations and swaps have reshaped that route around it, and with only the
    inserting move available the search had no way to undo the commitment.

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

        # --- close a route: redistribute one vehicle's whole workload ----
        # The exact mirror of the bulk reinsertion below, and needed for the
        # same reason.  Emptying a vehicle is only worth it once *all* of its
        # bins have moved elsewhere; every intermediate state still pays for an
        # open route, so a search that relocates one bin at a time is stopped by
        # a barrier it can never cross, and a plan that opened a second truck
        # can never collapse back onto one.  Costing the whole redistribution as
        # a single move removes the barrier.
        for a in vids:
            if not orders[a]:
                continue
            if time.perf_counter() > deadline:
                return orders, unserved
            trial = {vid: list(orders[vid]) for vid in vids}
            leftovers: List[BinTask] = []
            for task in orders[a]:
                best_spot = None
                for b in vids:
                    if b == a or not by_id[b].accepts(task.stream):
                        continue
                    base = vehicle_cost(b, trial[b])
                    if base is None:
                        continue
                    found = _best_insertion(task, trial[b], by_id[b], travel,
                                            weights, base)
                    if found is not None and (best_spot is None
                                              or found[0] < best_spot[0]):
                        best_spot = (found[0], found[1], b)
                if best_spot is None:
                    leftovers.append(task)
                else:
                    _, position, b = best_spot
                    trial[b] = trial[b][:position] + [task] + trial[b][position:]
            trial[a] = []
            changes = {vid: order for vid, order in trial.items()
                       if len(order) != len(orders[vid])
                       or any(x is not y for x, y in zip(order, orders[vid]))}
            if changes and try_move(changes, new_unserved=unserved + leftovers):
                improved = True

        # --- bring skipped bins back into service, in bulk ---------------
        # Deliberately *not* one bin at a time.  Serving a distant bin alone
        # means opening a route for it -- depot out, depot back, a tipping stop
        # -- which almost never pays for a single prize, so a one-bin-at-a-time
        # test rejects every candidate and an idle vehicle stays idle no matter
        # how many bins are stranded.  Measured on a ten-bin instance with three
        # trucks: four bins abandoned at 42.75 penalty each while two vehicles
        # sat unused, because the *first* bin onto an empty truck cost 47.  The
        # same four bins together cost 55 for all of them.  Costing the batch
        # cumulatively, and keeping the prefix that minimises the total, is what
        # lets the search discover that a second route is worth opening.
        for vid in vids:
            if time.perf_counter() > deadline:
                return orders, unserved
            batch = _bulk_insertion(vid, by_id[vid], orders[vid], unserved,
                                    travel, weights, costs[vid], deadline)
            if batch is None:
                continue
            _, new_order, taken = batch
            remaining = [t for t in unserved if not any(t is x for x in taken)]
            if try_move({vid: new_order}, new_unserved=remaining):
                improved = True

        # --- drop a bin that no longer pays for itself -------------------
        for vid in vids:
            i = 0
            while i < len(orders[vid]):
                if time.perf_counter() > deadline:
                    return orders, unserved
                task = orders[vid][i]
                trimmed = orders[vid][:i] + orders[vid][i + 1:]
                if try_move({vid: trimmed}, new_unserved=unserved + [task]):
                    improved = True
                else:
                    i += 1

        if not improved:
            break

    return orders, unserved


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------
#: Fixed so that "deterministic given the inputs" stays true.  The ruin step
#: needs to make arbitrary choices; making them pseudo-random from a constant
#: seed keeps the search unbiased without making the published numbers depend on
#: an unrecorded seed.
RUIN_SEED = 20240517
#: Consecutive non-improving ruin-and-recreate rounds before giving up.
RUIN_PATIENCE = 30
#: Share of the served bins torn out per round, cycled across rounds.
RUIN_FRACTIONS = (0.15, 0.25, 0.40, 0.25)


#: What the planner is charged for declining the head of the overdue queue.
#: Large enough that no route in this problem can outweigh it, finite so that
#: sums and differences of objectives stay numbers.
MANDATORY_SKIP_COST = 1e9


def mark_overdue_head(tasks: Sequence[BinTask]) -> Optional[BinTask]:
    """
    Mark the longest-waiting promoted bin as mandatory, and return it.

    `overdue_pressure` is `wait/(2 tau)` for a promoted bin and zero otherwise,
    so a positive value is the promotion test and its magnitude is the wait.
    Selection ignores the tier on purpose: a bin that is both hazardous and
    overdue is filed under the hazard tier, and keying on the tier would skip
    exactly the bins that are most overdue.
    """
    promoted = [t for t in tasks if _finite(t.overdue_pressure) > 0.0]
    if not promoted:
        return None
    head = max(promoted, key=lambda t: _finite(t.overdue_pressure))
    head.mandatory = True
    return head


def reserve_overdue_head(orders: Dict[int, List[BinTask]],
                         unserved: List[BinTask],
                         by_id: Dict[int, VehicleSpec],
                         travel: TravelModel,
                         weights: ObjectiveWeights
                         ) -> Tuple[Dict[int, List[BinTask]], List[BinTask]]:
    """
    Serve the oldest overdue bin, ejecting whatever is needed to make room.

    The wait bound needs the overdue tier served oldest first.  Construction
    enforces that, and the descent that follows does not: every move the descent
    makes is judged on the objective, and an objective is a preference rather
    than a constraint.  Measured on the rollout, a younger overdue bin was served
    ahead of an older one in 20 percent of contested cycles, and a bin reached
    180 h against a bound of 144 h.  Pricing is not the defect.  The escalating
    skip price does buy service: the farthest bin in the instance is collected at
    every wait from 48 h upward while 32 others compete for the same shift.  What
    fails is that nothing obliges the planner to take the *oldest* one.

    So this makes the head of the tier a constraint.  The oldest overdue bin is
    placed first and the rest of that vehicle's round is rebuilt around it, each
    bin re-inserted at its cheapest feasible position and dropped when it no
    longer fits.  The vehicle chosen is the one whose rebuilt round costs least.

    A bin that no vehicle can serve even alone is left where it is.  No dispatch
    rule can collect it, and assumption (A2) of the bound excludes it already;
    forcing the issue here would only produce an infeasible plan.

    The objective this gives up is the price of the guarantee, and it is measured
    rather than assumed: `solve(..., reserve_overdue=False)` is the same planner
    without the rule.
    """
    # Selection is on the wait, not on the tier, and the difference is not
    # cosmetic.  A bin that is both hazardous and overdue is assigned
    # `TIER_HAZARD` by `aging.effective_priorities`, so filtering on
    # `TIER_OVERDUE` silently drops exactly the bins that are most overdue and
    # most urgent.  That defect left the head of the queue unserved in 1 of 96
    # contested cycles here, which is one more than a guarantee permits.
    # `overdue_pressure` is `wait/(2 tau)` for a promoted bin and zero for every
    # other, so a positive value *is* the promotion test, and it is strictly
    # increasing in the wait.
    promoted = [t for order in orders.values() for t in order
                if _finite(t.overdue_pressure) > 0.0]
    promoted += [t for t in unserved if _finite(t.overdue_pressure) > 0.0]
    if not promoted:
        return orders, unserved

    oldest = max(promoted, key=lambda t: _finite(t.overdue_pressure))
    if any(t is oldest for order in orders.values() for t in order):
        return orders, unserved                      # already at the front

    best: Optional[Tuple[float, int, List[BinTask], List[BinTask]]] = None
    for vid, order in orders.items():
        vehicle = by_id[vid]
        if not vehicle.accepts(oldest.stream):
            continue
        if evaluate_route([oldest], vehicle, travel) is None:
            continue                                 # (A2): unservable even alone

        kept: List[BinTask] = [oldest]
        base = route_cost(evaluate_route(kept, vehicle, travel), weights)
        dropped: List[BinTask] = []
        for task in order:
            spot = _best_insertion(task, kept, vehicle, travel, weights, base)
            if spot is None:
                dropped.append(task)
                continue
            delta, pos = spot
            kept = kept[:pos] + [task] + kept[pos:]
            base += delta

        # Score the whole change, not just this vehicle: the bins dropped here
        # become unserved and are charged for it.
        cost = base + sum(skip_cost(t, weights) for t in dropped)
        previous = evaluate_route(order, vehicle, travel) if order else None
        cost -= route_cost(previous, weights) if previous is not None else 0.0
        cost -= skip_cost(oldest, weights)
        if best is None or cost < best[0]:
            best = (cost, vid, kept, dropped)

    if best is None:
        return orders, unserved

    _cost, vid, kept, dropped = best
    orders = {k: (list(kept) if k == vid else list(v)) for k, v in orders.items()}
    unserved = [t for t in unserved if t is not oldest] + dropped
    return orders, unserved


def _objective_now(orders: Dict[int, List[BinTask]], unserved: Sequence[BinTask],
                   by_id: Dict[int, VehicleSpec], travel: TravelModel,
                   weights: ObjectiveWeights) -> float:
    total = 0.0
    for vid, order in orders.items():
        if not order:
            continue
        evaluated = evaluate_route(order, by_id[vid], travel)
        if evaluated is None:
            return math.inf
        total += route_cost(evaluated, weights)
    return total + unserved_cost(unserved, weights)


def solve(tasks: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
          travel: TravelModel, weights: Optional[ObjectiveWeights] = None,
          improve: bool = True, time_budget_s: float = 6.0,
          algorithm: str = "regret2_ls",
          reserve_overdue: bool = True) -> FleetPlan:
    """
    Construct, then improve, then report a fully evaluated fleet plan.

    Improvement is a descent to a local optimum followed by bounded
    ruin-and-recreate: a slice of the served bins is torn out and the descent is
    run again, keeping the incumbent only when it improves.  A single descent
    from a single construction is fragile in a way that matters here -- changing
    the constructor's skip rule, which is unambiguously more correct, moved the
    live instance into a *worse* basin, because the starting point determines
    which local optimum is reachable and nothing else does.  Restarting keeps
    the best solution seen, so the reported number can never be worse than the
    plain descent, and it uses time the planner already had (a descent converges
    in ~50 ms against a 5 s budget).
    """
    weights = weights or ObjectiveWeights()
    started = time.perf_counter()
    deadline = started + max(0.1, time_budget_s)
    by_id = {v.vehicle_id: v for v in vehicles}

    # Mark the head before anything looks at the tasks, so construction, the
    # descent and the ruin-and-recreate all price it the same way. Marking it
    # afterwards would leave the search free to discard it and leave the repair
    # to rebuild a round the search never evaluated.
    head = mark_overdue_head(tasks) if reserve_overdue else None
    orders, unserved = construct_regret2(tasks, vehicles, travel, weights)
    if improve:
        def descend(current_orders, current_unserved):
            remaining = deadline - time.perf_counter()
            return local_search(current_orders, vehicles, travel, weights,
                                current_unserved,
                                time_budget_s=max(0.05, remaining))

        orders, unserved = descend(orders, unserved)
        best_orders = {vid: list(order) for vid, order in orders.items()}
        best_unserved = list(unserved)
        best_objective = _objective_now(best_orders, best_unserved, by_id,
                                        travel, weights)

        rng = random.Random(RUIN_SEED)
        served_total = sum(len(order) for order in best_orders.values())
        # Give up once restarts stop paying, rather than burning the whole
        # budget every time.  A plan request is an interactive action; spending
        # five seconds to confirm a solution found in fifty milliseconds is a
        # latency cost with no benefit.
        stale = 0
        iteration = 0
        # A descent is not cheap at fleet scale (~0.8 s for 24 bins), so a round
        # started near the deadline is abandoned half-finished and its budget is
        # simply wasted.  Requiring room for a full round -- estimated from the
        # rounds already timed -- is what keeps `time_budget_s` an honest bound
        # rather than a target the solver always spends in full.
        round_cost = time.perf_counter() - started
        while (served_total > 1 and stale < RUIN_PATIENCE
               and time.perf_counter() + round_cost < deadline):
            iteration += 1
            round_started = time.perf_counter()
            trial_orders = {vid: list(order) for vid, order in best_orders.items()}
            trial_unserved = list(best_unserved)

            pool = [(vid, task) for vid, order in trial_orders.items() for task in order]
            # The ruin size is cycled rather than fixed: a small tear intensifies
            # around the incumbent, a large one is the only way out of a deep
            # basin, and neither alone does both jobs.
            fraction = RUIN_FRACTIONS[iteration % len(RUIN_FRACTIONS)]
            k = max(1, min(len(pool) - 1, int(round(fraction * len(pool)))))

            if iteration % 2 == 0:
                victims = rng.sample(pool, k)
            else:
                # Related ("Shaw") removal: tear out a geographic neighbourhood
                # rather than a scatter.  Removing bins at random leaves the
                # route's shape essentially intact, so the repair puts almost
                # everything back where it was; removing a contiguous cluster
                # lets the repair rebuild that part of the plan from scratch,
                # which is what actually escapes a local optimum in a routing
                # problem.
                anchor = pool[rng.randrange(len(pool))]
                victims = sorted(
                    pool, key=lambda vt: travel.distance(anchor[1].index,
                                                         vt[1].index))[:k]

            for vid, task in victims:
                trial_orders[vid] = [t for t in trial_orders[vid] if t is not task]
                trial_unserved.append(task)

            trial_orders, trial_unserved = descend(trial_orders, trial_unserved)
            trial_objective = _objective_now(trial_orders, trial_unserved, by_id,
                                             travel, weights)
            if trial_objective < best_objective - 1e-9:
                best_orders = {vid: list(o) for vid, o in trial_orders.items()}
                best_unserved = list(trial_unserved)
                best_objective = trial_objective
                served_total = sum(len(o) for o in best_orders.values())
                stale = 0
            else:
                stale += 1
            # Track the worst round seen, so the guard above is conservative.
            round_cost = max(round_cost, time.perf_counter() - round_started)

        orders, unserved = best_orders, best_unserved

    # The descent optimises the objective and does not consult the tier, so it
    # can and does take the oldest overdue bin back out of the plan it was
    # constructed into.  Restoring it here is what makes the wait bound a
    # guarantee rather than a tendency; see `reserve_overdue_head`.
    if reserve_overdue:
        orders, unserved = reserve_overdue_head(orders, unserved, by_id,
                                                travel, weights)

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

    # The mark belongs to this call. Leaving it set would make the bin
    # mandatory in every later plan built from the same task objects, and would
    # put MANDATORY_SKIP_COST into the reported objective if it were still
    # unserved, which would corrupt every comparison that reads it.
    if head is not None:
        head.mandatory = False

    objective = plan_objective(routes, unserved, weights)
    plan = FleetPlan(
        routes=routes,
        unserved=unserved,
        objective=objective,
        metrics=summarise(routes, unserved, tasks, weights),
        compute_ms=(time.perf_counter() - started) * 1000.0,
        algorithm=algorithm,
    )
    return plan


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarise(routes: Sequence[VehicleRoute], unserved: Sequence[BinTask],
              all_tasks: Sequence[BinTask],
              weights: Optional[ObjectiveWeights] = None) -> Dict:
    """Service, sustainability and equity metrics for a completed plan."""
    weights = weights or ObjectiveWeights()
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
    # A skipped bin that would overflow inside the horizon is also a miss.  The
    # horizon test matters: counting every finite deadline made a bin due to
    # overflow next week indistinguishable from one due this afternoon, and it
    # disagreed with the objective, which only ever charges a served bin for
    # being genuinely late.
    for t in unserved:
        if _overflows_within_horizon(t, weights):
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
        "payload_kg": round(sum(r.payload_kg for r in routes), 1),
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
        # Measured against the constraint that actually binds: compacted payload
        # versus body capacity, summed over trips because each trip is a fresh
        # body.  Using the raw lifted mass here reported 250% for a route that
        # filled its body exactly three times.
        "capacity_utilisation_pct": round(
            100.0 * sum(r.payload_kg for r in routes)
            / max(1e-9, sum(r.vehicle.capacity_kg * r.trips for r in routes)), 2),
    }


def plan_to_dict(plan: FleetPlan) -> Dict:
    """
    JSON-serialisable representation for the API and the audit ledger.

    Every numeric field is forced finite on the way out.  ``json.dumps`` happily
    emits bare ``NaN`` and ``Infinity`` tokens, which are not valid JSON: browsers
    and the ledger's hash-chain verifier both choke on them, and a single
    unmodelled bin was enough to produce one.
    """
    def num(value: float, digits: int) -> float:
        return round(_finite(value), digits)

    return {
        "algorithm": plan.algorithm,
        "objective": num(plan.objective, 4),
        "compute_ms": num(plan.compute_ms, 2),
        "metrics": plan.metrics,
        "routes": [
            {
                "vehicle_id": r.vehicle.vehicle_id,
                "vehicle_name": r.vehicle.name,
                "trips": r.trips,
                "distance_km": num(r.distance_m / 1000.0, 3),
                "co2_kg": num(r.co2_kg, 3),
                "duration_min": num(r.duration_min, 2),
                "load_kg": num(r.load_kg, 1),
                "payload_kg": num(r.payload_kg, 1),
                "capacity_kg": r.vehicle.capacity_kg,
                "stops": [
                    {
                        "node_id": s.task.node_id,
                        "sequence": i,
                        "trip_index": s.trip_index,
                        "arrival_min": num(s.arrival_min, 2),
                        "start_service_min": num(s.start_service_min, 2),
                        "departure_min": num(s.departure_min, 2),
                        "wait_min": num(s.wait_min, 2),
                        "leg_distance_m": num(s.leg_distance_m, 1),
                        "leg_co2_kg": num(s.leg_co2_kg, 4),
                        "load_after_kg": num(s.load_after_kg, 1),
                        "prize": num(s.task.prize, 4),
                        "hazard": bool(s.task.hazard),
                    }
                    for i, s in enumerate(r.stops)
                ],
            }
            for r in plan.routes
        ],
        "unserved": [
            {"node_id": t.node_id, "prize": num(t.prize, 4), "hazard": bool(t.hazard)}
            for t in plan.unserved
        ],
    }
