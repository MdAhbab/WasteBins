"""
Competing routing methods, evaluated on identical terms.
========================================================

Reviewer 2 objected that the proposed policy was only ever compared against a
static sweep.  This module supplies the missing comparators, each implemented
against the *same* :func:`wastebins_core.vrp.evaluate_route` feasibility
simulator and the *same* objective, so the comparison isolates the search
strategy rather than differences in what each method is allowed to ignore.

Implemented
-----------
``genetic``
    Permutation genetic algorithm in the Prins (2004) "route-first,
    cluster-second" style: a chromosome is a giant tour over all bins, decoded
    into vehicle routes by a feasibility-aware split.  Order crossover (OX),
    swap/inversion mutation, tournament selection and elitism.

``aco``
    Max-Min Ant System (Stutzle & Hoos 2000).  Ants build routes probabilistically
    from pheromone and a prize-per-metre heuristic; only the iteration-best ant
    deposits, and pheromone is clamped to [tau_min, tau_max] to delay stagnation.

``risk_graph``
    A priority-warped greedy walk: edge weights are deflated by the destination's
    priority, so an urgent container appears nearer, and the tour is the greedy
    walk on that warped graph.  This is the method this project itself used
    before the fleet model was built, so it establishes whether the gain comes
    from the warping idea or from the fleet model constructed around it.  It is
    not attributed to any particular publication, because we have not verified
    that any specific paper uses this exact form.

``ortools``
    Optional. Uses Google OR-Tools routing when it is installed, giving a
    strong reference point; the module reports its absence rather than silently
    skipping the comparison.

Every solver honours a wall-clock ``time_budget_s`` and a seed, and returns a
:class:`wastebins_core.vrp.FleetPlan` so downstream reporting is uniform.

What a fair comparison needed
-----------------------------
A second review found three ways in which these comparators were weaker than
their names suggested, and each was a property of this module rather than of
the methods.

* The decoder could not decline a container.  It skipped one only when nothing
  fitted, so on a prize-collecting problem the genetic algorithm and the ant
  colony were solving the visit-everything problem and being scored on the
  other one.  `split_giant_tour` now declines a container whose marginal cost
  exceeds its skip penalty.
* The iteration caps were set for a three-second budget.  Given longer, the ant
  colony stopped at sixty iterations and idled.  Both population methods now run
  until the budget is spent.
* OR-Tools was never told which vehicle may carry which stream, was not charged
  for waiting or for the tipping stop, and knew nothing of overflow deadlines.
  On a network with two streams it put every container on a vehicle licensed
  for one of them and the decoder then dropped half the plan.  All four are now
  in its model.

Every one of these makes a baseline stronger.  None of them changes what a plan
is scored on.
"""
from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import emissions as EM
from . import vrp
from .vrp import BinTask, FleetPlan, ObjectiveWeights, TravelModel, VehicleSpec


def EM_NOMINAL_CO2_PER_KM(profile: EM.VehicleProfile) -> float:
    """Nominal kg CO2 per km at half payload in moderate traffic, for solvers
    that can only carry a single scalar factor on an arc."""
    leg = EM.leg_emissions(1000.0, 20.0, profile.capacity_kg * 0.5,
                           friction=0.35, profile=profile)
    return leg.co2_kg(profile)


class FlatEmissionTravel(TravelModel):
    """
    The same distances and speeds, with emissions as one factor per kilometre.

    This is the emission model a solver with fixed arc costs can represent
    exactly.  Scoring every plan under it answers a specific question: how much
    of a margin over such a solver comes from the solver being scored on a
    model it cannot see.
    """

    def __init__(self, base: TravelModel, co2_per_km: float):
        super().__init__(base.distance_m, travel_context=base.ctx,
                         default_speed_kmh=base.default_speed_kmh,
                         freeflow_kmh=base.freeflow_kmh)
        self.co2_per_km = float(co2_per_km)

    def leg_co2(self, i: int, j: int, payload_kg: float,
                profile: EM.VehicleProfile, idle_minutes: float = 0.0,
                lifts: int = 0, lifted_kg: float = 0.0) -> float:
        return self.co2_per_km * self.distance(i, j) / 1000.0


# ---------------------------------------------------------------------------
# Shared decoding: giant tour -> feasible vehicle routes
# ---------------------------------------------------------------------------
def split_giant_tour(tour: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
                     travel: TravelModel, weights: ObjectiveWeights,
                     prize_collecting: bool = False
                     ) -> Tuple[Dict[int, List[BinTask]], List[BinTask]]:
    """
    Decode a permutation into vehicle routes.

    Bins are assigned to the current vehicle while the route stays feasible; when
    the next bin cannot be appended, the vehicle is closed and the next one
    opened.  A bin that no vehicle can take is left unserved and pays the prize
    penalty.

    With ``prize_collecting`` the decoder also makes the selection decision the
    problem contains.  A bin is appended only where the cost it adds to that
    route is no more than the penalty for skipping it, which is the test the
    objective applies.  Without it a permutation method visits every bin that
    fits, pays for the detours, and is then scored against planners that were
    allowed to leave a bin out.
    """
    orders: Dict[int, List[BinTask]] = {v.vehicle_id: [] for v in vehicles}
    costs: Dict[int, float] = {v.vehicle_id: 0.0 for v in vehicles}
    unserved: List[BinTask] = []
    vehicle_iter = list(vehicles)
    cursor = 0

    for task in tour:
        placed = False
        forgone = vrp.skip_cost(task, weights) if prize_collecting else math.inf
        # Try the active vehicle first, then any later one, then any earlier one.
        order_of_attempt = list(range(cursor, len(vehicle_iter))) + list(range(0, cursor))
        for vi in order_of_attempt:
            vehicle = vehicle_iter[vi]
            if not vehicle.accepts(task.stream):
                continue
            vid = vehicle.vehicle_id
            trial = orders[vid] + [task]
            evaluated = vrp.evaluate_route(trial, vehicle, travel)
            if evaluated is None:
                continue
            if prize_collecting:
                cost = vrp.route_cost(evaluated, weights)
                if cost - costs[vid] > forgone:
                    continue                 # it fits here, but does not pay here
                costs[vid] = cost
            orders[vid] = trial
            cursor = vi
            placed = True
            break
        if not placed:
            unserved.append(task)

    return orders, unserved


def _finalise(orders: Dict[int, List[BinTask]], unserved: List[BinTask],
              vehicles: Sequence[VehicleSpec], travel: TravelModel,
              weights: ObjectiveWeights, all_tasks: Sequence[BinTask],
              algorithm: str, compute_ms: float) -> FleetPlan:
    by_id = {v.vehicle_id: v for v in vehicles}
    routes = []
    leftovers = list(unserved)
    for vid, order in orders.items():
        if not order:
            continue
        evaluated = vrp.evaluate_route(order, by_id[vid], travel)
        if evaluated is None:
            feasible: List[BinTask] = []
            for task in order:
                if vrp.evaluate_route(feasible + [task], by_id[vid], travel) is not None:
                    feasible.append(task)
                else:
                    leftovers.append(task)
            evaluated = vrp.evaluate_route(feasible, by_id[vid], travel) if feasible else None
        if evaluated is not None:
            routes.append(evaluated)
    return FleetPlan(
        routes=routes,
        unserved=leftovers,
        objective=vrp.plan_objective(routes, leftovers, weights),
        metrics=vrp.summarise(routes, leftovers, all_tasks, weights),
        compute_ms=compute_ms,
        algorithm=algorithm,
    )


def _objective_of(orders: Dict[int, List[BinTask]], unserved: List[BinTask],
                  vehicles: Sequence[VehicleSpec], travel: TravelModel,
                  weights: ObjectiveWeights) -> float:
    by_id = {v.vehicle_id: v for v in vehicles}
    total = 0.0
    for vid, order in orders.items():
        if not order:
            continue
        r = vrp.evaluate_route(order, by_id[vid], travel)
        if r is None:
            return math.inf
        total += vrp.route_cost(r, weights)
    return total + vrp.unserved_cost(unserved, weights)


def _polish(orders: Dict[int, List[BinTask]], unserved: List[BinTask],
            vehicles: Sequence[VehicleSpec], travel: TravelModel,
            weights: ObjectiveWeights, deadline: float
            ) -> Tuple[Dict[int, List[BinTask]], List[BinTask]]:
    """
    Spend what is left of the budget on the shared local search.

    A population method with no local search is the weak form of the method:
    Prins's algorithm is memetic.  The descent used here is the one the proposed
    planner uses, over the same neighbourhoods, so a hybrid baseline and the
    proposed method differ in how they reach a starting point and in nothing
    they are allowed to do afterwards.
    """
    remaining = deadline - time.perf_counter()
    if remaining <= 0.1:
        return orders, unserved
    return vrp.local_search(orders, vehicles, travel, weights, list(unserved),
                            time_budget_s=remaining)


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
def genetic_algorithm(tasks: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
                      travel: TravelModel, weights: Optional[ObjectiveWeights] = None,
                      population_size: int = 40, generations: Optional[int] = None,
                      crossover_rate: float = 0.85, mutation_rate: float = 0.20,
                      elite: int = 4, tournament: int = 3,
                      seed: int = 42, time_budget_s: float = 8.0,
                      prize_collecting: bool = True,
                      polish_fraction: float = 0.0) -> FleetPlan:
    """
    Route-first cluster-second GA with order crossover.

    ``generations`` caps the run when given; left at ``None`` the algorithm runs
    until the budget is spent.  ``polish_fraction`` is the share of the budget
    held back for a final descent on the best individual.
    """
    weights = weights or ObjectiveWeights()
    rng = random.Random(seed)
    started = time.perf_counter()
    deadline = started + max(0.2, time_budget_s)
    evolve_until = started + max(0.2, time_budget_s) * (1.0 - max(0.0, min(0.9, polish_fraction)))
    task_list = list(tasks)
    n = len(task_list)
    if n == 0:
        return _finalise({}, [], vehicles, travel, weights, tasks, "genetic", 0.0)

    def decode(chromosome: List[int]):
        return split_giant_tour([task_list[i] for i in chromosome],
                                vehicles, travel, weights,
                                prize_collecting=prize_collecting)

    def fitness(chromosome: List[int]) -> float:
        orders, unserved = decode(chromosome)
        return _objective_of(orders, unserved, vehicles, travel, weights)

    # --- seed the population with informed and random individuals -------
    base = list(range(n))
    by_prize = sorted(base, key=lambda i: (task_list[i].tier, -task_list[i].prize))
    by_index = sorted(base, key=lambda i: task_list[i].index)
    population: List[List[int]] = [by_prize[:], by_index[:]]
    while len(population) < population_size:
        individual = base[:]
        rng.shuffle(individual)
        population.append(individual)

    scored = [(fitness(c), c) for c in population]
    scored.sort(key=lambda s: s[0])
    best_score, best = scored[0][0], scored[0][1][:]

    def select() -> List[int]:
        pool = [scored[rng.randrange(len(scored))] for _ in range(tournament)]
        return min(pool, key=lambda s: s[0])[1]

    def order_crossover(p1: List[int], p2: List[int]) -> List[int]:
        a, b = sorted(rng.sample(range(n), 2)) if n > 1 else (0, 0)
        child: List[Optional[int]] = [None] * n
        child[a:b + 1] = p1[a:b + 1]
        taken = set(p1[a:b + 1])
        fill = [g for g in p2 if g not in taken]
        k = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[k]
                k += 1
        return [g for g in child if g is not None]

    def mutate(chromosome: List[int]) -> List[int]:
        c = chromosome[:]
        if n < 2:
            return c
        if rng.random() < 0.5:
            i, j = rng.randrange(n), rng.randrange(n)
            c[i], c[j] = c[j], c[i]
        else:
            i, j = sorted(rng.sample(range(n), 2))
            c[i:j + 1] = reversed(c[i:j + 1])
        return c

    generation = 0
    while (generations is None or generation < generations) \
            and time.perf_counter() < evolve_until:
        generation += 1
        next_pop = [c[:] for _, c in scored[:elite]]
        while len(next_pop) < population_size:
            p1, p2 = select(), select()
            child = order_crossover(p1, p2) if rng.random() < crossover_rate else p1[:]
            if rng.random() < mutation_rate:
                child = mutate(child)
            next_pop.append(child)
            if time.perf_counter() > evolve_until:
                break
        scored = [(fitness(c), c) for c in next_pop]
        scored.sort(key=lambda s: s[0])
        if scored[0][0] < best_score:
            best_score, best = scored[0][0], scored[0][1][:]

    orders, unserved = decode(best)
    if polish_fraction > 0.0:
        orders, unserved = _polish(orders, unserved, vehicles, travel, weights, deadline)
    plan = _finalise(orders, unserved, vehicles, travel, weights, tasks,
                     "genetic", (time.perf_counter() - started) * 1000.0)
    plan.metrics["generations"] = generation
    plan.metrics["population"] = population_size
    return plan


# ---------------------------------------------------------------------------
# Ant colony optimisation (Max-Min Ant System)
# ---------------------------------------------------------------------------
def ant_colony(tasks: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
               travel: TravelModel, weights: Optional[ObjectiveWeights] = None,
               n_ants: int = 16, iterations: Optional[int] = None,
               alpha: float = 1.0, beta: float = 2.5, rho: float = 0.10,
               q0: float = 0.25, seed: int = 42,
               time_budget_s: float = 8.0,
               prize_collecting: bool = True,
               polish_fraction: float = 0.0) -> FleetPlan:
    """
    Max-Min Ant System over the bin graph.

    Pheromone lives on ordered pairs of bins (plus a depot-start row).  The
    heuristic is prize per metre, so ants are drawn towards urgent bins that are
    also cheap to reach.  Only the iteration-best ant reinforces, and pheromone
    is clamped, which is what distinguishes MMAS from classic AS and keeps it
    from collapsing onto one tour within a handful of iterations.

    ``iterations`` caps the run when given; left at ``None`` the colony runs
    until the budget is spent.  ``polish_fraction`` is the share of the budget
    held back for a final descent on the best tour.
    """
    weights = weights or ObjectiveWeights()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    started = time.perf_counter()
    deadline = started + max(0.2, time_budget_s)
    build_until = started + max(0.2, time_budget_s) * (1.0 - max(0.0, min(0.9, polish_fraction)))

    task_list = list(tasks)
    n = len(task_list)
    if n == 0:
        return _finalise({}, [], vehicles, travel, weights, tasks, "aco", 0.0)

    depot = vehicles[0].depot_index

    # Heuristic desirability: prize gained per metre travelled.
    eta = np.zeros((n + 1, n), dtype=float)
    for j, tj in enumerate(task_list):
        d0 = max(travel.distance(depot, tj.index), 1.0)
        eta[0, j] = (tj.prize + 0.05) / d0
        for i, ti in enumerate(task_list):
            if i == j:
                continue
            d = max(travel.distance(ti.index, tj.index), 1.0)
            eta[i + 1, j] = (tj.prize + 0.05) / d
    eta *= 1000.0                                  # scale into a workable range

    tau_max = 1.0
    tau_min = tau_max / (2.0 * n)
    tau = np.full((n + 1, n), tau_max, dtype=float)

    best_score = math.inf
    best_tour: List[int] = []

    def build_tour() -> List[int]:
        unvisited = set(range(n))
        tour: List[int] = []
        current = 0                                # row 0 == depot
        while unvisited:
            candidates = sorted(unvisited)
            w = np.array([(tau[current, j] ** alpha) * (eta[current, j] ** beta)
                          for j in candidates], dtype=float)
            # Hazard-tier bins get a hard preference, matching the dispatch rule.
            for k, j in enumerate(candidates):
                if task_list[j].tier == 0:
                    w[k] *= 8.0
            total = float(w.sum())
            if not np.isfinite(total) or total <= 0.0:
                choice = candidates[rng.randrange(len(candidates))]
            elif rng.random() < q0:
                choice = candidates[int(np.argmax(w))]          # exploitation
            else:
                choice = candidates[int(np_rng.choice(len(candidates), p=w / total))]
            tour.append(choice)
            unvisited.discard(choice)
            current = choice + 1
        return tour

    iteration = 0
    while (iterations is None or iteration < iterations) \
            and time.perf_counter() < build_until:
        iteration += 1
        iteration_best_score = math.inf
        iteration_best: List[int] = []
        for _ in range(n_ants):
            if time.perf_counter() > build_until:
                break
            tour = build_tour()
            orders, unserved = split_giant_tour([task_list[i] for i in tour],
                                                vehicles, travel, weights,
                                                prize_collecting=prize_collecting)
            score = _objective_of(orders, unserved, vehicles, travel, weights)
            if score < iteration_best_score:
                iteration_best_score, iteration_best = score, tour
        if not iteration_best:
            break
        if iteration_best_score < best_score:
            best_score, best_tour = iteration_best_score, iteration_best

        # --- evaporate, then let only the iteration best deposit ---------
        tau *= (1.0 - rho)
        deposit = 1.0 / max(iteration_best_score, 1e-6)
        prev = 0
        for j in iteration_best:
            tau[prev, j] += deposit
            prev = j + 1
        np.clip(tau, tau_min, tau_max, out=tau)

    if not best_tour:
        best_tour = list(range(n))
    orders, unserved = split_giant_tour([task_list[i] for i in best_tour],
                                        vehicles, travel, weights,
                                        prize_collecting=prize_collecting)
    if polish_fraction > 0.0:
        orders, unserved = _polish(orders, unserved, vehicles, travel, weights, deadline)
    plan = _finalise(orders, unserved, vehicles, travel, weights, tasks,
                     "aco", (time.perf_counter() - started) * 1000.0)
    plan.metrics["iterations"] = iteration
    plan.metrics["ants"] = n_ants
    return plan


# ---------------------------------------------------------------------------
# Risk-penalised graph baseline
# ---------------------------------------------------------------------------
def risk_penalised_graph(tasks: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
                         travel: TravelModel, weights: Optional[ObjectiveWeights] = None,
                         alpha: float = 1.0, refine: bool = False) -> FleetPlan:
    """
    Greedy walk on a priority-warped graph, which is the direct analogue of the
    original manuscript's own method.

    Edge cost is ``distance / (1 + alpha * 10 * priority(destination))``, so
    urgent bins appear nearer.  No capacity, window or shift reasoning enters the
    *search*; feasibility is enforced only when the tour is decoded, which is
    exactly the limitation the fleet model is meant to remove.  The decoder is
    left without the selection test for the same reason: this rule visits what
    fits, and that is the behaviour it is here to represent.
    """
    weights = weights or ObjectiveWeights()
    started = time.perf_counter()
    task_list = list(tasks)
    if not task_list:
        return _finalise({}, [], vehicles, travel, weights, tasks, "risk_graph", 0.0)

    depot = vehicles[0].depot_index
    remaining = list(task_list)
    tour: List[BinTask] = []
    current = depot
    while remaining:
        def warped(t: BinTask) -> float:
            d = travel.distance(current, t.index)
            boost = 1.0 + alpha * 10.0 * max(0.0, t.prize)
            if t.tier == 0:
                boost *= 3.0
            return d / boost
        nxt = min(remaining, key=warped)
        tour.append(nxt)
        remaining.remove(nxt)
        current = nxt.index

    orders, unserved = split_giant_tour(tour, vehicles, travel, weights)
    if refine:
        orders, unserved = vrp.local_search(orders, vehicles, travel, weights,
                                            unserved, time_budget_s=3.0)
    label = "risk_graph_ls" if refine else "risk_graph"
    return _finalise(orders, unserved, vehicles, travel, weights, tasks,
                     label, (time.perf_counter() - started) * 1000.0)


# ---------------------------------------------------------------------------
# OR-Tools reference (optional dependency)
# ---------------------------------------------------------------------------
def ortools_available() -> bool:
    try:
        import ortools.constraint_solver.pywrapcp  # noqa: F401
        return True
    except Exception:
        return False


#: Cost units are multiplied by this before rounding to the integers OR-Tools
#: needs, so one unit of the shared objective is a thousand of the solver's.
ORTOOLS_COST_SCALE = 1000.0
#: Time is carried in tenths of a minute.  Whole minutes lose up to half a
#: minute per leg, which over a thirty-stop round is a quarter of an hour of a
#: shift, and the plan then fails the exact simulator at the last stop.
ORTOOLS_TIME_SCALE = 10


def ortools_solver(tasks: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
                   travel: TravelModel, weights: Optional[ObjectiveWeights] = None,
                   time_budget_s: float = 10.0,
                   first_solution: str = "PATH_CHEAPEST_ARC",
                   metaheuristic: str = "GUIDED_LOCAL_SEARCH",
                   emission_model: str = "mean_load") -> Optional[FleetPlan]:
    """
    Reference solution from OR-Tools' routing library, when it is installed.

    The model is the shared objective, stated in the solver's own terms as far
    as those terms reach.

    *Skipping.*  Each bin is optional, at the penalty `vrp.skip_cost` charges.

    *Licences.*  A bin may only be assigned to a vehicle that accepts its
    stream.

    *Time.*  Travel, service, waiting for a window and the tipping stop at the
    end of the round all count against the shift, and the whole duration of a
    round is charged at the objective's hourly rate through a span cost.  Travel
    times are rounded up, so a round the solver believes fits does fit.

    *Overflow deadlines.*  The objective charges a fixed penalty for reaching a
    bin after it overflows.  A bin that can be reached either side of its
    deadline is given two copies, one that must be served by the deadline and
    one that carries the penalty, of which at most one is visited.  That
    represents the step exactly.

    *Emissions.*  The solver needs arc costs fixed before the search, and
    emissions depend on the load carried, which is not known until a route
    exists.  ``emission_model="mean_load"`` costs each arc with the full model
    at the average load a vehicle carries, at that arc's own speed and
    congestion, and adds the idling and lifting at the bin it leads to.
    ``"flat"`` uses one factor per kilometre, which is the earlier behaviour and
    is kept so the effect of the approximation can be measured.

    Two things remain outside the model.  A vehicle may tip mid-shift and
    continue in the simulator; here each vehicle makes one trip, so the solver
    cannot use a second.  And the decoded routes are re-scored with the exact
    evaluator, so the reported numbers are never the solver's own estimate.

    Returns ``None`` when OR-Tools is unavailable.
    """
    if not ortools_available():
        return None
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    weights = weights or ObjectiveWeights()
    started = time.perf_counter()
    task_list = list(tasks)
    if not task_list:
        return _finalise({}, [], vehicles, travel, weights, tasks, "ortools", 0.0)
    if emission_model not in ("mean_load", "flat"):
        raise ValueError(f"unknown emission model {emission_model!r}")

    cost_scale = ORTOOLS_COST_SCALE
    time_scale = ORTOOLS_TIME_SCALE
    depot_index = vehicles[0].depot_index
    n_vehicles = len(vehicles)
    horizon_min = float(max(v.shift_minutes for v in vehicles))
    tipping_min = float(max(v.tipping_minutes for v in vehicles))

    def ticks_up(minutes: float) -> int:
        return int(math.ceil(minutes * time_scale - 1e-9))

    # --- nodes: the depot, then one or two copies of each bin ---------------
    node_task: List[Optional[BinTask]] = [None]
    node_late: List[bool] = [False]
    node_window: List[Tuple[int, int]] = [(0, ticks_up(horizon_min))]
    groups: List[List[int]] = []
    excluded: List[BinTask] = []
    for task in task_list:
        opens = int(math.ceil(task.window_start_min * time_scale - 1e-9))
        closes = int(math.floor(min(task.window_end_min, horizon_min) * time_scale + 1e-9))
        if opens > closes or not any(v.accepts(task.stream) for v in vehicles):
            # Its window opens after every shift ends, or no vehicle is licensed
            # for it.  No plan can serve it, so it never enters the model.
            excluded.append(task)
            continue
        tto = task.time_to_overflow_h
        deadline = int(math.floor(tto * 60.0 * time_scale + 1e-9)) if math.isfinite(tto) else None
        if deadline is None or deadline >= closes:
            copies = [(opens, closes, False)]              # cannot be reached late
        elif deadline < opens:
            copies = [(opens, closes, True)]               # late whenever it is served
        else:
            copies = [(opens, deadline, False), (deadline + 1, closes, True)]
        group = []
        for lo, hi, late in copies:
            group.append(len(node_task))
            node_task.append(task)
            node_late.append(late)
            node_window.append((lo, hi))
        groups.append(group)

    n_nodes = len(node_task)
    if n_nodes == 1:
        return _finalise({}, list(task_list), vehicles, travel, weights, tasks,
                         "ortools", (time.perf_counter() - started) * 1000.0)

    def matrix_index(node: int) -> int:
        return depot_index if node == 0 else node_task[node].index

    # --- arc costs and times, as matrices so the search makes no Python calls
    profile = vehicles[0].profile
    total_load = sum(max(0.0, vrp._finite(t.load_kg)) for t in task_list)
    mean_payload = 0.5 * min(float(min(v.capacity_kg for v in vehicles)),
                             total_load / max(1, n_vehicles))
    flat_co2_per_km = EM_NOMINAL_CO2_PER_KM(profile)

    cost_matrix = [[0] * n_nodes for _ in range(n_nodes)]
    time_matrix = [[0] * n_nodes for _ in range(n_nodes)]
    for a in range(n_nodes):
        ia = matrix_index(a)
        service_a = 0.0 if a == 0 else float(node_task[a].service_minutes)
        for b in range(n_nodes):
            if a == b:
                continue
            ib = matrix_index(b)
            km = travel.distance(ia, ib) / 1000.0
            target = node_task[b]
            if emission_model == "flat":
                co2 = flat_co2_per_km * km
            elif target is None:
                co2 = travel.leg_co2(ia, ib, mean_payload, profile)
            else:
                co2 = travel.leg_co2(ia, ib, mean_payload, profile,
                                     idle_minutes=float(target.service_minutes),
                                     lifts=1, lifted_kg=vrp._finite(target.load_kg))
            cost = weights.distance_km * km + weights.co2_kg * co2
            if node_late[b]:
                cost += weights.missed_overflow_penalty
            cost_matrix[a][b] = int(round(cost * cost_scale))
            minutes = service_a + travel.minutes(ia, ib)
            if b == 0:
                minutes += tipping_min
            time_matrix[a][b] = ticks_up(minutes)

    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    transit = routing.RegisterTransitMatrix(cost_matrix)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    # --- capacity ------------------------------------------------------
    # Mass, in kilograms, not divided by the compaction ratio: compacting waste
    # reduces the volume it occupies, not its mass.
    demands = [0] + [int(round(vrp._finite(node_task[k].load_kg)))
                     for k in range(1, n_nodes)]
    demand = routing.RegisterUnaryTransitVector(demands)
    routing.AddDimensionWithVehicleCapacity(
        demand, 0, [int(v.capacity_kg) for v in vehicles], True, "Capacity")

    # Compacted volume, in litres so it stays integral for the solver.  Added as
    # a second dimension only when the fleet declares a body volume, since
    # otherwise there is nothing to bound.
    if any(float(v.body_volume_m3) > 0.0 for v in vehicles):
        ratio = max(float(vehicles[0].compaction_ratio), 1e-9)
        litres = [0] + [int(round(node_task[k].loose_volume_m3 * 1000.0 / ratio))
                        for k in range(1, n_nodes)]
        volume = routing.RegisterUnaryTransitVector(litres)
        routing.AddDimensionWithVehicleCapacity(
            volume, 0,
            [int(round(float(v.body_volume_m3) * 1000.0)) or 10 ** 9 for v in vehicles],
            True, "Volume")

    # --- time: windows, shift, and the hourly rate on the whole round ----
    time_transit = routing.RegisterTransitMatrix(time_matrix)
    horizon = ticks_up(horizon_min)
    routing.AddDimension(time_transit, horizon, horizon, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for k in range(1, n_nodes):
        lo, hi = node_window[k]
        time_dim.CumulVar(manager.NodeToIndex(k)).SetRange(lo, hi)
    for vi, vehicle in enumerate(vehicles):
        time_dim.CumulVar(routing.End(vi)).SetMax(
            int(math.floor(float(vehicle.shift_minutes) * time_scale + 1e-9)))
    time_dim.SetSpanCostCoefficientForAllVehicles(
        int(round(weights.hours * cost_scale / (60.0 * time_scale))))

    # --- stream licences -------------------------------------------------
    # Removed from the vehicle variable one at a time.  The routing model has a
    # call that takes the allowed list directly, but its Python binding rejects
    # a list in this release, and removing values is equivalent.
    for k in range(1, n_nodes):
        vehicle_var = routing.VehicleVar(manager.NodeToIndex(k))
        for vi, vehicle in enumerate(vehicles):
            if not vehicle.accepts(node_task[k].stream):
                vehicle_var.RemoveValue(vi)

    # --- optional visits with the prize penalty -------------------------
    # The penalty handed to the solver has to be the one the plan is scored
    # against, or the comparison is unfair in a way that flatters us.
    # `skip_cost` is the single definition, so it is what gets used here.
    for group in groups:
        penalty = vrp.skip_cost(node_task[group[0]], weights)
        routing.AddDisjunction([manager.NodeToIndex(k) for k in group],
                               int(round(penalty * cost_scale)), 1)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, first_solution)
    params.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic)
    remaining = max(0.5, float(time_budget_s) - (time.perf_counter() - started))
    params.time_limit.FromMilliseconds(int(remaining * 1000))

    solution = routing.SolveWithParameters(params)
    status = int(routing.status())
    build_and_solve_ms = (time.perf_counter() - started) * 1000.0
    if solution is None:
        # No plan at all is a result, not an absence: every bin is unserved and
        # the comparison keeps the row instead of silently dropping it.
        plan = _finalise({}, list(task_list), vehicles, travel, weights, tasks,
                         "ortools", build_and_solve_ms)
        plan.metrics.update({"ortools_status": status, "ortools_solved": False,
                             "decode_dropped": 0})
        return plan

    orders: Dict[int, List[BinTask]] = {v.vehicle_id: [] for v in vehicles}
    visited = set()
    for vi, vehicle in enumerate(vehicles):
        index = routing.Start(vi)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                task = node_task[node]
                orders[vehicle.vehicle_id].append(task)
                visited.add(id(task))
            index = solution.Value(routing.NextVar(index))
    unserved = [t for t in task_list if id(t) not in visited]

    plan = _finalise(orders, unserved, vehicles, travel, weights, tasks,
                     "ortools", (time.perf_counter() - started) * 1000.0)
    plan.metrics.update({
        "ortools_status": status,
        "ortools_solved": True,
        # Bins the solver routed and the exact simulator then refused.  Zero
        # means the model and the simulator agree about what is feasible.
        "decode_dropped": len(plan.unserved) - len(unserved),
        "ortools_objective": solution.ObjectiveValue() / cost_scale,
        "first_solution": first_solution,
        "metaheuristic": metaheuristic,
        "emission_model": emission_model,
    })
    return plan


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def available_solvers() -> Dict[str, bool]:
    return {
        "proposed": True,
        "genetic": True,
        "aco": True,
        "risk_graph": True,
        "ortools": ortools_available(),
    }


def _run(name: str, tasks: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
         travel: TravelModel, weights: Optional[ObjectiveWeights],
         seed: int, time_budget_s: float, config: Dict) -> Optional[FleetPlan]:
    if name == "genetic":
        return genetic_algorithm(tasks, vehicles, travel, weights, seed=seed,
                                 time_budget_s=time_budget_s, **config)
    if name == "aco":
        return ant_colony(tasks, vehicles, travel, weights, seed=seed,
                          time_budget_s=time_budget_s, **config)
    if name == "risk_graph":
        return risk_penalised_graph(tasks, vehicles, travel, weights, **config)
    if name == "risk_graph_ls":
        return risk_penalised_graph(tasks, vehicles, travel, weights, refine=True)
    if name == "ortools":
        return ortools_solver(tasks, vehicles, travel, weights,
                              time_budget_s=time_budget_s, **config)
    raise ValueError(f"unknown solver {name!r}")


def solve_with(name: str, tasks: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
               travel: TravelModel, weights: Optional[ObjectiveWeights] = None,
               seed: int = 42, time_budget_s: float = 8.0,
               config: Optional[Dict] = None,
               reserve: int = 0) -> Optional[FleetPlan]:
    """
    Dispatch by name; returns ``None`` only for an unavailable optional solver.

    ``config`` passes solver parameters through, so a tuned configuration is a
    dictionary that can be stored beside the results it produced.

    ``reserve`` puts the reservation rule around a solver that has none.  The
    heads are certified and marked before the solver runs, so it sees them as
    bins it cannot afford to skip, and any head it still leaves out is restored
    afterwards.  The wait bound is a property of the rule and not of the planner
    it sits on, and this is how that is shown rather than said.
    """
    name = name.lower()
    config = dict(config or {})
    weights = weights or ObjectiveWeights()
    if name in ("proposed", "regret2_ls", "cvrptw"):
        return vrp.solve(tasks, vehicles, travel, weights, improve=True,
                         time_budget_s=time_budget_s,
                         reserve_overdue=config.pop("reserve_overdue",
                                                    reserve if reserve > 0 else True),
                         **config)
    if reserve <= 0:
        return _run(name, tasks, vehicles, travel, weights, seed, time_budget_s, config)

    started = time.perf_counter()
    queue = vrp.overdue_queue(tasks)
    heads, certificate, unservable = vrp.certify_heads(queue, vehicles, travel,
                                                       weights, reserve)
    for head in heads:
        head.mandatory = True
    try:
        plan = _run(name, tasks, vehicles, travel, weights, seed,
                    max(0.2, time_budget_s - (time.perf_counter() - started)), config)
        if plan is None:
            return None
        orders: Dict[int, List[BinTask]] = {v.vehicle_id: [] for v in vehicles}
        for route in plan.routes:
            orders[route.vehicle.vehicle_id] = [s.task for s in route.stops]
        served_before = {t.node_id for order in orders.values() for t in order}
        missing = sum(1 for h in heads if h.node_id not in served_before)
        orders, unserved = vrp.restore_heads(heads, certificate, orders,
                                             list(plan.unserved), vehicles,
                                             travel, weights)
    finally:
        for head in heads:
            head.mandatory = False

    extra = {k: v for k, v in plan.metrics.items()
             if k not in vrp.summarise([], [], [], weights)}
    rebuilt = _finalise(orders, unserved, vehicles, travel, weights, tasks,
                        f"{plan.algorithm}+reserve",
                        (time.perf_counter() - started) * 1000.0)
    served_ids = {s.task.node_id for r in rebuilt.routes for s in r.stops}
    rebuilt.metrics.update(extra)
    rebuilt.metrics.update(vrp.reservation_report(queue, heads, unservable,
                                                  served_ids, reserve,
                                                  repaired=missing))
    return rebuilt
