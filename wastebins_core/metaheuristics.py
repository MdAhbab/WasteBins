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
    The risk-penalised graph approach from the related-work section: edge weights
    are inflated by the destination's risk, and the tour is the greedy walk on
    that warped graph.  This is the closest published analogue of the original
    manuscript's own method, so it establishes whether the gain comes from the
    warping idea or from the fleet model built around it.

``ortools``
    Optional. Uses Google OR-Tools routing when it is installed, giving a
    strong reference point; the module reports its absence rather than silently
    skipping the comparison.

Every solver honours a wall-clock ``time_budget_s`` and a seed, and returns a
:class:`wastebins_core.vrp.FleetPlan` so downstream reporting is uniform.
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


# ---------------------------------------------------------------------------
# Shared decoding: giant tour -> feasible vehicle routes
# ---------------------------------------------------------------------------
def split_giant_tour(tour: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
                     travel: TravelModel, weights: ObjectiveWeights
                     ) -> Tuple[Dict[int, List[BinTask]], List[BinTask]]:
    """
    Decode a permutation into vehicle routes.

    Bins are assigned to the current vehicle while the route stays feasible; when
    the next bin cannot be appended, the vehicle is closed and the next one
    opened.  A bin that no vehicle can take is left unserved and pays the prize
    penalty, which is what makes the encoding prize-collecting rather than
    all-or-nothing.
    """
    orders: Dict[int, List[BinTask]] = {v.vehicle_id: [] for v in vehicles}
    unserved: List[BinTask] = []
    vehicle_iter = list(vehicles)
    cursor = 0

    for task in tour:
        placed = False
        # Try the active vehicle first, then any later one, then any earlier one.
        order_of_attempt = list(range(cursor, len(vehicle_iter))) + list(range(0, cursor))
        for vi in order_of_attempt:
            vehicle = vehicle_iter[vi]
            if not vehicle.accepts(task.stream):
                continue
            trial = orders[vehicle.vehicle_id] + [task]
            if vrp.evaluate_route(trial, vehicle, travel) is not None:
                orders[vehicle.vehicle_id] = trial
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


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
def genetic_algorithm(tasks: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
                      travel: TravelModel, weights: Optional[ObjectiveWeights] = None,
                      population_size: int = 40, generations: int = 120,
                      crossover_rate: float = 0.85, mutation_rate: float = 0.20,
                      elite: int = 4, tournament: int = 3,
                      seed: int = 42, time_budget_s: float = 8.0) -> FleetPlan:
    """Route-first cluster-second GA with order crossover."""
    weights = weights or ObjectiveWeights()
    rng = random.Random(seed)
    started = time.perf_counter()
    deadline = started + max(0.2, time_budget_s)
    task_list = list(tasks)
    n = len(task_list)
    if n == 0:
        return _finalise({}, [], vehicles, travel, weights, tasks, "genetic", 0.0)

    def fitness(chromosome: List[int]) -> float:
        orders, unserved = split_giant_tour([task_list[i] for i in chromosome],
                                            vehicles, travel, weights)
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
    while generation < generations and time.perf_counter() < deadline:
        generation += 1
        next_pop = [c[:] for _, c in scored[:elite]]
        while len(next_pop) < population_size:
            p1, p2 = select(), select()
            child = order_crossover(p1, p2) if rng.random() < crossover_rate else p1[:]
            if rng.random() < mutation_rate:
                child = mutate(child)
            next_pop.append(child)
            if time.perf_counter() > deadline:
                break
        scored = [(fitness(c), c) for c in next_pop]
        scored.sort(key=lambda s: s[0])
        if scored[0][0] < best_score:
            best_score, best = scored[0][0], scored[0][1][:]

    orders, unserved = split_giant_tour([task_list[i] for i in best],
                                        vehicles, travel, weights)
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
               n_ants: int = 16, iterations: int = 60,
               alpha: float = 1.0, beta: float = 2.5, rho: float = 0.10,
               q0: float = 0.25, seed: int = 42,
               time_budget_s: float = 8.0) -> FleetPlan:
    """
    Max-Min Ant System over the bin graph.

    Pheromone lives on ordered pairs of bins (plus a depot-start row).  The
    heuristic is prize per metre, so ants are drawn towards urgent bins that are
    also cheap to reach.  Only the iteration-best ant reinforces, and pheromone
    is clamped, which is what distinguishes MMAS from classic AS and keeps it
    from collapsing onto one tour within a handful of iterations.
    """
    weights = weights or ObjectiveWeights()
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    started = time.perf_counter()
    deadline = started + max(0.2, time_budget_s)

    task_list = list(tasks)
    n = len(task_list)
    if n == 0:
        return _finalise({}, [], vehicles, travel, weights, tasks, "aco", 0.0)

    depot = vehicles[0].depot_index
    idx = [t.index for t in task_list]

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
    while iteration < iterations and time.perf_counter() < deadline:
        iteration += 1
        iteration_best_score = math.inf
        iteration_best: List[int] = []
        for _ in range(n_ants):
            if time.perf_counter() > deadline:
                break
            tour = build_tour()
            orders, unserved = split_giant_tour([task_list[i] for i in tour],
                                                vehicles, travel, weights)
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
                                        vehicles, travel, weights)
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
    Greedy walk on a priority-warped graph -- the published risk-penalised
    approach, and the direct analogue of the original manuscript's own method.

    Edge cost is ``distance / (1 + alpha * 10 * priority(destination))``, so
    urgent bins appear nearer.  No capacity, window or shift reasoning enters the
    *search*; feasibility is enforced only when the tour is decoded, which is
    exactly the limitation the fleet model is meant to remove.
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


def ortools_solver(tasks: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
                   travel: TravelModel, weights: Optional[ObjectiveWeights] = None,
                   time_budget_s: float = 10.0) -> Optional[FleetPlan]:
    """
    Reference solution from OR-Tools' routing library, when it is installed.

    Models capacity, time windows, shift horizon and optional visits with the
    same penalties as :class:`ObjectiveWeights`, so the objective is comparable.
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

    depot_local = 0
    locals_to_matrix = [vehicles[0].depot_index] + [t.index for t in task_list]
    n_nodes = len(locals_to_matrix)
    n_vehicles = len(vehicles)

    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, depot_local)
    routing = pywrapcp.RoutingModel(manager)

    # OR-Tools needs integers, and every term must live in the SAME currency as
    # ObjectiveWeights -- otherwise arc costs and drop penalties are on different
    # scales and the solver either drops everything or visits everything.
    # We therefore express costs in milli-units of the shared objective.
    SCALE = 1000.0
    # CO2 is not separately representable as an arc cost, so it enters through a
    # nominal per-km factor at half payload.  The decoded routes are re-scored
    # afterwards with the exact evaluator, so the reported numbers stay honest.
    nominal_profile = vehicles[0].profile
    nominal_co2_per_km = EM_NOMINAL_CO2_PER_KM(nominal_profile)

    def arc_cost(from_index, to_index):
        i = locals_to_matrix[manager.IndexToNode(from_index)]
        j = locals_to_matrix[manager.IndexToNode(to_index)]
        km = travel.distance(i, j) / 1000.0
        hours = travel.minutes(i, j) / 60.0
        cost = (weights.distance_km * km
                + weights.co2_kg * nominal_co2_per_km * km
                + weights.hours * hours)
        return int(round(cost * SCALE))

    transit = routing.RegisterTransitCallback(arc_cost)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    # --- capacity ------------------------------------------------------
    # Mass, in kilograms, not divided by the compaction ratio: compacting waste
    # reduces the volume it occupies, not its mass.
    def demand_cb(from_index):
        node = manager.IndexToNode(from_index)
        if node == depot_local:
            return 0
        return int(round(task_list[node - 1].load_kg))

    demand = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand, 0, [int(v.capacity_kg) for v in vehicles], True, "Capacity")

    # Compacted volume, in litres so it stays integral for the solver.  Added as
    # a second dimension only when the fleet declares a body volume, since
    # otherwise there is nothing to bound.
    if any(float(v.body_volume_m3) > 0.0 for v in vehicles):
        ratio = max(float(vehicles[0].compaction_ratio), 1e-9)

        def volume_cb(from_index):
            node = manager.IndexToNode(from_index)
            if node == depot_local:
                return 0
            litres = task_list[node - 1].loose_volume_m3 * 1000.0 / ratio
            return int(round(litres))

        volume = routing.RegisterUnaryTransitCallback(volume_cb)
        routing.AddDimensionWithVehicleCapacity(
            volume, 0,
            [int(round(float(v.body_volume_m3) * 1000.0)) or 10 ** 9 for v in vehicles],
            True, "Volume")

    # --- time windows + shift ------------------------------------------
    def time_cb(from_index, to_index):
        i_node = manager.IndexToNode(from_index)
        j_node = manager.IndexToNode(to_index)
        i = locals_to_matrix[i_node]
        j = locals_to_matrix[j_node]
        service = 0.0 if i_node == depot_local else task_list[i_node - 1].service_minutes
        return int(round(travel.minutes(i, j) + service))

    time_transit = routing.RegisterTransitCallback(time_cb)
    horizon = int(max(v.shift_minutes for v in vehicles))
    routing.AddDimension(time_transit, horizon, horizon, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for k, t in enumerate(task_list, start=1):
        index = manager.NodeToIndex(k)
        time_dim.CumulVar(index).SetRange(int(t.window_start_min),
                                          int(min(t.window_end_min, horizon)))

    # --- optional visits with the prize penalty -------------------------
    for k, t in enumerate(task_list, start=1):
        penalty = weights.lambda_prize * max(0.0, t.prize)
        if t.hazard or t.tier == 0:
            penalty *= weights.hazard_multiplier
        if math.isfinite(t.time_to_overflow_h):
            penalty += weights.missed_overflow_penalty
        routing.AddDisjunction([manager.NodeToIndex(k)], int(round(penalty * SCALE)))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(int(max(1, time_budget_s)))

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return None

    orders: Dict[int, List[BinTask]] = {v.vehicle_id: [] for v in vehicles}
    visited = set()
    for vi, vehicle in enumerate(vehicles):
        index = routing.Start(vi)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != depot_local:
                orders[vehicle.vehicle_id].append(task_list[node - 1])
                visited.add(node - 1)
            index = solution.Value(routing.NextVar(index))
    unserved = [t for k, t in enumerate(task_list) if k not in visited]

    return _finalise(orders, unserved, vehicles, travel, weights, tasks,
                     "ortools", (time.perf_counter() - started) * 1000.0)


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


def solve_with(name: str, tasks: Sequence[BinTask], vehicles: Sequence[VehicleSpec],
               travel: TravelModel, weights: Optional[ObjectiveWeights] = None,
               seed: int = 42, time_budget_s: float = 8.0) -> Optional[FleetPlan]:
    """Dispatch by name; returns ``None`` only for an unavailable optional solver."""
    name = name.lower()
    if name in ("proposed", "regret2_ls", "cvrptw"):
        return vrp.solve(tasks, vehicles, travel, weights, improve=True,
                         time_budget_s=time_budget_s)
    if name == "genetic":
        return genetic_algorithm(tasks, vehicles, travel, weights, seed=seed,
                                 time_budget_s=time_budget_s)
    if name == "aco":
        return ant_colony(tasks, vehicles, travel, weights, seed=seed,
                          time_budget_s=time_budget_s)
    if name == "risk_graph":
        return risk_penalised_graph(tasks, vehicles, travel, weights)
    if name == "risk_graph_ls":
        return risk_penalised_graph(tasks, vehicles, travel, weights, refine=True)
    if name == "ortools":
        return ortools_solver(tasks, vehicles, travel, weights,
                              time_budget_s=time_budget_s)
    raise ValueError(f"unknown solver {name!r}")
