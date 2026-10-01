"""
Scenario assembly: turn a fleet snapshot into a solvable routing problem.

Keeping this in one place means the Django service, the experiment suite and the
tests all build *identical* problem instances from the same inputs, so a number
reported in the paper is the number the deployed planner would produce.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import aging
from . import emissions as EM
from . import traffic as TR
from . import vrp
from .geo import Coord, dist_matrix

DEFAULT_DETOUR_FACTOR = 1.30   # street circuity vs great-circle in a dense grid


def build_travel(coords: Sequence[Coord], when: datetime,
                 provider: Optional[TR.TrafficProvider] = None,
                 detour_factor: float = DEFAULT_DETOUR_FACTOR,
                 default_speed_kmh: float = 20.0,
                 freeflow_kmh: float = TR.FREEFLOW_KMH,
                 use_traffic: bool = True,
                 road_network: Optional[object] = None) -> vrp.TravelModel:
    """
    Distance matrix plus a traffic-aware travel model.

    With ``road_network`` set to a :class:`wastebins_core.roadnet.RoadNetwork`,
    distances are shortest paths on the real drivable street graph and each leg
    carries its own uncongested speed, taken from the arcs that path uses.  The
    matrix is then asymmetric, because one-way streets are.

    Without it, distances are great-circle multiplied by ``detour_factor`` and
    every leg shares one free-flow constant.  That is the earlier behaviour and
    it is kept so the two can be compared on identical instances, which is what
    the distance-model ablation does.
    """
    if road_network is not None:
        matrix, freeflow_matrix, _snap = road_network.matrices(coords)
    else:
        matrix = dist_matrix(coords, detour_factor=detour_factor)
        freeflow_matrix = None
    ctx = None
    if use_traffic:
        provider = provider or TR.SyntheticTrafficProvider()
        ctx = TR.build_travel_context(coords, matrix, provider, when, freeflow_kmh,
                                      freeflow_matrix=freeflow_matrix)
    return vrp.TravelModel(matrix, travel_context=ctx,
                           default_speed_kmh=default_speed_kmh,
                           freeflow_kmh=freeflow_kmh)


def bin_load_kg(fill_fraction: float, capacity_liters: float = 1100.0,
                density_kg_per_m3: float = 220.0) -> float:
    """Mass of waste in a bin at a given fill fraction."""
    return max(0.0, float(fill_fraction)) * (capacity_liters / 1000.0) * density_kg_per_m3


def make_tasks(node_ids: Sequence[int],
               fills: Dict[int, float],
               prizes: Dict[int, float],
               *,
               index_of: Optional[Dict[int, int]] = None,
               hazards: Optional[Dict[int, bool]] = None,
               tiers: Optional[Dict[int, int]] = None,
               overdue_pressures: Optional[Dict[int, float]] = None,
               waits: Optional[Dict[int, float]] = None,
               overdue: Optional[Dict[int, bool]] = None,
               tto_hours: Optional[Dict[int, float]] = None,
               streams: Optional[Dict[int, str]] = None,
               capacities_l: Optional[Dict[int, float]] = None,
               densities: Optional[Dict[int, float]] = None,
               service_minutes: Optional[Dict[int, float]] = None,
               windows: Optional[Dict[int, Tuple[float, float]]] = None,
               ) -> List[vrp.BinTask]:
    """Assemble ``BinTask`` objects from per-node dictionaries."""
    index_of = index_of or {nid: i for i, nid in enumerate(node_ids)}
    hazards = hazards or {}
    tiers = tiers or {}
    overdue_pressures = overdue_pressures or {}
    waits = waits or {}
    overdue = overdue or {}
    tto_hours = tto_hours or {}
    streams = streams or {}
    capacities_l = capacities_l or {}
    densities = densities or {}
    service_minutes = service_minutes or {}
    windows = windows or {}

    tasks: List[vrp.BinTask] = []
    for nid in node_ids:
        fill = float(fills.get(nid, 0.0))
        win = windows.get(nid, (0.0, 1440.0))
        tasks.append(vrp.BinTask(
            node_id=nid,
            index=index_of[nid],
            prize=float(prizes.get(nid, 0.0)),
            load_kg=bin_load_kg(fill,
                                capacities_l.get(nid, 1100.0),
                                densities.get(nid, 220.0)),
            service_minutes=float(service_minutes.get(nid, 4.0)),
            window_start_min=float(win[0]),
            window_end_min=float(win[1]),
            stream=str(streams.get(nid, "general")),
            hazard=bool(hazards.get(nid, False)),
            tier=int(tiers.get(nid, aging.TIER_HAZARD if hazards.get(nid)
                               else aging.TIER_NORMAL)),
            overdue_pressure=float(overdue_pressures.get(nid, 0.0)),
            wait_hours=float(waits.get(nid, 0.0)),
            overdue=bool(overdue.get(nid, False)),
            time_to_overflow_h=float(tto_hours.get(nid, float("inf"))),
            density_kg_per_m3=float(densities.get(nid, 220.0)),
        ))
    return tasks


def make_fleet(n_vehicles: int = 2,
               depot_index: int = 0,
               capacity_kg: float = 6000.0,
               shift_minutes: float = 480.0,
               avg_speed_kmh: float = 20.0,
               euro_class: str = "euro4",
               kerb_mass_kg: float = 12000.0,
               body_volume_m3: float = 16.0,
               accepts_streams: Sequence[str] = ()) -> List[vrp.VehicleSpec]:
    """
    A homogeneous fleet; heterogeneous fleets are built directly.

    ``body_volume_m3`` defaults to 16 cubic metres, a common rear-loader body.
    With a compaction ratio of 2.5 that holds about 40 cubic metres of loose
    waste, which at 220 kilograms per cubic metre is 8800 kilograms. The mass
    rating of 6000 kilograms is therefore reached first, using about 10.9 of the
    16 cubic metres. Mass binds at these settings; volume binds below a density
    of 150 kilograms per cubic metre. Both are enforced because a fleet meets
    both regimes.
    """
    fleet = []
    for i in range(n_vehicles):
        fleet.append(vrp.VehicleSpec(
            vehicle_id=i + 1,
            name=f"Truck-{i + 1}",
            depot_index=depot_index,
            capacity_kg=capacity_kg,
            body_volume_m3=body_volume_m3,
            shift_minutes=shift_minutes,
            avg_speed_kmh=avg_speed_kmh,
            accepts_streams=tuple(accepts_streams),
            profile=EM.profile_from_vehicle(capacity_kg, kerb_mass_kg, euro_class,
                                            f"Truck-{i + 1}"),
        ))
    return fleet


def static_sweep_plan(tasks: Sequence[vrp.BinTask], vehicles: Sequence[vrp.VehicleSpec],
                      travel: vrp.TravelModel,
                      weights: Optional[vrp.ObjectiveWeights] = None,
                      start_node: Optional[int] = None) -> vrp.FleetPlan:
    """
    The traditional fixed-schedule baseline: a round of every container in the
    order given, regardless of fill.  Callers pass the angular order about the
    depot (`sweep_order`).  No fill, priority or waiting time is used.

    The round is built as in the classic sweep heuristic.  Containers are taken
    in order, and each is inserted where it adds least distance into the route
    of the vehicle working that part of the round.  A container that vehicle may
    not carry goes to the next vehicle licensed for its stream.  When the working
    vehicle has no time or capacity left for a container it may carry, the next
    vehicle takes over, so each route covers one arc of the round.

    ``start_node`` continues a round from one shift to the next.  The order is
    rotated to begin at that container.  A multi-cycle caller sets it to the
    container the plan reports as ``metrics["round_next"]``, the first one the
    previous shift did not reach, so the round goes on where it stopped instead
    of beginning again at the same place every shift.  Without it the round
    begins at the start of the order.
    """
    weights = weights or vrp.ObjectiveWeights()
    ordered = list(tasks)
    if start_node is not None:
        at = next((k for k, t in enumerate(ordered) if t.node_id == start_node), 0)
        ordered = ordered[at:] + ordered[:at]

    orders: List[List[vrp.BinTask]] = [[] for _ in vehicles]
    unserved: List[vrp.BinTask] = []
    working = 0
    for task in ordered:
        placed = False
        for k in range(working, len(vehicles)):
            vehicle = vehicles[k]
            if not vehicle.accepts(task.stream):
                continue
            extended = _cheapest_feasible_insertion(orders[k], task, vehicle, travel)
            if extended is not None:
                orders[k] = extended
                placed = True
                break
            if k == working:
                # No time or capacity left on the working vehicle: its arc ends.
                working += 1
        if not placed:
            unserved.append(task)

    routes = []
    for vehicle, order in zip(vehicles, orders):
        if not order:
            continue
        evaluated = vrp.evaluate_route(order, vehicle, travel)
        if evaluated is not None:
            routes.append(evaluated)

    metrics = vrp.summarise(routes, unserved, tasks, weights)
    if ordered:
        metrics["round_next"] = (unserved[0] if unserved else ordered[0]).node_id
    return vrp.FleetPlan(
        routes=routes,
        unserved=unserved,
        objective=vrp.plan_objective(routes, unserved, weights),
        metrics=metrics,
        algorithm="static_sweep",
    )


def _cheapest_feasible_insertion(order: List[vrp.BinTask], task: vrp.BinTask,
                                 vehicle: vrp.VehicleSpec,
                                 travel: vrp.TravelModel) -> Optional[List[vrp.BinTask]]:
    """``order`` with ``task`` at the feasible position of least distance, or None."""
    best, best_m = None, float("inf")
    for position in range(len(order) + 1):
        trial = order[:position] + [task] + order[position:]
        evaluated = vrp.evaluate_route(trial, vehicle, travel)
        if evaluated is not None and evaluated.distance_m < best_m:
            best, best_m = trial, evaluated.distance_m
    return best


def sweep_order(coords: Sequence[Coord], depot: Coord) -> List[int]:
    """Indices ordered by polar angle about the depot (classic sweep heuristic)."""
    import math
    angles = []
    for i, (lat, lng) in enumerate(coords):
        angles.append((math.atan2(lat - depot[0], lng - depot[1]), i))
    return [i for _, i in sorted(angles)]


def threshold_plan(tasks: Sequence[vrp.BinTask], vehicles: Sequence[vrp.VehicleSpec],
                   travel: vrp.TravelModel, fill_threshold: float = 0.70,
                   fills: Optional[Dict[int, float]] = None,
                   weights: Optional[vrp.ObjectiveWeights] = None,
                   time_budget_s: float = 6.0,
                   improve_budget_s: Optional[float] = None) -> vrp.FleetPlan:
    """
    The common commercial baseline: collect only bins above a fill threshold,
    routed by the same construction+improvement machinery.  This isolates the
    benefit of *predictive* prioritisation from the benefit of simply not
    visiting empty bins.

    No queue head is reserved.  A threshold policy is defined by ignoring
    everything below the threshold, and reserving the longest-waiting bin among
    the ones it kept would give it part of the method it is compared against.
    """
    fills = fills or {}
    selected = [t for t in tasks
                if fills.get(t.node_id, t.load_kg / max(t.load_kg, 1e-9)) >= fill_threshold
                or t.hazard]
    skipped = [t for t in tasks if t not in selected]
    plan = vrp.solve(selected, vehicles, travel, weights, improve=True,
                     time_budget_s=time_budget_s,
                     improve_budget_s=improve_budget_s,
                     reserve_overdue=0,
                     algorithm=f"threshold_{fill_threshold:g}")
    plan.unserved = list(plan.unserved) + skipped
    weights = weights or vrp.ObjectiveWeights()
    plan.metrics = vrp.summarise(plan.routes, plan.unserved, tasks, weights)
    plan.objective = vrp.plan_objective(plan.routes, plan.unserved, weights)
    return plan
