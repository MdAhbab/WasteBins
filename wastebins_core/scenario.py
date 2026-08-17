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
                 use_traffic: bool = True) -> vrp.TravelModel:
    """Distance matrix plus a traffic-aware travel model."""
    matrix = dist_matrix(coords, detour_factor=detour_factor)
    ctx = None
    if use_traffic:
        provider = provider or TR.SyntheticTrafficProvider()
        ctx = TR.build_travel_context(coords, matrix, provider, when, freeflow_kmh)
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
                      weights: Optional[vrp.ObjectiveWeights] = None) -> vrp.FleetPlan:
    """
    The traditional fixed-schedule baseline: visit *every* bin regardless of
    fill, partitioned across the fleet by a geographic sweep about the depot and
    ordered by nearest neighbour.  No priority information is used at all.
    """
    weights = weights or vrp.ObjectiveWeights()
    depot = vehicles[0].depot_index
    matrix = travel.distance_m

    # Sweep: sort by polar angle about the depot, then split into equal arcs.
    def angle(t: vrp.BinTask) -> float:
        # Use matrix-free ordering when coordinates are unavailable: fall back to
        # index order.  Callers that want a true sweep pass coordinates via
        # ``sweep_angles``.
        return float(t.index)

    ordered = sorted(tasks, key=angle)
    per_vehicle = max(1, int(np.ceil(len(ordered) / max(1, len(vehicles)))))
    orders: Dict[int, List[vrp.BinTask]] = {}
    unserved: List[vrp.BinTask] = []

    for vi, vehicle in enumerate(vehicles):
        chunk = ordered[vi * per_vehicle:(vi + 1) * per_vehicle]
        # nearest-neighbour ordering on real distance
        remaining = list(chunk)
        seq: List[vrp.BinTask] = []
        cur = depot
        while remaining:
            nxt = min(remaining, key=lambda t: matrix[cur][t.index])
            seq.append(nxt)
            remaining.remove(nxt)
            cur = nxt.index
        # Trim to the longest feasible prefix; the rest simply cannot be done.
        feasible: List[vrp.BinTask] = []
        for task in seq:
            trial = feasible + [task]
            if vrp.evaluate_route(trial, vehicle, travel) is not None:
                feasible = trial
            else:
                unserved.append(task)
        orders[vehicle.vehicle_id] = feasible

    leftovers = ordered[len(vehicles) * per_vehicle:]
    unserved.extend(leftovers)

    routes = []
    for vehicle in vehicles:
        order = orders.get(vehicle.vehicle_id) or []
        if not order:
            continue
        evaluated = vrp.evaluate_route(order, vehicle, travel)
        if evaluated is not None:
            routes.append(evaluated)

    return vrp.FleetPlan(
        routes=routes,
        unserved=unserved,
        objective=vrp.plan_objective(routes, unserved, weights),
        metrics=vrp.summarise(routes, unserved, tasks, weights),
        algorithm="static_sweep",
    )


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
                   weights: Optional[vrp.ObjectiveWeights] = None) -> vrp.FleetPlan:
    """
    The common commercial baseline: collect only bins above a fill threshold,
    routed by the same construction+improvement machinery.  This isolates the
    benefit of *predictive* prioritisation from the benefit of simply not
    visiting empty bins.
    """
    fills = fills or {}
    selected = [t for t in tasks
                if fills.get(t.node_id, t.load_kg / max(t.load_kg, 1e-9)) >= fill_threshold
                or t.hazard]
    skipped = [t for t in tasks if t not in selected]
    plan = vrp.solve(selected, vehicles, travel, weights, improve=True,
                     algorithm=f"threshold_{fill_threshold:g}")
    plan.unserved = list(plan.unserved) + skipped
    weights = weights or vrp.ObjectiveWeights()
    plan.metrics = vrp.summarise(plan.routes, plan.unserved, tasks, weights)
    plan.objective = vrp.plan_objective(plan.routes, plan.unserved, weights)
    return plan
