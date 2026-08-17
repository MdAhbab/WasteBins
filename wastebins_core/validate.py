"""
Independent feasibility checking for a routing plan.

The point of this module is that it does **not** reuse ``vrp.evaluate_route``.
It re-derives the clock, the on-board load, the tipping trips and the time
windows from the raw travel matrix, so a bug in the evaluator cannot hide behind
the evaluator's own accounting.  A planner that scores its own output is only
ever self-consistent; the claim that a plan is feasible has to come from
somewhere else.

Two things live here:

* :func:`plan_violations` -- the checker, returning a list of human-readable
  violations (empty means feasible).
* :func:`random_instance` / :func:`build_instance` -- instance generators, so
  tests and experiments can build problems without a database.

Both are used by the test suite and by the experiment scripts; keeping them in
the core package rather than in a throwaway script is deliberate, because an
audit that cannot be re-run is not an audit.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import vrp

TOLERANCE_MIN = 1e-6
TOLERANCE_KG = 1e-6
# The re-simulated duration is compared against the reported one.  Half a minute
# is loose enough for float accumulation over a long route and tight enough that
# a genuine accounting error -- a missed tipping trip, a dropped service time --
# shows up immediately.
DURATION_TOLERANCE_MIN = 0.5


def plan_violations(plan, travel, tol: float = TOLERANCE_MIN) -> List[str]:
    """
    Re-simulate ``plan`` and return every constraint it breaks.

    Checks, per route: stream licensing, arrival against the window end,
    compacted load against capacity, total elapsed time against the shift, and
    the re-simulated duration against the reported one.  Across routes: that no
    bin is served twice, and that no bin is both served and listed unserved.
    """
    bad: List[str] = []

    for route in plan.routes:
        vehicle = route.vehicle
        clock = 0.0
        load = 0.0
        position = vehicle.depot_index
        trip = 0

        for stop in route.stops:
            task = stop.task

            if not vehicle.accepts(task.stream):
                bad.append(f"vehicle {vehicle.vehicle_id} stream licensing: "
                           f"{task.stream!r} not in {vehicle.accepts_streams}")

            # Capacity binds on compacted mass, which is what the constraint in
            # `evaluate_route` uses -- checking raw mass here would disagree with
            # the planner for reasons that are not bugs.
            effective = task.load_kg / max(vehicle.compaction_ratio, 1e-9)

            if stop.trip_index != trip:
                # A tipping trip: back to the depot, empty, then out again.
                clock += travel.minutes(position, vehicle.depot_index)
                clock += vehicle.tipping_minutes
                load = 0.0
                position = vehicle.depot_index
                trip = stop.trip_index

            clock += travel.minutes(position, task.index)
            arrival = clock
            if arrival > task.window_end_min + tol:
                bad.append(f"vehicle {vehicle.vehicle_id} node {task.node_id} late: "
                           f"arrives {arrival:.2f} > window end {task.window_end_min:.2f}")

            # Arriving early means waiting for the window to open.
            clock = max(arrival, task.window_start_min) + task.service_minutes

            load += effective
            if load > vehicle.capacity_kg + TOLERANCE_KG:
                bad.append(f"vehicle {vehicle.vehicle_id} node {task.node_id} capacity: "
                           f"{load:.1f} kg > {vehicle.capacity_kg:.1f} kg")

            position = task.index

        if route.stops:
            clock += travel.minutes(position, vehicle.depot_index)
            clock += vehicle.tipping_minutes

        if clock > vehicle.shift_minutes + tol:
            bad.append(f"vehicle {vehicle.vehicle_id} shift: "
                       f"{clock:.2f} min > {vehicle.shift_minutes:.2f} min")

        if abs(clock - route.duration_min) > DURATION_TOLERANCE_MIN:
            bad.append(f"vehicle {vehicle.vehicle_id} duration mismatch: "
                       f"re-simulated {clock:.2f} vs reported {route.duration_min:.2f}")

    served = [stop.task.node_id for route in plan.routes for stop in route.stops]
    if len(served) != len(set(served)):
        duplicates = sorted({n for n in served if served.count(n) > 1})
        bad.append(f"bins served more than once: {duplicates}")

    both = set(served) & {task.node_id for task in plan.unserved}
    if both:
        bad.append(f"bins both served and reported unserved: {sorted(both)}")

    return bad


def _euclidean_matrix(coords: Sequence[Tuple[float, float]]) -> np.ndarray:
    n = len(coords)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            matrix[i][j] = math.dist(coords[i], coords[j])
    return matrix


def random_instance(rng, n_bins: int = 24, n_veh: int = 3,
                    spread: float = 6000.0, cluster: bool = False):
    """
    A synthetic instance with the heterogeneity that makes routing hard.

    Mixed capacities and shift lengths, mixed waste streams, a mixture of bins
    with a real overflow deadline and bins with none, and optionally clustered
    geography -- which is the layout that punishes a search unable to open a
    second route.  Returns ``(matrix, task_dicts, vehicle_dicts)`` so callers can
    perturb the dictionaries before building the typed objects.
    """
    if cluster:
        centres = [(spread, 0.0), (-spread, spread * 0.6), (0.0, -spread)]
        points = []
        for i in range(n_bins):
            cx, cy = centres[i % len(centres)]
            points.append((cx + rng.normal(0, 350), cy + rng.normal(0, 350)))
    else:
        points = [(rng.uniform(-spread, spread), rng.uniform(-spread, spread))
                  for _ in range(n_bins)]

    coords = [(0.0, 0.0)] + points          # index 0 is the depot
    matrix = _euclidean_matrix(coords)

    tasks: List[Dict] = []
    for i in range(n_bins):
        fill = rng.uniform(0.05, 1.0)
        # Half the bins have no deadline at all, so the prize-collecting decision
        # is genuinely exercised rather than every bin being urgent.
        tto = math.inf if rng.random() < 0.5 else float(rng.uniform(2.0, 30.0))
        hazard = bool(rng.random() < 0.08)
        tasks.append(dict(
            node_id=i + 1,
            index=i + 1,
            prize=float(np.clip(fill + rng.normal(0, 0.1), 0.0, 1.0)),
            load_kg=fill * 242.0,
            service_minutes=float(rng.uniform(2.5, 6.0)),
            window_start_min=float(rng.choice([0, 60, 120, 180])),
            window_end_min=float(rng.choice([360, 420, 480])),
            stream=str(rng.choice(["general", "general", "organic", "recyclable"])),
            hazard=hazard,
            tier=0 if hazard else 1,
            time_to_overflow_h=tto,
        ))

    vehicles = [dict(vehicle_id=k + 1, depot_index=0,
                     capacity_kg=float(rng.choice([3000, 4500, 6000])),
                     shift_minutes=float(rng.choice([240, 360, 480])),
                     avg_speed_kmh=20.0)
                for k in range(n_veh)]
    return matrix, tasks, vehicles


def build_instance(matrix, tasks: Sequence[Dict], vehicles: Sequence[Dict]):
    """Turn the dictionaries from :func:`random_instance` into typed objects."""
    travel = vrp.TravelModel(matrix, default_speed_kmh=20.0)
    return (travel,
            [vrp.BinTask(**t) for t in tasks],
            [vrp.VehicleSpec(**v) for v in vehicles])
