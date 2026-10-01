"""
Two controlled experiments on the wait guarantee, kept apart on purpose.
========================================================================

The method has two mechanisms, and each makes a different claim.

**The price.**  Skipping an overdue container costs ``lambda * mu * w / (2 tau)``,
which grows without limit.  A planner with no reservation therefore takes a
container whose dedicated round trip costs ``C`` once the wait passes
``2 tau C / (lambda mu)``.  ``price_sweep`` measures that: one container, one
vehicle with room to spare, the container's distance swept, the wait advanced a
cycle at a time until the planner takes it.  The reservation is switched *off*
there, and has to be.  One overdue container is a queue of one, so with the
reservation on it is served the moment it is overdue at every distance, and the
sweep would say nothing about the price.  The same sweep is also run with the
reservation on, so that statement is a measurement rather than a remark.

**The reservation.**  The first ``r`` containers of the overdue queue are served
in every cycle.  That gives the bound

    W = Delta * ceil(tau / Delta) + (ceil(M / r) - 1) * Delta

for a backlog of at most ``M``.  ``tightness`` builds the instance on which the
bound is attained: ``n`` containers each on its own spoke from the depot, and a
shift that fits exactly ``r`` of them.  Every container starts at zero, so all
of them reach the deadline together, the backlog is the whole network, and the
last container of the first pass is served at exactly ``W``.

Both instances are synthetic and are described as such.  The price sweep is a
two-node distance matrix at a constant 20 km/h, not a place in Dhaka, which is
why a container can sit 100 km from its depot.

Out: results/wait_bound.json
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wastebins_core import aging as AG      # noqa: E402
from wastebins_core import scenario as SC    # noqa: E402
from wastebins_core import vrp as VRP        # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

DISTANCES_KM = (10.0, 20.0, 25.0, 30.0, 35.0, 40.0, 60.0, 100.0)
CYCLE_H = 12.0
MAX_WAIT_H = 4800.0          # 400 cycles, far past any predicted collection
SPEED_KMH = 20.0

# Waits at which the bounded penalty is evaluated.  A million hours is not a
# realistic wait; it is the point of the demonstration.  The bounded penalty
# converges to lambda_prize * overdue_multiplier from below and never reaches
# it, so a container costing more than that to insert is skipped at every wait,
# and no finite horizon can be searched to prove it.
SATURATION_WAITS_H = (48.0, 96.0, 240.0, 1e3, 1e4, 1e6)

# Star networks for the tightness instance: (containers, reserved heads).
TIGHTNESS_CASES = tuple((n, r) for n in (6, 12, 24) for r in (1, 2, 3, 4))
SPOKE_KM = 10.0


def one_task(wait_h: float, tau_h: float = AG.DEFAULT_TAU_H) -> VRP.BinTask:
    """
    Build the container through the production path, not by hand.

    Constructing a `BinTask` directly would let this experiment set the pricing
    and queue fields itself, which are exactly the fields under test.  Going
    through `effective_priorities` and `make_tasks` means the experiment measures
    the wiring the service uses rather than a convenient reconstruction of it.
    """
    tiered = AG.effective_priorities({1: 0.30}, {1: wait_h}, {1: 0.0}, tau_h=tau_h)
    return SC.make_tasks(
        [1], {1: 0.5}, {1: tiered[1].score}, index_of={1: 1},
        hazards={1: False},
        tiers={1: tiered[1].tier},
        overdue_pressures={1: tiered[1].pressure},
        waits={1: tiered[1].wait_hours},
        overdue={1: tiered[1].overdue},
        densities={1: 220.0}, capacities_l={1: 1100.0},
    )[0]


def travel_for(km: float) -> VRP.TravelModel:
    matrix = np.array([[0.0, km * 1000.0], [km * 1000.0, 0.0]])
    return VRP.TravelModel(matrix, default_speed_kmh=SPEED_KMH)


def round_trip_cost(km: float, weights: VRP.ObjectiveWeights,
                    tau_h: float) -> dict:
    """Cost of a dedicated round trip, with the terms it is made of."""
    vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0, shift_minutes=1200.0)
    evaluated = VRP.evaluate_route([one_task(tau_h, tau_h)], vehicle, travel_for(km))
    if evaluated is None:
        return {"total": math.inf}
    distance = weights.distance_km * evaluated.distance_m / 1000.0
    co2 = weights.co2_kg * evaluated.co2_kg
    hours = weights.hours * evaluated.duration_min / 60.0
    return {
        "total": float(VRP.route_cost(evaluated, weights)),
        "distance_term": float(distance),
        "co2_term": float(co2),
        "time_term": float(hours),
        "co2_kg": float(evaluated.co2_kg),
        "duration_min": float(evaluated.duration_min),
    }


def first_served(km: float, weights: VRP.ObjectiveWeights, tau_h: float,
                 reserve: int):
    """First wait, in whole cycles from the deadline, at which the planner serves."""
    travel = travel_for(km)
    vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0, shift_minutes=1200.0)
    wait = tau_h
    while wait <= MAX_WAIT_H:
        plan = VRP.solve([one_task(wait, tau_h)], [vehicle], travel, weights,
                         time_budget_s=0.4, reserve_overdue=reserve)
        if any(route.stops for route in plan.routes):
            return wait
        wait += CYCLE_H
    return None


def price_sweep(weights: VRP.ObjectiveWeights,
                tau_h: float = AG.DEFAULT_TAU_H) -> list:
    rows = []
    for km in DISTANCES_KM:
        cost = round_trip_cost(km, weights, tau_h)
        threshold = AG.wait_to_outprice(cost["total"], tau_h, weights.lambda_prize,
                                        weights.overdue_multiplier)
        # First dispatch instant at which the container is overdue and its
        # penalty is at least the cost of the trip.
        predicted = CYCLE_H * math.ceil(max(tau_h, threshold) / CYCLE_H - 1e-9)
        priced = first_served(km, weights, tau_h, reserve=0)
        reserved = first_served(km, weights, tau_h, reserve=1)
        rows.append({
            "distance_km": km,
            "round_trip_cost": round(cost["total"], 3),
            "cost_terms": {k: round(v, 4) for k, v in cost.items() if k != "total"},
            "price_threshold_h": round(threshold, 3),
            "predicted_h": round(predicted, 1),
            "served_at_h_price_only": priced,
            "served_at_h_reserved": reserved,
            "price_prediction_exact": priced is not None
            and abs(priced - predicted) < 1e-9,
        })
    return rows


def bounded_penalty_saturation(weights: VRP.ObjectiveWeights,
                               tau_h: float = AG.DEFAULT_TAU_H) -> dict:
    """
    Show that the bounded penalty has a ceiling, and where it bites.

    The bounded form prices skipping on the ranking score ``w/(tau + w)``, which
    never exceeds 1, so the penalty never exceeds
    ``lambda_prize * overdue_multiplier``.  This evaluates both penalties at
    waits up to a million hours and reports the first distance at which a
    dedicated round trip costs more than the ceiling.
    """
    ceiling = weights.lambda_prize * weights.overdue_multiplier
    by_wait = []
    for wait in SATURATION_WAITS_H:
        score = AG.overdue_score(wait, tau_h)
        by_wait.append({
            "wait_h": wait,
            "ranking_score": round(float(score), 9),
            "bounded_penalty": round(float(weights.lambda_prize * score
                                           * weights.overdue_multiplier), 6),
            "unbounded_penalty": round(float(weights.lambda_prize
                                             * AG.overdue_pressure(wait, tau_h)
                                             * weights.overdue_multiplier), 6),
        })

    first_unaffordable = None
    for km in [float(k) for k in range(5, 101)]:
        if round_trip_cost(km, weights, tau_h)["total"] > ceiling:
            first_unaffordable = km
            break

    # Under the bounded penalty and no reservation the container beyond that
    # distance is declined at every wait searched.
    beyond = first_unaffordable + 2.0 if first_unaffordable else None
    never_served = None
    if beyond is not None:
        vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0, shift_minutes=1200.0)
        never_served = True
        for wait in SATURATION_WAITS_H:
            tiered = AG.effective_priorities({1: 0.30}, {1: wait}, {1: 0.0},
                                             tau_h=tau_h, escalating_price=False)
            task = SC.make_tasks(
                [1], {1: 0.5}, {1: tiered[1].score}, index_of={1: 1},
                hazards={1: False}, tiers={1: tiered[1].tier},
                overdue_pressures={1: tiered[1].pressure},
                waits={1: tiered[1].wait_hours}, overdue={1: tiered[1].overdue},
                densities={1: 220.0}, capacities_l={1: 1100.0})[0]
            plan = VRP.solve([task], [vehicle], travel_for(beyond), weights,
                             time_budget_s=0.4, reserve_overdue=0)
            if any(route.stops for route in plan.routes):
                never_served = False
    return {
        "ceiling": ceiling,
        "by_wait": by_wait,
        "first_unaffordable_km": first_unaffordable,
        "checked_km": beyond,
        "never_served_at_checked_km": never_served,
    }


def star_travel(n: int, spoke_km: float = SPOKE_KM) -> VRP.TravelModel:
    """A depot and ``n`` containers, each on its own spoke through the depot."""
    size = n + 1
    matrix = np.full((size, size), 2.0 * spoke_km * 1000.0)
    matrix[0, :] = matrix[:, 0] = spoke_km * 1000.0
    np.fill_diagonal(matrix, 0.0)
    return VRP.TravelModel(matrix, default_speed_kmh=SPEED_KMH)


def star_tasks(waits: dict, tau_h: float) -> list:
    ids = sorted(waits)
    tiered = AG.effective_priorities({i: 0.30 for i in ids}, waits,
                                     {i: 0.0 for i in ids}, tau_h=tau_h)
    return SC.make_tasks(
        ids, {i: 0.5 for i in ids}, {i: tiered[i].score for i in ids},
        index_of={i: i for i in ids}, hazards={i: False for i in ids},
        tiers={i: tiered[i].tier for i in ids},
        overdue_pressures={i: tiered[i].pressure for i in ids},
        waits={i: tiered[i].wait_hours for i in ids},
        overdue={i: tiered[i].overdue for i in ids},
        densities={i: 220.0 for i in ids},
        capacities_l={i: 1100.0 for i in ids})


def tightness(weights: VRP.ObjectiveWeights,
              tau_h: float = AG.DEFAULT_TAU_H) -> list:
    """
    Roll the policy forward on the star network and compare with the bound.

    A container served in a cycle restarts at one cycle, which is the time
    since the dispatch instant of the cycle that collected it.
    """
    leg_min = 60.0 * SPOKE_KM / SPEED_KMH
    rows = []
    for n, reserve in TIGHTNESS_CASES:
        # Room for exactly `reserve` spokes: the legs, the services, one tip,
        # and two minutes of slack that no further container can use.
        shift = 2.0 * reserve * leg_min + 4.0 * reserve + 15.0 + 2.0
        vehicle = VRP.VehicleSpec(vehicle_id=0, depot_index=0, shift_minutes=shift)
        travel = star_travel(n)
        waits = {i: 0.0 for i in range(1, n + 1)}
        cycles = math.ceil(tau_h / CYCLE_H) + 2 * math.ceil(n / reserve) + 4
        worst_served, worst_backlog, least_reserved = 0.0, 0, None
        all_heads_served = True
        for _cycle in range(cycles):
            tasks = star_tasks(waits, tau_h)
            backlog = len(VRP.overdue_queue(tasks))
            worst_backlog = max(worst_backlog, backlog)
            plan = VRP.solve(tasks, [vehicle], travel, weights,
                             time_budget_s=0.2, reserve_overdue=reserve)
            if backlog:
                reserved = plan.metrics["heads_reserved"]
                least_reserved = reserved if least_reserved is None \
                    else min(least_reserved, reserved)
                all_heads_served &= (plan.metrics["heads_served"] == reserved)
            served = {s.task.node_id for r in plan.routes for s in r.stops}
            for node_id in waits:
                if node_id in served:
                    worst_served = max(worst_served, waits[node_id])
                    waits[node_id] = CYCLE_H
                else:
                    waits[node_id] += CYCLE_H
        bound = AG.queue_wait_bound(tau_h, CYCLE_H, n, reserve)
        rows.append({
            "containers": n,
            "reserved_per_cycle": reserve,
            "shift_min": shift,
            "cycles": cycles,
            "max_backlog": worst_backlog,
            "min_reserved": least_reserved,
            "all_heads_served": bool(all_heads_served),
            "bound_h": bound,
            "worst_wait_at_service_h": worst_served,
            "attained": abs(worst_served - bound) < 1e-9,
            "holds": worst_served <= bound + 1e-9,
        })
    return rows


def main() -> None:
    weights = VRP.ObjectiveWeights()
    tau = AG.DEFAULT_TAU_H
    sweep = price_sweep(weights, tau)
    saturation = bounded_penalty_saturation(weights, tau)
    tight = tightness(weights, tau)

    print("Price sweep: one container, one vehicle, no reservation")
    print("=" * 76)
    print(f"{'km':>6}{'trip cost':>12}{'threshold h':>13}{'predicted h':>13}"
          f"{'served h':>10}{'reserved h':>12}")
    print("-" * 76)
    for r in sweep:
        priced = "never" if r["served_at_h_price_only"] is None \
            else f"{r['served_at_h_price_only']:.0f}"
        print(f"{r['distance_km']:>6.0f}{r['round_trip_cost']:>12.2f}"
              f"{r['price_threshold_h']:>13.1f}{r['predicted_h']:>13.0f}"
              f"{priced:>10}{r['served_at_h_reserved']:>12.0f}")
    n_exact = sum(1 for r in sweep if r["price_prediction_exact"])
    print("-" * 76)
    print(f"  price prediction exact at {n_exact} of {len(sweep)} distances; "
          f"with the reservation every container is served at "
          f"{sorted(set(r['served_at_h_reserved'] for r in sweep))} h")

    print()
    print(f"Bounded penalty: ceiling {saturation['ceiling']:.0f}, first exceeded "
          f"by a round trip at {saturation['first_unaffordable_km']} km")
    for r in saturation["by_wait"]:
        print(f"{r['wait_h']:>12.0f}{r['bounded_penalty']:>16.4f}"
              f"{r['unbounded_penalty']:>16.2f}")

    print()
    print("Tightness of the queueing bound on a star network")
    print("=" * 76)
    print(f"{'n':>4}{'r':>4}{'backlog':>9}{'bound h':>10}{'worst h':>10}{'verdict':>12}")
    for r in tight:
        verdict = "attained" if r["attained"] else ("holds" if r["holds"] else "BREACH")
        print(f"{r['containers']:>4}{r['reserved_per_cycle']:>4}{r['max_backlog']:>9}"
              f"{r['bound_h']:>10.0f}{r['worst_wait_at_service_h']:>10.0f}{verdict:>12}")

    payload = {
        "protocol": {
            "cycle_h": CYCLE_H,
            "tau_h": tau,
            "speed_kmh": SPEED_KMH,
            "max_wait_searched_h": MAX_WAIT_H,
            "weights": {
                "lambda_prize": weights.lambda_prize,
                "overdue_multiplier": weights.overdue_multiplier,
                "distance_km": weights.distance_km,
                "co2_kg": weights.co2_kg,
                "hours": weights.hours,
            },
            "price_sweep_instance": (
                "synthetic: depot and one container on a two-node symmetric "
                "distance matrix, constant 20 km/h, no congestion, 1100 L "
                "container half full at 220 kg per cubic metre, 4 min service, "
                "15 min tipping, no overflow deadline, shift 1200 min"),
            "tightness_instance": (
                f"synthetic: n containers each {SPOKE_KM:.0f} km from the depot on "
                f"its own spoke, travel between containers through the depot, "
                f"shift sized to fit exactly r spokes, all waits start at zero"),
        },
        "price_sweep": sweep,
        "bounded_penalty": saturation,
        "tightness": tight,
        "n_price_exact": n_exact,
        "n_price_cases": len(sweep),
        "n_tight": sum(1 for r in tight if r["attained"]),
        "n_tight_cases": len(tight),
        "n_breaches": sum(1 for r in tight if not r["holds"]),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "wait_bound.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {RESULTS / 'wait_bound.json'}")


if __name__ == "__main__":
    main()
