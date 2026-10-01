"""
What the street graph says that a constant detour factor cannot.
================================================================

Routing studies in this area commonly measure distance as the great-circle
separation between two stops multiplied by one detour factor, typically between
1.2 and 1.4.  This script measures the error that approximation makes on the
exact instances the routing study uses, before any planner runs, so the finding
is a property of the two cities and not of a heuristic.

Per study area it reports the following.

``circuity``
    Road distance divided by great-circle distance.

``constant_factor_error``
    Signed error of the constant-factor distance against the shortest path,
    ``1.30 * great-circle / road - 1``, so a negative value means the constant
    understates the road distance.

``asymmetry``
    The relative gap between the two directions of travel between the same pair
    of stops.  A symmetric matrix cannot represent it, and it comes from one-way
    restrictions rather than from noise.

These three are measured over one population: ordered pairs of stops at least
100 m apart in a straight line.  Closer pairs are left out of all three, because
the ratio of a road distance to a few metres says nothing about the street
pattern.  An earlier version filtered the circuity and not the error, so the two
rows of one table described different pairs.

``short_pairs``
    The pairs left out, and the road distance actually needed between them.

``round_trips``
    For every container, the dedicated round trip from the depot and back: its
    length under both distance models, and what it costs in the units of the
    routing objective.  This is the quantity the pricing result is stated in.
    The longest single leg in an instance is a different thing, joins two
    containers and not the depot, and says nothing about it.

Run:  python -m experiments.exp_network
Out:  results/network.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import aging as AG              # noqa: E402
from wastebins_core import roadnet as RN            # noqa: E402
from wastebins_core import traffic as TR            # noqa: E402
from wastebins_core import vrp as VRP               # noqa: E402
from wastebins_core.geo import dist_matrix, haversine  # noqa: E402
from experiments import exp_fleet as EF             # noqa: E402
from experiments import instances as IN             # noqa: E402
from experiments import policies as PO              # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

#: The factor the earlier version of this work used, and the value most commonly
#: quoted for a dense urban grid.
CONSTANT_DETOUR = 1.30
MIN_STRAIGHT_M = 100.0


def _describe(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
        "min": float(finite.min()),
    }


def instance(study: str) -> Dict:
    """The one geometry each study area is routed on."""
    if study == "wyndham":
        return IN.wyndham_snapshots(1)[0]
    return IN.snapshots("S0", 1, 6)[0]


def pairs(study: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Circuity, constant-factor error and asymmetry over pairs at least 100 m apart.

    Also returns the matrices, for callers that need more than the three arrays.
    """
    snapshot = instance(study)
    coords = snapshot["coords"]
    net = RN.load(EF.STUDY["area"])
    road, freeflow, snapped = net.matrices(coords)
    approx = dist_matrix(coords, detour_factor=CONSTANT_DETOUR)
    n = len(coords)
    straight = np.array([[haversine(coords[i][0], coords[i][1],
                                    coords[j][0], coords[j][1])
                          for j in range(n)] for i in range(n)])
    off = ~np.eye(n, dtype=bool)
    kept = off & (straight >= MIN_STRAIGHT_M) & (road > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        circuity = (road / straight)[kept]
        error = ((approx - road) / road)[kept]
        asymmetry = (np.abs(road - road.T) / road)[kept]
    extra = {"snapshot": snapshot, "net": net, "road": road, "approx": approx,
             "freeflow": freeflow, "snapped": snapped, "straight": straight,
             "off": off, "kept": kept}
    return circuity, error, asymmetry, extra


def _round_trips(study: str, extra: Dict) -> Dict:
    """Dedicated depot round trips: length under both models, and objective cost."""
    snapshot, road, approx = extra["snapshot"], extra["road"], extra["approx"]
    n = len(snapshot["coords"])
    weights = VRP.ObjectiveWeights()
    fleet = EF.build_fleet(snapshot)

    road_km = np.array([(road[0][i] + road[i][0]) / 1000.0 for i in range(1, n)])
    out_km = np.array([road[0][i] / 1000.0 for i in range(1, n)])
    back_km = np.array([road[i][0] / 1000.0 for i in range(1, n)])
    approx_km = np.array([(approx[0][i] + approx[i][0]) / 1000.0 for i in range(1, n)])
    relative = (approx_km - road_km) / road_km
    worst = int(np.argmax(road_km))

    # A full container with no overflow deadline, so the cost depends on the
    # network and the fleet and not on the state of one snapshot.
    state = dict(snapshot)
    state["fills"] = {nid: 1.0 for nid in snapshot["node_ids"]}
    state["tto"] = {nid: float("inf") for nid in snapshot["node_ids"]}
    state["hazards"] = {nid: False for nid in snapshot["node_ids"]}
    tasks = PO.build_tasks(state, PO.RULES["full"])

    provider = TR.SyntheticTrafficProvider(seed=EF.SEED)

    def model(matrix, freeflow, speed_scale: float = 1.0) -> VRP.TravelModel:
        scaled = None if freeflow is None else freeflow * speed_scale
        ctx = TR.build_travel_context(snapshot["coords"], matrix, provider, IN.WHEN,
                                      TR.FREEFLOW_KMH * speed_scale,
                                      freeflow_matrix=scaled)
        return VRP.TravelModel(matrix, travel_context=ctx, default_speed_kmh=20.0,
                               freeflow_kmh=TR.FREEFLOW_KMH * speed_scale)

    def costs(travel: VRP.TravelModel) -> Tuple[np.ndarray, int]:
        values, unservable = [], 0
        for task in tasks:
            best = None
            for vehicle in fleet:
                if not vehicle.accepts(task.stream):
                    continue
                route = VRP.evaluate_route([task], vehicle, travel)
                if route is not None:
                    cost = VRP.route_cost(route, weights)
                    best = cost if best is None else min(best, cost)
            if best is None:
                unservable += 1
                values.append(np.nan)
            else:
                values.append(best)
        return np.array(values), unservable

    threshold = weights.lambda_prize * weights.overdue_multiplier / 2.0
    street, street_unservable = costs(model(road, extra["freeflow"]))
    constant, constant_unservable = costs(model(approx, None))
    slow, slow_unservable = costs(model(road, extra["freeflow"], speed_scale=0.5))
    cost_error = (constant - street) / street
    top = int(np.nanargmax(street))

    def price_wait(cost: float) -> float:
        return AG.wait_to_outprice(cost, AG.DEFAULT_TAU_H, weights.lambda_prize,
                                   weights.overdue_multiplier)

    return {
        "containers": int(n - 1),
        "road_km": _describe(road_km),
        "longest": {"road_km": float(road_km[worst]), "out_km": float(out_km[worst]),
                    "back_km": float(back_km[worst]),
                    "constant_km": float(approx_km[worst]),
                    "relative_error": float(relative[worst])},
        "constant_relative_error": _describe(relative),
        "share_understated": float(np.mean(relative < 0.0)),
        "cost": {
            "street": _describe(street),
            "constant": _describe(constant),
            "half_speed": _describe(slow),
            "relative_error_of_constant": _describe(cost_error),
            "largest": {"street": float(street[top]), "constant": float(constant[top]),
                        "relative_error": float(cost_error[top]),
                        "half_speed": float(slow[top]),
                        "price_wait_h": float(price_wait(street[top])),
                        "price_wait_constant_h": float(price_wait(constant[top])),
                        "price_wait_half_speed_h": float(price_wait(slow[top]))},
            "above_half_ceiling": {
                "threshold": float(threshold),
                "street": int(np.nansum(street > threshold)),
                "constant": int(np.nansum(constant > threshold)),
                "half_speed": int(np.nansum(slow > threshold))},
            "not_servable_alone": {"street": street_unservable,
                                   "constant": constant_unservable,
                                   "half_speed": slow_unservable},
            "container": "full, no overflow deadline",
        },
    }


def analyse(study: str) -> Dict:
    """Measure one study area on the geometry the routing study uses."""
    circuity, error, asymmetry, extra = pairs(study)
    net, road, approx = extra["net"], extra["road"], extra["approx"]
    off, kept, straight = extra["off"], extra["kept"], extra["straight"]
    # Every pair left out of the three measurements, including pairs that snap
    # to one graph node and so have a road distance of zero.
    short = off & (straight < MIN_STRAIGHT_M)

    # The single longest leg between any two stops.  Kept because the earlier
    # version quoted it, and reported with whether it touches the depot, because
    # it was read there as the leg that sets the insertion cost and it is not.
    k = np.unravel_index(int(np.argmax(np.where(off, road, 0.0))), road.shape)
    travel = EF.travel_for(extra["snapshot"], IN.WHEN)
    leg_speed = np.array([travel.speed_kmh(i, j) for i in range(len(road))
                          for j in range(len(road)) if i != j])
    return {
        "study": study,
        "area": EF.STUDY["area"],
        "n_stops": len(extra["snapshot"]["coords"]),
        "pair_filter": f"ordered pairs at least {MIN_STRAIGHT_M:.0f} m apart in a "
                       f"straight line",
        "pairs_total": int(off.sum()),
        "pairs_kept": int(kept.sum()),
        "graph": {
            "nodes": net.n_nodes,
            "arcs": net.n_arcs,
            "fingerprint": RN.fingerprint(net),
            "bbox": net.meta["bbox"],
            "fetched_utc": net.meta["fetched_utc"],
            "ways_parsed": net.meta["ways_parsed"],
            "strong_components_before_trim": net.meta["strong_components"],
            "tagged_maxspeed_share": net.meta["tagged_maxspeed_share"],
            "speed_scale": net.meta["speed_scale"],
            "oneway_arc_share": float(1.0 - _reciprocal_share(net)),
        },
        "snap_distance_m": {"max": float(extra["snapped"].max())},
        "circuity": _describe(circuity),
        "asymmetry": _describe(asymmetry),
        "freeflow_kmh": _describe(extra["freeflow"][off]),
        "leg_speed_kmh": _describe(leg_speed),
        "constant_factor_error": {
            "factor": CONSTANT_DETOUR,
            "relative": _describe(error),
            "share_underestimated": float(np.mean(error < 0.0)),
            "note": "negative relative error means the constant-factor distance is "
                    "shorter than the shortest path on the street graph",
        },
        "short_pairs_under_100m": {
            "pairs": int(short.sum()),
            "mean_straight_m": float(straight[short].mean()) if short.any() else None,
            "mean_road_m": float(road[short].mean()) if short.any() else None,
            "max_road_m": float(road[short].max()) if short.any() else None,
        },
        "longest_leg": {
            "road_km": float(road[k] / 1000.0),
            "constant_km": float(approx[k] / 1000.0),
            "relative_error": float((approx[k] - road[k]) / road[k]),
            "touches_depot": bool(0 in k),
        },
        "round_trips": _round_trips(study, extra),
        "best_fit_constant": float(np.mean(circuity)),
    }


def _reciprocal_share(net: RN.RoadNetwork) -> float:
    """Share of arcs whose reverse also exists, meaning the way is two-way."""
    arcs = set(zip(net.src.tolist(), net.dst.tolist()))
    reciprocal = sum(1 for a, b in arcs if (b, a) in arcs)
    return reciprocal / max(1, len(arcs))


def main() -> int:
    payload = {
        "constant_detour_factor": CONSTANT_DETOUR,
        "areas": {name: analyse(name) for name in ("dhaka", "wyndham")},
    }
    (RESULTS / "network.json").write_text(json.dumps(payload, indent=2))

    for name, row in payload["areas"].items():
        c, e = row["circuity"], row["constant_factor_error"]["relative"]
        print(f"\n{name}: {row['n_stops']} stops, {row['pairs_kept']} of "
              f"{row['pairs_total']} ordered pairs at least 100 m apart")
        print(f"  circuity            mean {c['mean']:.3f}  median {c['median']:.3f}  "
              f"p90 {c['p90']:.3f}  p99 {c['p99']:.3f}")
        print(f"  constant-factor err mean {100 * e['mean']:+.1f} percent, "
              f"p10 {100 * e['p10']:+.1f}, p90 {100 * e['p90']:+.1f}; "
              f"{100 * row['constant_factor_error']['share_underestimated']:.0f} percent "
              f"of pairs understated")
        print(f"  direction asymmetry mean {100 * row['asymmetry']['mean']:.1f} percent, "
              f"p90 {100 * row['asymmetry']['p90']:.1f}")
        print(f"  leg speed km/h      min {row['leg_speed_kmh']['min']:.1f}  "
              f"median {row['leg_speed_kmh']['median']:.1f}  "
              f"max {row['leg_speed_kmh']['max']:.1f}")
        ll = row["longest_leg"]
        print(f"  longest leg         {ll['road_km']:.2f} km by road, "
              f"{ll['constant_km']:.2f} under the constant; touches the depot: "
              f"{ll['touches_depot']}")
        rt = row["round_trips"]
        lo = rt["longest"]
        print(f"  longest round trip  {lo['road_km']:.2f} km by road "
              f"({lo['out_km']:.2f} out, {lo['back_km']:.2f} back), "
              f"{lo['constant_km']:.2f} under the constant "
              f"({100 * lo['relative_error']:+.1f} percent)")
        cost = rt["cost"]
        print(f"  round-trip cost     largest {cost['largest']['street']:.1f} "
              f"(constant {cost['largest']['constant']:.1f}, "
              f"half speed {cost['largest']['half_speed']:.1f}); "
              f"above {cost['above_half_ceiling']['threshold']:.0f}: "
              f"{cost['above_half_ceiling']['street']} street, "
              f"{cost['above_half_ceiling']['constant']} constant, "
              f"{cost['above_half_ceiling']['half_speed']} half speed")
        sp = row["short_pairs_under_100m"]
        if sp["pairs"]:
            print(f"  pairs under 100 m:  {sp['pairs']}, mean {sp['mean_straight_m']:.0f} m "
                  f"apart and {sp['mean_road_m']:.0f} m by road, worst "
                  f"{sp['max_road_m']:.0f} m")
    print(f"\nSaved {RESULTS / 'network.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
