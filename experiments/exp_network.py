"""
What the street graph says that a constant detour factor cannot.
================================================================

Routing studies in this area commonly measure distance as the great-circle
separation between two stops multiplied by one detour factor, typically between
1.2 and 1.4.  This script measures the error that approximation makes on the
exact instances the routing study uses, before any planner runs, so the finding
is a property of the two cities and not of our heuristic.

Four quantities are reported per study area.

``circuity``
    Road distance divided by great-circle distance, over ordered pairs at least
    100 m apart.  The mean answers whether a constant is centred correctly; the
    spread and the upper tail answer whether a constant can work at all.

``asymmetry``
    The relative gap between the two directions of travel between the same pair
    of stops.  A symmetric matrix cannot represent it, and it comes from one-way
    restrictions rather than from noise.

``short pairs``
    Ordered pairs less than 100 m apart in a straight line, and the road
    distance actually needed between them.  A great-circle model treats these as
    nearly free.

``matrix error``
    Signed error of the constant-factor matrix against the shortest-path matrix,
    which is the quantity a planner's cost function inherits directly.

Run:  python -m experiments.exp_network
Out:  results/network.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import roadnet as RN            # noqa: E402
from wastebins_core.geo import dist_matrix          # noqa: E402
from experiments import exp_fleet as EF             # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

#: The factor the earlier version of this work used, and the value most commonly
#: quoted for a dense urban grid.
CONSTANT_DETOUR = 1.30


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


def analyse(study: str, n_snapshots: int = 25) -> Dict:
    """Measure one study area over the snapshots the routing study uses."""
    EF.set_study(study)
    net = RN.load(EF.STUDY["area"])
    rng = np.random.default_rng(EF.SEED)
    snapshots = [EF.make_snapshot(rng, i) for i in range(n_snapshots)]

    circuity: List[np.ndarray] = []
    asymmetry: List[np.ndarray] = []
    rel_error: List[np.ndarray] = []
    abs_error_m: List[np.ndarray] = []
    snap_m: List[float] = []
    short_pairs: List[Dict] = []
    freeflow: List[np.ndarray] = []

    for snapshot in snapshots:
        coords = snapshot["coords"]
        road, ff, snapped = net.matrices(coords)
        approx = dist_matrix(coords, detour_factor=CONSTANT_DETOUR)
        off = ~np.eye(len(coords), dtype=bool)

        circuity.append(net.circuity(coords, road)[off])
        with np.errstate(invalid="ignore", divide="ignore"):
            asymmetry.append((np.abs(road - road.T) / np.where(road > 0, road, np.nan))[off])
            rel_error.append(((approx - road) / np.where(road > 0, road, np.nan))[off])
        abs_error_m.append((approx - road)[off])
        snap_m.append(float(snapped.max()))
        short_pairs.append(net.short_pair_detour(coords, road))
        freeflow.append(ff[off])

    circuity_all = np.concatenate(circuity)
    rel_all = np.concatenate(rel_error)
    pairs_total = int(sum(sp.get("pairs", 0) for sp in short_pairs))

    out = {
        "study": study,
        "area": EF.STUDY["area"],
        "n_snapshots": n_snapshots,
        "n_stops": len(snapshots[0]["coords"]),
        "graph": {
            "nodes": net.n_nodes,
            "arcs": net.n_arcs,
            "fingerprint": RN.fingerprint(net),
            "bbox": net.meta["bbox"],
            "ways_parsed": net.meta["ways_parsed"],
            "strong_components_before_trim": net.meta["strong_components"],
            "tagged_maxspeed_share": net.meta["tagged_maxspeed_share"],
            "speed_scale": net.meta["speed_scale"],
            "oneway_arc_share": float(1.0 - _reciprocal_share(net)),
        },
        "snap_distance_m": {"max_over_snapshots": float(max(snap_m)),
                            "mean_of_max": float(np.mean(snap_m))},
        "circuity": _describe(circuity_all),
        "asymmetry": _describe(np.concatenate(asymmetry)),
        "freeflow_kmh": _describe(np.concatenate(freeflow)),
        "constant_factor_error": {
            "factor": CONSTANT_DETOUR,
            "relative": _describe(rel_all),
            "absolute_m": _describe(np.concatenate(abs_error_m)),
            "share_underestimated": float(np.mean(rel_all[np.isfinite(rel_all)] < 0.0)),
            "note": "negative relative error means the constant-factor matrix is "
                    "shorter than the shortest path on the street graph",
        },
        "short_pairs_under_100m": {
            "pairs_per_instance": pairs_total / max(1, len(short_pairs)),
            "mean_straight_m": float(np.mean([sp["mean_straight_m"] for sp in short_pairs
                                              if sp.get("pairs")])) if pairs_total else None,
            "mean_road_m": float(np.mean([sp["mean_road_m"] for sp in short_pairs
                                          if sp.get("pairs")])) if pairs_total else None,
            "max_road_m": float(max([sp["max_road_m"] for sp in short_pairs
                                     if sp.get("pairs")])) if pairs_total else None,
        },
        "best_fit_constant": float(np.nanmean(circuity_all)),
    }
    return out


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
        print(f"\n{name}: {row['n_stops']} stops on {row['graph']['nodes']} nodes, "
              f"{row['graph']['arcs']} arcs")
        print(f"  one-way arcs        {100 * row['graph']['oneway_arc_share']:.1f} percent")
        print(f"  maxspeed tagged     {100 * row['graph']['tagged_maxspeed_share']:.1f} percent of ways")
        print(f"  circuity            mean {c['mean']:.3f}  median {c['median']:.3f}  "
              f"p90 {c['p90']:.3f}  p99 {c['p99']:.3f}")
        print(f"  direction asymmetry mean {row['asymmetry']['mean']:.3f}  "
              f"p90 {row['asymmetry']['p90']:.3f}")
        print(f"  free-flow km/h      mean {row['freeflow_kmh']['mean']:.1f}  "
              f"p10 {row['freeflow_kmh']['p10']:.1f}  p90 {row['freeflow_kmh']['p90']:.1f}")
        print(f"  constant-factor err mean {100 * e['mean']:+.1f} percent, "
              f"p10 {100 * e['p10']:+.1f}, p90 {100 * e['p90']:+.1f}; "
              f"{100 * row['constant_factor_error']['share_underestimated']:.0f} percent "
              f"of pairs understated")
        sp = row["short_pairs_under_100m"]
        if sp["pairs_per_instance"]:
            print(f"  stops under 100 m apart: {sp['pairs_per_instance']:.1f} ordered pairs "
                  f"per instance, mean {sp['mean_straight_m']:.0f} m apart in a straight "
                  f"line and {sp['mean_road_m']:.0f} m by road, worst {sp['max_road_m']:.0f} m")
    print(f"\nSaved {RESULTS / 'network.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
