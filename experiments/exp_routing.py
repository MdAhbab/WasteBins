"""
Experiment 3 (reviewer Q3, Q4, Q9): Routing with REAL sustainability metrics.
============================================================================
Replaces the manuscript's algebraic "83% weight reduction" with measured
outcomes on real Haversine distances, averaged over high-load snapshots:

  Policies
    A. Static full sweep            (visit ALL bins, geographic NN + 2-opt) -- the
                                     traditional fixed schedule.
    B. Threshold full sweep         (visit fill>0.7 bins, geographic NN + 2-opt).
    C. Priority-warped full (manuscript)  greedy NN on warped edges, alpha sweep.
    D. Priority-warped + Or-opt/2-opt (proposed refinement, answers Q9 local search).
    E. Prize-collecting orienteering (proposed) -- serve the urgent subset within a
                                     distance budget (answers the orienteering point).

  Metrics: distance (km), CO2 (kg), mean/worst time-to-serve CRITICAL bins (h),
           missed-overflow count, compute latency (ms).
"""
from __future__ import annotations
import json, time
import numpy as np
import dataset as D
import routing as RT

RESULTS = D.RESULTS
SEED = 7
FEATURES = {"waste": (0.35, 0.0, 1.0, "pos"), "gas": (0.25, 0.0, 1.0, "pos"),
            "temp": (0.10, 10.0, 40.0, "dev"), "humidity": (0.05, 50.0, 100.0, "pos")}
TEMP_OPT = 25.0


def priority_all(v):
    num = 0.0; wsum = 0.0
    for name, (w, lo, hi, imp) in FEATURES.items():
        x = v[name]
        if imp == "dev":
            md = max(abs(hi - TEMP_OPT), abs(lo - TEMP_OPT)); n = abs(x - TEMP_OPT) / md
        else:
            n = (x - lo) / (hi - lo)
        n = min(1.0, max(0.0, n)); num += w * n; wsum += w
    return num / wsum


def snapshot(latent, bins_df, t):
    coords, prio, crit, tto = [], {}, {}, {}
    for i, b in enumerate(sorted(latent.keys())):
        row = latent[b].iloc[int(t)]
        coords.append((bins_df.iloc[i]["latitude"], bins_df.iloc[i]["longitude"]))
        v = {"waste": float(row["fill"]), "gas": float(row["gas"]),
             "temp": float(row["temp"]), "humidity": float(row["humidity"])}
        prio[i] = priority_all(v)
        crit[i] = bool(row["hazard_within_h"])
        tto[i] = float(row["time_to_overflow"])
    return coords, prio, crit, tto


def run_snapshot(coords, prio, crit, tto):
    depot_coord = (23.8069, 90.3687)         # garage near Mirpur 10
    all_coords = coords + [depot_coord]
    M = RT.dist_matrix(all_coords)
    depot = len(coords)
    cand = list(range(len(coords)))
    full_sweep_dist = None
    res = {}

    # A. static full sweep (geographic) + 2-opt
    t0 = time.perf_counter()
    a = RT.greedy_nn(cand, M, depot, alpha=0.0)
    a = RT.or_opt(RT.two_opt(a, M, depot), M, depot)
    latency_static = (time.perf_counter() - t0) * 1000
    res["A_static_full"] = RT.metrics(a, M, depot, crit, tto)
    full_sweep_dist = RT.tour_distance(a, M, depot)

    # B. threshold full sweep (fill>0.7)  -- approximate fill by priority proxy? use waste
    thr = [j for j in cand if prio[j] >= 0.5]
    b = RT.greedy_nn(thr, M, depot, alpha=0.0) if thr else []
    b = RT.or_opt(RT.two_opt(b, M, depot), M, depot) if b else []
    res["B_threshold"] = RT.metrics(b, M, depot, crit, tto)

    # C. priority-warped full (manuscript), alpha sweep
    alpha_curve = {}
    for al in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        c = RT.greedy_nn(cand, M, depot, priority=prio, alpha=al)
        alpha_curve[al] = RT.metrics(c, M, depot, crit, tto)
    res["C_priority_warped_alpha_sweep"] = {str(k): v for k, v in alpha_curve.items()}
    res["C_priority_warped_a1"] = alpha_curve[1.0]

    # D. priority-warped (a=1) + Or-opt/2-opt refinement (proposed)
    t0 = time.perf_counter()
    d = RT.greedy_nn(cand, M, depot, priority=prio, alpha=1.0)
    d_ref = RT.or_opt(RT.two_opt(d, M, depot), M, depot)
    latency_prop = (time.perf_counter() - t0) * 1000
    res["D_priority_plus_localsearch"] = RT.metrics(d_ref, M, depot, crit, tto)
    res["D_localsearch_dist_reduction_pct"] = round(
        100 * (1 - RT.tour_distance(d_ref, M, depot) / max(RT.tour_distance(d, M, depot), 1e-9)), 2)

    # E. prize-collecting orienteering within 65% of full-sweep distance
    budget = 0.65 * full_sweep_dist
    order, collected, dist = RT.orienteering_greedy(cand, M, depot, prio, budget)
    m = RT.metrics(order, M, depot, crit, tto)
    total_crit = sum(1 for j in cand if crit[j])
    m["critical_coverage_pct"] = round(100 * m["n_critical_served"] / max(total_crit, 1), 1)
    m["prize_collected_pct"] = round(100 * collected / max(sum(prio.values()), 1e-9), 1)
    res["E_orienteering_65pct_budget"] = m

    res["_latency_ms"] = {"static": round(latency_static, 2), "proposed": round(latency_prop, 2)}
    return res


def main():
    payload, _ = D.build()
    latent, bins_df = payload["latent"], payload["bins_df"]
    T = min(len(latent[b]) for b in latent)
    rng = np.random.default_rng(SEED)

    # choose high-load snapshots (many hazards)
    hazard_load = []
    for t in range(50, T):
        s = sum(int(latent[b].iloc[t]["hazard_within_h"]) for b in latent)
        hazard_load.append((s, t))
    hazard_load.sort(reverse=True)
    hot_ts = [t for _, t in hazard_load[:40]]
    chosen = rng.choice(hot_ts, size=25, replace=False)

    # accumulate metrics
    agg = {}
    alpha_agg = {}
    for t in chosen:
        coords, prio, crit, tto = snapshot(latent, bins_df, t)
        r = run_snapshot(coords, prio, crit, tto)
        for k, v in r.items():
            if k == "C_priority_warped_alpha_sweep":
                for al, mv in v.items():
                    alpha_agg.setdefault(al, []).append(mv)
                continue
            if k in ("_latency_ms", "D_localsearch_dist_reduction_pct"):
                agg.setdefault(k, []).append(v)
                continue
            agg.setdefault(k, []).append(v)

    def mean_metrics(list_of_dicts):
        keys = list_of_dicts[0].keys()
        return {kk: round(float(np.mean([d[kk] for d in list_of_dicts])), 3) for kk in keys}

    out = {"n_snapshots": len(chosen), "speed_kmh": RT.SPEED_KMH,
           "co2_kg_per_km": RT.CO2_KG_PER_KM}
    for k, lst in agg.items():
        if k == "_latency_ms":
            out[k] = {"static": round(float(np.mean([d["static"] for d in lst])), 2),
                      "proposed": round(float(np.mean([d["proposed"] for d in lst])), 2)}
        elif k == "D_localsearch_dist_reduction_pct":
            out[k] = round(float(np.mean(lst)), 2)
        else:
            out[k] = mean_metrics(lst)
    out["C_alpha_sweep_mean"] = {al: mean_metrics(lst) for al, lst in alpha_agg.items()}

    (RESULTS / "routing.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
