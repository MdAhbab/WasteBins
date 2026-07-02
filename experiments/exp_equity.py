"""
Experiment 4 (reviewer Q8): Service equity / anti-starvation.
=============================================================
Cost-warping structurally favours high-priority clusters, so low-priority bins
can be starved. We add an AGING term to the routing prize:

    P_eff(bin) = P(bin) + gamma * min(1, hours_since_last_visit / TAU)

and run a multi-shift collection simulation (8-hour shifts over 14 days). Each
shift the truck serves an urgent subset within a fixed distance budget
(prize-collecting orienteering). We compare gamma sweep on:
  * worst-case wait (h)  -- starvation indicator
  * mean wait (h)
  * Gini of per-bin visit counts (0 = perfectly equal)
  * fraction of bins never served
  * mean time-to-serve genuinely critical bins (must not degrade much)
"""
from __future__ import annotations
import json
import numpy as np
import dataset as D
import routing as RT
from exp_routing import snapshot   # reuse the snapshot builder

RESULTS = D.RESULTS
SHIFT_H = 8
DAYS = 14
TAU = 48.0            # aging saturates after 48h unattended


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def run(latent, bins_df, gamma):
    bins = sorted(latent.keys())
    n = len(bins)
    last_visit = {i: 0.0 for i in range(n)}     # hours
    visit_count = {i: 0 for i in range(n)}
    max_wait = {i: 0.0 for i in range(n)}
    crit_times = []
    depot_coord = (23.8069, 90.3687)

    for shift in range(int(DAYS * 24 / SHIFT_H)):
        t = 50 + shift * SHIFT_H
        if t >= min(len(latent[b]) for b in bins):
            break
        coords, prio, crit, tto = snapshot(latent, bins_df, t)
        # effective prize with aging
        p_eff = {}
        for i in range(n):
            wait = t - last_visit[i]
            p_eff[i] = prio[i] + gamma * min(1.0, wait / TAU)

        all_coords = coords + [depot_coord]
        M = RT.dist_matrix(all_coords)
        depot = n
        # budget = 55% of a full geographic sweep
        full = RT.or_opt(RT.two_opt(RT.greedy_nn(list(range(n)), M, depot, alpha=0.0), M, depot), M, depot)
        budget = 0.55 * RT.tour_distance(full, M, depot)
        order, _, _ = RT.orienteering_greedy(list(range(n)), M, depot, p_eff, budget)

        at = RT.arrival_times_h(order, M, depot)
        for j in order:
            visit_count[j] += 1
            last_visit[j] = t + at[j]
        # track critical response
        for j in order:
            if crit[j]:
                crit_times.append(at[j])
        # update running max wait for everyone
        for i in range(n):
            max_wait[i] = max(max_wait[i], t - last_visit[i])

    waits = list(max_wait.values())
    never = sum(1 for i in range(n) if visit_count[i] == 0)
    return {
        "gamma": gamma,
        "worst_wait_h": round(float(np.max(waits)), 1),
        "mean_wait_h": round(float(np.mean(waits)), 1),
        "gini_visits": round(gini(list(visit_count.values())), 3),
        "never_served": int(never),
        "mean_time_to_critical_h": round(float(np.mean(crit_times)) if crit_times else 0.0, 3),
        "total_visits": int(sum(visit_count.values())),
    }


def main():
    payload, _ = D.build()
    latent, bins_df = payload["latent"], payload["bins_df"]
    out = {"shift_h": SHIFT_H, "days": DAYS, "tau_h": TAU, "sweep": []}
    for g in (0.0, 0.25, 0.5, 1.0):
        r = run(latent, bins_df, g)
        out["sweep"].append(r)
        print(r)
    # headline: no-aging vs aging=0.5
    base = out["sweep"][0]; aged = [s for s in out["sweep"] if s["gamma"] == 0.5][0]
    out["headline"] = {
        "worst_wait_reduction_h": round(base["worst_wait_h"] - aged["worst_wait_h"], 1),
        "gini_improvement": round(base["gini_visits"] - aged["gini_visits"], 3),
        "never_served_before": base["never_served"], "never_served_after": aged["never_served"],
        "critical_response_cost_h": round(aged["mean_time_to_critical_h"] - base["mean_time_to_critical_h"], 3),
    }
    (RESULTS / "equity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["headline"], indent=2))


if __name__ == "__main__":
    main()
