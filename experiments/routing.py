"""
Routing primitives: Haversine matrix, priority-warped edges, greedy nearest
neighbour, 2-opt + Or-opt local search, and prize-collecting orienteering.

References (cited in the manuscript revision):
  * Croes (1958) 2-opt local search.
  * Lin & Kernighan (1973) improved tour heuristics.
  * Balas (1989) prize-collecting TSP.
  * Golden, Levy, Vohra (1987); Vansteenwegen et al. (2011) orienteering.
"""
from __future__ import annotations
import math
import numpy as np

R_EARTH = 6371000.0
SPEED_KMH = 20.0                 # urban refuse-truck average speed
CO2_KG_PER_KM = 1.05             # refuse collection vehicle emission proxy (diesel HGV)


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R_EARTH * 2 * math.asin(math.sqrt(a))


def dist_matrix(coords):
    n = len(coords)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j] = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
    return M


def tour_distance(order, M, depot):
    """Real metric distance of visiting `order` starting/ending at depot index."""
    if not order:
        return 0.0
    d = M[depot, order[0]]
    for a, b in zip(order[:-1], order[1:]):
        d += M[a, b]
    d += M[order[-1], depot]      # return to depot
    return d


def greedy_nn(cand, M, depot, priority=None, alpha=0.0):
    """
    Greedy nearest-neighbour over candidates.
    alpha=0 -> pure geographic (baseline).
    alpha>0 -> priority-warped edge cost = dist / (1 + alpha*10*P[dest]) (manuscript).
    """
    unvisited = set(cand)
    order = []
    cur = depot
    while unvisited:
        def cost(j):
            d = M[cur, j]
            if alpha > 0 and priority is not None:
                return d / (1.0 + alpha * 10.0 * priority[j])
            return d
        nxt = min(unvisited, key=cost)
        order.append(nxt); unvisited.remove(nxt); cur = nxt
    return order


def two_opt(order, M, depot, max_pass=30):
    """Standard 2-opt on real distance; returns improved order."""
    if len(order) < 4:
        return order
    best = order[:]
    improved = True
    full = [depot] + best + [depot]
    def seglen(seq):
        return sum(M[seq[i], seq[i+1]] for i in range(len(seq)-1))
    cur = seglen(full)
    passes = 0
    while improved and passes < max_pass:
        improved = False; passes += 1
        for i in range(1, len(full) - 2):
            for k in range(i + 1, len(full) - 1):
                if k - i == 1:
                    continue
                new = full[:i] + full[i:k+1][::-1] + full[k+1:]
                nl = seglen(new)
                if nl + 1e-9 < cur:
                    full = new; cur = nl; improved = True
    return full[1:-1]


def or_opt(order, M, depot, seg_sizes=(1, 2, 3), max_pass=10):
    """Or-opt: relocate short segments to cheaper positions (real distance)."""
    full = [depot] + order[:] + [depot]
    def seglen(seq):
        return sum(M[seq[i], seq[i+1]] for i in range(len(seq)-1))
    cur = seglen(full); improved = True; passes = 0
    while improved and passes < max_pass:
        improved = False; passes += 1
        for s in seg_sizes:
            for i in range(1, len(full) - s - 1):
                seg = full[i:i+s]
                rest = full[:i] + full[i+s:]
                for j in range(1, len(rest)):
                    cand = rest[:j] + seg + rest[j:]
                    nl = seglen(cand)
                    if nl + 1e-9 < cur:
                        full = cand; cur = nl; improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return full[1:-1]


def orienteering_greedy(cand, M, depot, prize, budget_m):
    """
    Prize-collecting orienteering: greedily insert the node with the best
    prize/added-distance ratio while staying within a distance budget.
    Returns (order, collected_prize, distance).
    """
    order = []
    remaining = set(cand)
    while remaining:
        best_j, best_ratio, best_pos = None, -1, None
        for j in remaining:
            # cheapest insertion position
            seq = [depot] + order + [depot]
            best_ins = None
            for pos in range(1, len(seq)):
                add = M[seq[pos-1], j] + M[j, seq[pos]] - M[seq[pos-1], seq[pos]]
                if best_ins is None or add < best_ins[0]:
                    best_ins = (add, pos)
            add, pos = best_ins
            ratio = prize[j] / max(add, 1.0)
            if ratio > best_ratio:
                best_ratio, best_j, best_pos, best_add = ratio, j, pos, add
        # try to add best_j
        trial = order[:best_pos-1] + [best_j] + order[best_pos-1:]
        if tour_distance(trial, M, depot) <= budget_m:
            order = trial; remaining.remove(best_j)
        else:
            remaining.remove(best_j)   # cannot afford; skip
        if not remaining:
            break
    order = two_opt(order, M, depot)
    collected = sum(prize[j] for j in order)
    return order, collected, tour_distance(order, M, depot)


def arrival_times_h(order, M, depot):
    """Cumulative arrival time (hours) at each visited node, at SPEED_KMH."""
    times = {}
    cur = depot; d = 0.0
    for j in order:
        d += M[cur, j]
        times[j] = (d / 1000.0) / SPEED_KMH
        cur = j
    return times


def metrics(order, M, depot, critical_mask, time_to_overflow_h=None):
    """Sustainability + service metrics for a completed tour."""
    dist_km = tour_distance(order, M, depot) / 1000.0
    at = arrival_times_h(order, M, depot)
    crit = [j for j in order if critical_mask.get(j, False)]
    crit_times = [at[j] for j in crit] if crit else [0.0]
    missed = 0
    if time_to_overflow_h is not None:
        for j in crit:
            if at[j] > time_to_overflow_h.get(j, 1e9):
                missed += 1
    return {
        "distance_km": round(dist_km, 3),
        "co2_kg": round(dist_km * CO2_KG_PER_KM, 3),
        "n_visited": len(order),
        "n_critical_served": len(crit),
        "mean_time_to_critical_h": round(float(np.mean(crit_times)), 3),
        "worst_time_to_critical_h": round(float(np.max(crit_times)), 3),
        "missed_overflow": int(missed),
    }
