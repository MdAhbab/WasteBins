"""
Experiment 2 (reviewer Q5): Controlled sensor-failure ablation.
===============================================================
The manuscript's central novelty is "Dynamic Weight Renormalisation": when a
sensor fails, its weight is redistributed over the surviving sensors instead of
defaulting the missing reading to zero (which would artificially deflate the
bin's priority). It was never tested. Here we test it.

Protocol
--------
For missingness p in {0.0, 0.1, ..., 0.9}:
  * draw many (bin, time) snapshots from the simulated latent state,
  * for each snapshot compute the TRUE priority (all sensors present),
  * independently drop each modality with prob p, then compute priority with
      (i) RENORMALISATION  -> denominator = sum of ACTIVE weights,
      (ii) NAIVE ZERO-FILL -> missing modality contributes 0 with FULL denominator.
  * report the mean absolute priority error vs the true priority, and the
    Spearman rank correlation of the 30-bin priority ordering vs the true order
    (route quality depends on ordering, not absolute values).
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import spearmanr
import dataset as D

RESULTS = D.RESULTS
SEED = 123

# priority feature config (matches the manuscript's runtime calculator)
FEATURES = {
    "waste":    {"w": 0.35, "min": 0.0, "max": 1.0, "impact": "pos"},
    "gas":      {"w": 0.25, "min": 0.0, "max": 1.0, "impact": "pos"},
    "temp":     {"w": 0.10, "min": 10.0, "max": 40.0, "opt": 25.0, "impact": "dev"},
    "humidity": {"w": 0.05, "min": 50.0, "max": 100.0, "impact": "pos"},
}


def _norm(name, val):
    c = FEATURES[name]
    if c["impact"] == "dev":
        max_dev = max(abs(c["max"] - c["opt"]), abs(c["min"] - c["opt"]))
        n = abs(val - c["opt"]) / max_dev if max_dev > 0 else 0.0
    else:
        n = (val - c["min"]) / (c["max"] - c["min"]) if c["max"] > c["min"] else 0.0
    return min(1.0, max(0.0, n))


def priority(values, active, mode):
    """values: dict name->val; active: dict name->bool; mode: 'renorm'|'naive'|'true'."""
    num = 0.0
    w_active = 0.0
    w_total = sum(FEATURES[n]["w"] for n in FEATURES)
    for name, c in FEATURES.items():
        if mode != "true" and not active[name]:
            # naive: contributes 0 but keeps its weight in the denominator (below)
            continue
        num += c["w"] * _norm(name, values[name])
        w_active += c["w"]
    if mode == "renorm" or mode == "true":
        denom = w_active if w_active > 0 else 1.0
    else:  # naive zero-fill keeps the FULL weight in the denominator
        denom = w_total
    return num / denom


def main():
    payload, _ = D.build()
    latent = payload["latent"]
    bins = sorted(latent.keys())
    rng = np.random.default_rng(SEED)

    # pick many common timestamps where all bins have data
    T = min(len(latent[b]) for b in bins)
    sample_ts = rng.choice(np.arange(50, T), size=400, replace=False)

    ps = [round(x, 1) for x in np.arange(0.0, 0.91, 0.1)]
    res = {"missingness": ps, "renorm_mae": [], "naive_mae": [],
           "renorm_spearman": [], "naive_spearman": []}

    for p in ps:
        renorm_err, naive_err = [], []
        renorm_rho, naive_rho = [], []
        for t in sample_ts:
            true_scores, renorm_scores, naive_scores = [], [], []
            for b in bins:
                row = latent[b].iloc[int(t)]
                vals = {"waste": float(row["fill"]), "gas": float(row["gas"]),
                        "temp": float(row["temp"]), "humidity": float(row["humidity"])}
                active = {n: (rng.random() >= p) for n in FEATURES}
                # guarantee at least one active sensor for renorm to be defined
                if not any(active.values()):
                    active[rng.choice(list(FEATURES))] = True
                pt = priority(vals, active, "true")
                pr = priority(vals, active, "renorm")
                pn = priority(vals, active, "naive")
                true_scores.append(pt); renorm_scores.append(pr); naive_scores.append(pn)
                renorm_err.append(abs(pr - pt)); naive_err.append(abs(pn - pt))
            # ranking preservation across the 30 bins at this timestamp
            if np.std(true_scores) > 1e-9:
                rr = spearmanr(true_scores, renorm_scores).correlation
                nn = spearmanr(true_scores, naive_scores).correlation
                if not np.isnan(rr): renorm_rho.append(rr)
                if not np.isnan(nn): naive_rho.append(nn)
        res["renorm_mae"].append(round(float(np.mean(renorm_err)), 4))
        res["naive_mae"].append(round(float(np.mean(naive_err)), 4))
        res["renorm_spearman"].append(round(float(np.mean(renorm_rho)), 4))
        res["naive_spearman"].append(round(float(np.mean(naive_rho)), 4))

    # headline numbers at 50% missingness
    i50 = ps.index(0.5)
    res["headline_at_50pct"] = {
        "renorm_mae": res["renorm_mae"][i50], "naive_mae": res["naive_mae"][i50],
        "mae_reduction_pct": round(100 * (1 - res["renorm_mae"][i50] / max(res["naive_mae"][i50], 1e-9)), 1),
        "renorm_spearman": res["renorm_spearman"][i50], "naive_spearman": res["naive_spearman"][i50],
    }
    (RESULTS / "ablation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
