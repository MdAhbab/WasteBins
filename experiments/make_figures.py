"""
Generate publication figures from the experiment result JSONs.
Writes PNGs to results/figures/ and also copies them into ../../Micro/images/.

Figures follow the shared IEEE style in figstyle.py: single-column plots are
drawn at 3.5 in, double-column plots at 7.16 in, so no text is rescaled in
the manuscript.
"""
from __future__ import annotations
import json, pathlib, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle
from figstyle import BLUE, ORANGE, GREEN, GREY, COL_W, FULL_W

figstyle.apply()

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"
FIG = RES / "figures"; FIG.mkdir(exist_ok=True)
IMG = HERE.parent.parent / "Micro" / "images"


def load(name):
    return json.loads((RES / name).read_text())


def fig_model():
    """Single-column bar chart: forward time-to-overflow bake-off."""
    m = load("model.json")
    reg = m["B_forward_regression_time_to_overflow"]["models"]
    names = ["persistence_mean", "ridge_current_state", "random_forest", "hist_grad_boost"]
    labels = ["Persistence", "Ridge\n(current)", "Random\nForest", "Hist. Grad.\nBoosting"]
    r2 = [max(0, reg[n]["r2"]) for n in names]
    mae = [reg[n]["mae_h"] for n in names]
    fig, ax1 = plt.subplots(figsize=(COL_W, 2.7))
    x = np.arange(len(names))
    ax1.bar(x - 0.2, r2, 0.4, label="$R^2$", color=BLUE)
    ax1.set_ylabel("$R^2$ (higher is better)", color=BLUE)
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, mae, 0.4, label="MAE (h)", color=ORANGE)
    ax2.set_ylabel("MAE, hours (lower is better)", color=ORANGE)
    ax2.set_ylim(0, 9.2)
    ax2.grid(False)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_title("Forward time-to-overflow prediction")
    for i, v in enumerate(r2):
        ax1.text(i - 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(mae):
        ax2.text(i + 0.2, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
    fig.savefig(FIG / "fig_model_bakeoff.png"); plt.close(fig)


def fig_calibration():
    """Single-column reliability curve for the calibrated hazard classifier."""
    m = load("model.json")
    rc = m["C_forward_classification_hazard"]["reliability_curve_calibrated"]
    fig, ax = plt.subplots(figsize=(COL_W, 3.1))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect calibration")
    ax.plot(rc["mean_predicted"], rc["fraction_positive"], "o-", color=BLUE,
            label="Calibrated gradient boosting")
    ax.set_xlabel("Mean predicted hazard probability")
    ax.set_ylabel("Observed hazard frequency")
    ax.set_title("Reliability curve (hazard within 6 h)")
    ax.legend(loc="upper left")
    fig.savefig(FIG / "fig_calibration.png"); plt.close(fig)


def fig_ablation():
    """Double-column, two panels: sensor-failure ablation."""
    a = load("ablation.json")
    p = a["missingness"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.6))
    ax1.plot(p, a["naive_mae"], "s--", color=ORANGE, label="Naive zero-fill")
    ax1.plot(p, a["renorm_mae"], "o-", color=BLUE, label="Dynamic renormalisation")
    ax1.set_xlabel("Sensor missingness rate")
    ax1.set_ylabel("Priority MAE vs full-sensor")
    ax1.set_title("Priority error under sensor loss")
    ax1.legend(loc="upper left")
    ax2.plot(p, a["naive_spearman"], "s--", color=ORANGE, label="Naive zero-fill")
    ax2.plot(p, a["renorm_spearman"], "o-", color=BLUE, label="Dynamic renormalisation")
    ax2.set_xlabel("Sensor missingness rate")
    ax2.set_ylabel("Spearman corr. vs full-sensor")
    ax2.set_title("Priority-ranking preservation")
    ax2.legend(loc="lower left")
    fig.savefig(FIG / "fig_ablation.png"); plt.close(fig)


def fig_routing():
    """Double-column, two panels: alpha trade-off and policy comparison."""
    r = load("routing.json")
    sweep = r["C_alpha_sweep_mean"]
    alphas = sorted(float(k) for k in sweep)
    dist = [sweep[str(a)]["distance_km"] for a in alphas]
    ttc = [sweep[str(a)]["mean_time_to_critical_h"] for a in alphas]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 2.7))
    axb = ax1.twinx()
    ax1.plot(alphas, dist, "o-", color=BLUE, label="Distance (km)")
    axb.plot(alphas, ttc, "s--", color=ORANGE, label="Mean time-to-critical (h)")
    axb.grid(False)
    ax1.set_xlabel(r"Priority warp $\alpha$")
    ax1.set_ylabel("Distance (km)", color=BLUE)
    axb.set_ylabel("Mean time-to-critical (h)", color=ORANGE)
    ax1.set_title(r"Cost vs response trade-off as $\alpha$ grows")
    pols = [("A_static_full", "Static\nsweep"), ("B_threshold", "Threshold"),
            ("C_priority_warped_a1", "Warped\ngreedy"),
            ("D_priority_plus_localsearch", "Warped\n+ 2-opt"),
            ("E_orienteering_65pct_budget", "Orient.\n(proposed)")]
    d = [r[k]["distance_km"] for k, _ in pols]
    co2 = [r[k]["co2_kg"] for k, _ in pols]
    x = np.arange(len(pols))
    ax2.bar(x - 0.2, d, 0.4, label="Distance (km)", color=BLUE)
    ax2.bar(x + 0.2, co2, 0.4, label="CO$_2$ (kg)", color=GREEN)
    for i, v in enumerate(d):
        ax2.text(i - 0.2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    for i, v in enumerate(co2):
        ax2.text(i + 0.2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    ax2.set_ylim(0, 37)
    ax2.set_xticks(x); ax2.set_xticklabels([l for _, l in pols], fontsize=9)
    ax2.set_title("Distance and CO$_2$ by routing policy")
    ax2.legend(loc="upper right")
    fig.savefig(FIG / "fig_routing.png"); plt.close(fig)


def fig_equity():
    """Single-column, two stacked panels: aging sweep."""
    e = load("equity.json")
    s = e["sweep"]
    g = [x["gamma"] for x in s]
    worst = [x["worst_wait_h"] for x in s]
    gini = [x["gini_visits"] for x in s]
    ttc = [x["mean_time_to_critical_h"] for x in s]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.9), sharex=True)
    ax1.plot(g, worst, "o-", color=BLUE, label="Worst-case wait (h)")
    for xi, v in zip(g, worst):
        ax1.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=9)
    ax1.set_ylabel("Worst-case wait (h)")
    ax1.set_ylim(0, 270)
    ax1.set_title("Service equity as the aging weight increases")
    ax2.plot(g, gini, "s--", color=ORANGE)
    ax2.plot(g, ttc, "^:", color=GREEN)
    ax2.annotate("Gini of visits", (g[1], gini[1]), color=ORANGE,
                 textcoords="offset points", xytext=(6, 8), fontsize=9)
    ax2.annotate("Mean time-to-critical (h)", (g[1], ttc[1]), color=GREEN,
                 textcoords="offset points", xytext=(6, -13), fontsize=9)
    ax2.set_xlabel(r"Aging weight $\gamma$")
    ax2.set_ylabel("Gini / time (h)")
    ax2.set_ylim(0.2, 0.4)
    fig.savefig(FIG / "fig_equity.png"); plt.close(fig)


def main():
    fig_model(); fig_calibration(); fig_ablation(); fig_routing(); fig_equity()
    made = sorted(FIG.glob("*.png"))
    IMG.mkdir(exist_ok=True)
    for f in made:
        shutil.copy(f, IMG / f.name)
    print("figures:", [f.name for f in made])
    print("copied to", IMG)


if __name__ == "__main__":
    main()
