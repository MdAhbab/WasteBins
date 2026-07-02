"""
Generate publication figures from the experiment result JSONs.
Writes PNGs to results/figures/ and also copies them into ../../Micro/images/.
"""
from __future__ import annotations
import json, pathlib, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"
FIG = RES / "figures"; FIG.mkdir(exist_ok=True)
IMG = HERE.parent.parent / "Micro" / "images"

plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
                     "figure.constrained_layout.use": True,
                     "axes.titlesize": 11, "axes.labelsize": 10.5})


def load(name):
    return json.loads((RES / name).read_text())


def fig_model():
    m = load("model.json")
    reg = m["B_forward_regression_time_to_overflow"]["models"]
    names = ["persistence_mean", "ridge_current_state", "random_forest", "hist_grad_boost"]
    labels = ["Persistence", "Ridge\n(current state)", "Random\nForest", "Hist Gradient\nBoosting"]
    r2 = [max(0, reg[n]["r2"]) for n in names]
    mae = [reg[n]["mae_h"] for n in names]
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(names))
    b = ax1.bar(x - 0.2, r2, 0.4, label="$R^2$", color="#2c7fb8")
    ax1.set_ylabel("$R^2$ (higher better)", color="#2c7fb8"); ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + 0.2, mae, 0.4, label="MAE (h)", color="#de2d26")
    ax2.set_ylabel("MAE hours (lower better)", color="#de2d26")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_title("Forward time-to-overflow prediction (temporal holdout)")
    for i, v in enumerate(r2): ax1.text(i - 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(mae): ax2.text(i + 0.2, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
    fig.savefig(FIG / "fig_model_bakeoff.png"); plt.close(fig)


def fig_calibration():
    m = load("model.json")
    rc = m["C_forward_classification_hazard"]["reliability_curve_calibrated"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
    ax.plot(rc["mean_predicted"], rc["fraction_positive"], "o-", color="#2c7fb8",
            label="Calibrated HGB (isotonic)")
    ax.set_xlabel("Mean predicted hazard probability")
    ax.set_ylabel("Observed hazard frequency")
    ax.set_title("Reliability curve (hazard within 6 h)")
    ax.legend(); fig.savefig(FIG / "fig_calibration.png"); plt.close(fig)


def fig_ablation():
    a = load("ablation.json")
    p = a["missingness"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(p, a["naive_mae"], "s--", color="#de2d26", label="Naive zero-fill")
    ax1.plot(p, a["renorm_mae"], "o-", color="#2c7fb8", label="Dynamic renormalisation")
    ax1.set_xlabel("Sensor missingness rate"); ax1.set_ylabel("Priority MAE vs full-sensor")
    ax1.set_title("Priority error under sensor loss"); ax1.legend()
    ax2.plot(p, a["naive_spearman"], "s--", color="#de2d26", label="Naive zero-fill")
    ax2.plot(p, a["renorm_spearman"], "o-", color="#2c7fb8", label="Dynamic renormalisation")
    ax2.set_xlabel("Sensor missingness rate"); ax2.set_ylabel("Spearman rank corr. vs full-sensor")
    ax2.set_title("Priority-ranking preservation"); ax2.legend()
    fig.savefig(FIG / "fig_ablation.png"); plt.close(fig)


def fig_routing():
    r = load("routing.json")
    sweep = r["C_alpha_sweep_mean"]
    alphas = sorted(float(k) for k in sweep)
    dist = [sweep[str(a)]["distance_km"] for a in alphas]
    ttc = [sweep[str(a)]["mean_time_to_critical_h"] for a in alphas]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # trade-off curve
    axb = ax1.twinx()
    ax1.plot(alphas, dist, "o-", color="#2c7fb8", label="Distance (km)")
    axb.plot(alphas, ttc, "s--", color="#de2d26", label="Mean time-to-critical (h)")
    ax1.set_xlabel(r"Priority warp $\alpha$"); ax1.set_ylabel("Distance (km)", color="#2c7fb8")
    axb.set_ylabel("Mean time-to-critical (h)", color="#de2d26")
    ax1.set_title(r"Sustainability$\leftrightarrow$response trade-off vs $\alpha$")
    # policy comparison
    pols = [("A_static_full", "Static\nsweep"), ("B_threshold", "Threshold"),
            ("C_priority_warped_a1", "Warped\ngreedy"),
            ("D_priority_plus_localsearch", "Warped+\nlocal search"),
            ("E_orienteering_65pct_budget", "Orienteering\n(proposed)")]
    d = [r[k]["distance_km"] for k, _ in pols]
    co2 = [r[k]["co2_kg"] for k, _ in pols]
    x = np.arange(len(pols))
    ax2.bar(x - 0.2, d, 0.4, label="Distance (km)", color="#2c7fb8")
    ax2.bar(x + 0.2, co2, 0.4, label="CO$_2$ (kg)", color="#31a354")
    ax2.set_xticks(x); ax2.set_xticklabels([l for _, l in pols], fontsize=8)
    ax2.set_title("Distance & CO$_2$ by routing policy"); ax2.legend()
    fig.savefig(FIG / "fig_routing.png"); plt.close(fig)


def fig_equity():
    e = load("equity.json")
    s = e["sweep"]
    g = [x["gamma"] for x in s]
    worst = [x["worst_wait_h"] for x in s]
    gini = [x["gini_visits"] for x in s]
    ttc = [x["mean_time_to_critical_h"] for x in s]
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(g, worst, "o-", color="#2c7fb8", label="Worst-case wait (h)")
    ax1.set_xlabel(r"Aging weight $\gamma$"); ax1.set_ylabel("Worst-case wait (h)", color="#2c7fb8")
    ax2 = ax1.twinx()
    ax2.plot(g, gini, "s--", color="#de2d26", label="Gini of visits")
    ax2.plot(g, ttc, "^:", color="#31a354", label="Time-to-critical (h)")
    ax2.set_ylabel("Gini / time-to-critical", color="#333")
    ax1.set_title("Equity vs hazard-response as aging increases")
    lines1, l1 = ax1.get_legend_handles_labels(); lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, fontsize=9, loc="center right")
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
