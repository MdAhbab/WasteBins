"""
Figure: the guarantee, and the pricing defect it replaced.
==========================================================

Both panels come from ``results/wait_bound.json``.

Left: the hour the bound predicts and the hour the planner actually collects,
against distance from the depot.  The two coincide at every distance, so the
bound is attained rather than merely respected.  Shading marks the region where
the retired pricing never collected the container at any waiting time.

Right: why.  The retired skip price is the ordering score times a constant, so it
approaches a ceiling and never reaches it.  The current price grows linearly in
the wait.  A container 35 kilometres from the depot costs more to insert than the
ceiling, so under the retired price it is skipped forever, and the horizontal
line showing that insertion cost never meets the retired curve.

Run:  python -m experiments.fig_bound
Out:  results/figures/fig_bound.png
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from experiments import figstyle as FS                # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    FS.apply()
    data = json.loads((RESULTS / "wait_bound.json").read_text())
    sweep = data["sweep"]
    saturation = data["retired_pricing_saturation"]
    weights = data["protocol"]["weights"]
    tau = data["protocol"]["tau_h"]

    distance = np.array([r["distance_km"] for r in sweep])
    predicted = np.array([r["predicted_bound_h"] for r in sweep])
    served = np.array([r["served_at_h"] for r in sweep])
    cutoff = saturation["first_unaffordable_km"]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(FS.FULL_W, FS.FULL_W * 0.36))

    # --- left: predicted against observed --------------------------------
    left.axvspan(cutoff, distance.max() * 1.05, color=FS.GREY, alpha=0.12,
                 lw=0, zorder=0)
    left.plot(distance, predicted, color=FS.GREY, lw=1.4, zorder=2,
              label="Bound of (8)")
    left.plot(distance, served, ls="none", marker="o", ms=6, mfc="white",
              mec=FS.BLUE, mew=1.8, zorder=3, label="Hour actually collected")
    left.annotate("retired pricing never\ncollects beyond 33 km",
                  xy=(cutoff + 2, predicted.max() * 0.30),
                  color=FS.TEXT, fontsize=9, ha="left", va="center")
    for x, y in ((distance[-1], served[-1]),):
        left.annotate(f"{y:.0f} h", xy=(x, y), xytext=(-4, -12),
                      textcoords="offset points", color=FS.BLUE, fontsize=9,
                      ha="right")
    left.set_xlabel("Container distance from the depot (km)")
    left.set_ylabel("Hours until collected")
    left.set_xlim(0, distance.max() * 1.05)
    left.set_ylim(0, predicted.max() * 1.18)
    left.legend(frameon=False, loc="upper left", handlelength=1.6,
                borderaxespad=0.3)

    # --- right: the two skip prices --------------------------------------
    ceiling = saturation["ceiling"]
    waits = np.logspace(np.log10(tau), 6, 400)
    retired = weights["lambda_prize"] * weights["overdue_multiplier"] * (
        waits / (tau + waits))
    current = weights["lambda_prize"] * weights["overdue_multiplier"] * (
        waits / (2.0 * tau))
    insertion_35km = next(r["insertion_cost"] for r in sweep
                          if r["distance_km"] == 35.0)

    # The vertical axis is linear and clipped. On a log axis the ceiling at 180
    # and the insertion cost at 192 sit on top of each other, which hides the
    # only thing this panel exists to show: that the retired price passes under
    # the insertion cost and stays there.
    top = insertion_35km * 1.55
    right.axhspan(ceiling, top, color=FS.ORANGE, alpha=0.08, lw=0, zorder=0)
    right.plot(waits, current, color=FS.BLUE, lw=1.6, zorder=3,
               label="Current price, (5)")
    right.plot(waits, retired, color=FS.ORANGE, lw=1.6, ls=(0, (5, 2)), zorder=3,
               label="Retired price")
    right.axhline(ceiling, color=FS.ORANGE, lw=0.9, ls=":", zorder=2)
    right.axhline(insertion_35km, color=FS.GREY, lw=1.3, ls="-.", zorder=2)

    crossing = insertion_35km * 2.0 * tau / (
        weights["lambda_prize"] * weights["overdue_multiplier"])
    right.plot([crossing], [insertion_35km], marker="o", ms=5, color=FS.BLUE,
               zorder=5)
    right.annotate(f"served at {crossing:.0f} h", xy=(crossing, insertion_35km),
                   xytext=(14, 22), textcoords="offset points", color=FS.BLUE,
                   fontsize=9, ha="left",
                   arrowprops=dict(arrowstyle="-", lw=0.9, color=FS.BLUE))
    right.annotate(f"cost to reach it, {insertion_35km:.0f}",
                   xy=(9e5, insertion_35km), xytext=(0, 5),
                   textcoords="offset points", color=FS.TEXT, fontsize=9,
                   ha="right")
    right.annotate(f"ceiling {ceiling:.0f}, never reached",
                   xy=(9e5, ceiling), xytext=(0, -14),
                   textcoords="offset points", color=FS.ORANGE, fontsize=9,
                   ha="right")
    right.set_xscale("log")
    right.set_xlabel("Waiting time (hours, log scale)")
    right.set_ylabel("Cost charged for skipping")
    right.set_xlim(tau, 1e6)
    right.set_ylim(60, top)
    right.legend(frameon=False, loc="upper left", handlelength=1.6,
                 borderaxespad=0.3)

    for axis in (left, right):
        axis.grid(True, color=FS.GRID, lw=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    fig.tight_layout(pad=0.4, w_pad=1.6)
    out = FIGDIR / "fig_bound.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"wrote {out}  ({data['n_tight']} of {data['n_cases']} attained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
