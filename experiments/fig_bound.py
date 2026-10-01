"""
Figure: what a skip penalty achieves without a reservation, and with one.
=========================================================================

Both panels come from ``results/wait_bound.json``.

Left: one overdue container at eight distances from the depot, one free
vehicle.  Without a reservation the planner collects it at the wait that
Proposition 3 predicts, which grows with the cost of the round trip.  With one
reserved head it is collected at the deadline at every distance.  Each of the
eight distances is marked on the axis.  Shading marks the distances at which a
bounded penalty never collects the container.

Right: why the bounded penalty fails.  The bounded penalty approaches its
ceiling from below, so a round trip that costs more than the ceiling is never
worth taking.  The unbounded penalty crosses any round-trip cost at a finite
wait.  The two horizontal lines are the round-trip costs of two swept
distances, taken from the same file.

Run:  python -m experiments.fig_bound
Out:  results/figures/fig_bound.pdf and .png
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

TAU_H = 48.0
LAMBDA, MU = 45.0, 4.0
SHOWN_KM = (20.0, 40.0)        # round-trip costs drawn in the right panel


def main() -> int:
    FS.apply()
    data = json.loads((RESULTS / "wait_bound.json").read_text())
    sweep = data["price_sweep"]
    ceiling_km = data["bounded_penalty"]["first_unaffordable_km"]
    ceiling = data["bounded_penalty"]["ceiling"]

    km = np.array([r["distance_km"] for r in sweep])
    predicted = np.array([r["predicted_h"] for r in sweep])
    priced = np.array([np.nan if r["served_at_h_price_only"] is None
                       else r["served_at_h_price_only"] for r in sweep])
    reserved = np.array([np.nan if r["served_at_h_reserved"] is None
                         else r["served_at_h_reserved"] for r in sweep])

    fig, (left, right) = plt.subplots(1, 2, figsize=(FS.FULL_W, 2.7))

    # The eight distances are equally spaced on the axis, so each case is
    # legible; the dashed line joins the predictions as a guide only.
    x = np.arange(len(km))
    first_never = int(np.argmax(km > ceiling_km))
    left.axvspan(first_never - 0.5, len(km) - 0.5, color=FS.GREY, alpha=0.12, lw=0,
                 label="Bounded penalty: never collected")
    left.plot(x, predicted, color=FS.GREY, ls="--", lw=1.2,
              label="Predicted, Proposition 3")
    left.plot(x, priced, ls="none", marker="o", color=FS.ORANGE, mfc="white",
              mew=1.3, label="Observed, no reservation")
    left.plot(x, reserved, ls="none", marker="s", color=FS.BLUE,
              label="Observed, one reserved head")
    left.set_xticks(x)
    left.set_xticklabels([f"{k:g}" for k in km])
    left.set_xlim(-0.5, len(km) - 0.5)
    left.set_ylim(0, max(np.nanmax(predicted), np.nanmax(priced)) * 1.45)
    left.set_xlabel("Distance from the depot (km)")
    left.set_ylabel("Wait at collection (h)")
    left.set_title("(a) Wait at collection", loc="left")
    left.legend(loc="upper left", handlelength=1.6)

    wait = np.linspace(TAU_H, 300.0, 400)
    bounded = LAMBDA * MU * wait / (TAU_H + wait)
    unbounded = LAMBDA * MU * wait / (2.0 * TAU_H)
    right.plot(wait, unbounded, color=FS.BLUE, label="Unbounded penalty")
    right.plot(wait, bounded, color=FS.ORANGE, ls="--", label="Bounded penalty")
    right.axhline(ceiling, color=FS.ORANGE, ls=":", lw=1.0)
    right.text(296, ceiling + 5, "Ceiling", ha="right", va="bottom", fontsize=8.5,
               color=FS.ORANGE)
    costs = {r["distance_km"]: r for r in sweep}
    for distance in SHOWN_KM:
        row = costs[distance]
        right.axhline(row["round_trip_cost"], color=FS.GREY, ls="-.", lw=1.0)
        right.text(296, row["round_trip_cost"] + 5, f"Round trip, {distance:g} km",
                   ha="right", va="bottom", fontsize=8.5, color=FS.GREY)
        right.plot([row["price_threshold_h"]], [row["round_trip_cost"]], marker="o",
                   color=FS.BLUE, ms=4.5, zorder=3)
    right.set_xlim(TAU_H, 300.0)
    right.set_ylim(0, 320.0)
    right.set_xlabel("Wait (h)")
    right.set_ylabel("Cost units")
    right.set_title("(b) Skip penalty of an overdue container", loc="left")
    right.legend(loc="lower right", handlelength=2.2)

    out = FIGDIR / "fig_bound.png"
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
