"""
Figure: what the method costs as the network grows.
===================================================

Both panels come from ``results/scale.json``.

Left: wall-clock time to produce a plan against the number of containers, on log
axes, for the proposed method and the strongest baseline. A reference slope is
drawn so the reader can read the growth exponent off the plot rather than take
our word for it. The matrix build, meaning the shortest-path computation on the
street graph, is shown separately because it is a fixed cost per instance and
not part of the search.

Right: whether quality degrades as the instance grows. The objective gap against
the baseline and the share of containers served are both plotted, because a
planner that keeps its objective advantage by serving fewer containers has not
kept anything worth having.

Run:  python -m experiments.fig_scale
Out:  results/figures/fig_scale.png
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
    data = json.loads((RESULTS / "scale.json").read_text())
    rows = data["sizes"]
    comparator = data["comparator"]

    n = np.array([r["n_bins"] for r in rows], dtype=float)
    solve = np.array([r["proposed"]["solve_s"] for r in rows])
    base_solve = np.array([r[comparator]["solve_s"] for r in rows])
    matrix = np.array([r["matrix_build_s"] for r in rows])
    gap = np.array([r["objective_gap_pct"] for r in rows])
    served = np.array([100 * r["proposed"]["served_share"] for r in rows])

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(FS.FULL_W, FS.FULL_W * 0.36))

    # --- left: cost ------------------------------------------------------
    left.plot(n, solve, marker="o", ms=5, lw=1.6, color=FS.BLUE,
              label="Proposed, search")
    left.plot(n, base_solve, marker="s", ms=4.5, lw=1.6, color=FS.ORANGE,
              ls=(0, (5, 2)), label=f"{comparator.upper()}, search")
    left.plot(n, matrix, marker="^", ms=4.5, lw=1.4, color=FS.GREEN,
              ls=(0, (1, 1.6)), label="Shortest paths, per instance")

    # Reference slopes, so the growth exponent is readable off the plot.
    anchor = solve[0] / (n[0] ** 2)
    left.plot(n, anchor * n ** 2, lw=0.9, color=FS.GREY, ls=":", zorder=1)
    left.annotate("slope 2", xy=(n[-1], anchor * n[-1] ** 2), xytext=(-4, 6),
                  textcoords="offset points", color=FS.GREY, fontsize=9,
                  ha="right")
    left.annotate(f"{solve[-1]:.0f} s at {n[-1]:.0f}", xy=(n[-1], solve[-1]),
                  xytext=(-6, -14), textcoords="offset points",
                  color=FS.BLUE, fontsize=9, ha="right")

    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("Containers in the instance (log scale)")
    left.set_ylabel("Seconds to produce a plan (log scale)")
    left.set_xticks(n)
    left.set_xticklabels([f"{int(v)}" for v in n])
    left.legend(frameon=False, loc="upper left", handlelength=1.8,
                borderaxespad=0.3)

    # --- right: quality --------------------------------------------------
    right.plot(n, gap, marker="o", ms=5, lw=1.6, color=FS.BLUE,
               label=f"Objective advantage over {comparator.upper()}")
    right.axhline(0.0, color=FS.GREY, lw=1.0, ls="--", zorder=1)
    for x, y in zip(n, gap):
        right.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 8),
                       textcoords="offset points", color=FS.BLUE, fontsize=9,
                       ha="center")

    twin = right.twinx()
    twin.plot(n, served, marker="s", ms=4.5, lw=1.6, color=FS.GREEN,
              ls=(0, (5, 2)), label="Containers served")
    twin.set_ylabel("Containers served (percent)", color=FS.GREEN)
    twin.tick_params(axis="y", colors=FS.GREEN)
    twin.set_ylim(0, 105)
    for side in ("top",):
        twin.spines[side].set_visible(False)

    right.set_xscale("log")
    right.set_xlabel("Containers in the instance (log scale)")
    right.set_ylabel("Objective advantage (percent)", color=FS.BLUE)
    right.tick_params(axis="y", colors=FS.BLUE)
    right.set_xticks(n)
    right.set_xticklabels([f"{int(v)}" for v in n])
    right.set_ylim(min(0.0, gap.min() * 1.4), max(gap.max() * 1.35, 5.0))

    handles = (right.get_legend_handles_labels()[0]
               + twin.get_legend_handles_labels()[0])
    labels = (right.get_legend_handles_labels()[1]
              + twin.get_legend_handles_labels()[1])
    right.legend(handles, labels, frameon=False, loc="lower left",
                 handlelength=1.8, borderaxespad=0.3)

    for axis in (left, right):
        axis.grid(True, color=FS.GRID, lw=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    right.spines["right"].set_visible(True)
    right.spines["right"].set_color(FS.GREEN)

    fig.tight_layout(pad=0.4, w_pad=1.8)
    out = FIGDIR / "fig_scale.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
