"""
Figure: what a constant detour factor cannot represent.
=======================================================

Two panels, both generated from ``results/network.json`` so the figure and the
table cannot disagree.

Left: the distribution of circuity over the routed instances in both cities,
with the constant factor of 1.30 drawn as a vertical reference.  A constant is
adequate only if the distribution is narrow around it, and in neither city is it.

Right: the signed error the constant factor makes against the shortest path, as
a cumulative distribution.  Where the curve crosses zero says what share of
ordered pairs the approximation understates.

Run:  python -m experiments.fig_network
Out:  results/figures/fig_network.png
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

from wastebins_core import roadnet as RN              # noqa: E402
from wastebins_core.geo import dist_matrix            # noqa: E402
from experiments import exp_fleet as EF               # noqa: E402
from experiments import figstyle as FS                # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

CITY_COLOUR = {"dhaka": FS.BLUE, "wyndham": FS.ORANGE}
CITY_LABEL = {"dhaka": "Dhaka", "wyndham": "Wyndham"}


def collect(study: str, n_snapshots: int = 25):
    """Circuity and constant-factor error over the routed instances."""
    EF.set_study(study)
    net = RN.load(EF.STUDY["area"])
    rng = np.random.default_rng(EF.SEED)
    circuity, error = [], []
    for i in range(n_snapshots):
        snapshot = EF.make_snapshot(rng, i)
        coords = snapshot["coords"]
        road, _ff, _snap = net.matrices(coords)
        approx = dist_matrix(coords, detour_factor=1.30)
        off = ~np.eye(len(coords), dtype=bool)
        circuity.append(net.circuity(coords, road)[off])
        with np.errstate(invalid="ignore", divide="ignore"):
            error.append(((approx - road) / np.where(road > 0, road, np.nan))[off])
    c = np.concatenate(circuity)
    e = np.concatenate(error)
    return c[np.isfinite(c)], e[np.isfinite(e)]


def main() -> int:
    FS.apply()
    summary = json.loads((RESULTS / "network.json").read_text())
    data = {study: collect(study) for study in ("dhaka", "wyndham")}

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(FS.FULL_W, FS.FULL_W * 0.36))

    # --- left: circuity distribution ------------------------------------
    # The mean is carried in the legend label rather than annotated on the
    # curve, because in Wyndham the mean sits on the reference line and the two
    # labels would overlap.
    bins = np.linspace(1.0, 2.6, 65)
    left.axvline(1.30, color=FS.GREY, lw=1.2, ls="--", zorder=1)
    for study in ("dhaka", "wyndham"):
        circuity, _err = data[study]
        mean = summary["areas"][study]["circuity"]["mean"]
        left.hist(circuity, bins=bins, density=True, histtype="step", lw=1.6,
                  color=CITY_COLOUR[study], zorder=3,
                  label=f"{CITY_LABEL[study]}, mean {mean:.2f}")
        left.plot([mean], [0.0], marker="v", ms=5, clip_on=False,
                  color=CITY_COLOUR[study], zorder=4)
    left.set_xlabel("Circuity (road distance / straight-line distance)")
    left.set_ylabel("Density")
    left.set_xlim(1.0, 2.6)
    left.set_ylim(0, left.get_ylim()[1] * 1.05)
    left.annotate("assumed constant 1.30", xy=(1.32, left.get_ylim()[1] * 0.55),
                  color=FS.TEXT, fontsize=9, ha="left", va="center")
    left.legend(frameon=False, loc="upper right", handlelength=1.3,
                borderaxespad=0.2)

    # --- right: signed error of the constant factor ----------------------
    right.axvline(0.0, color=FS.GREY, lw=1.0, ls="--", zorder=1)
    placement = {"dhaka": ((-10, 12), "right"), "wyndham": ((12, -4), "left")}
    for study in ("dhaka", "wyndham"):
        _circuity, err = data[study]
        values = np.sort(err) * 100.0
        cdf = np.arange(1, values.size + 1) / values.size
        # Dashed for the second city so the two curves stay distinguishable in
        # greyscale, where hue carries nothing.
        right.plot(values, cdf, lw=1.6, color=CITY_COLOUR[study], zorder=3,
                   ls="-" if study == "dhaka" else (0, (5, 2)),
                   label=CITY_LABEL[study])
        share = summary["areas"][study]["constant_factor_error"]["share_underestimated"]
        right.plot([0.0], [share], marker="o", ms=4.5,
                   color=CITY_COLOUR[study], zorder=5)
        offset, align = placement[study]
        right.annotate(f"{100 * share:.0f} percent\nunderstated", xy=(0.0, share),
                       xytext=offset, textcoords="offset points",
                       color=CITY_COLOUR[study], fontsize=9, ha=align, va="center")
    right.set_xlabel("Error of the constant factor against shortest path (percent)")
    right.set_ylabel("Cumulative share of pairs")
    right.set_xlim(-60, 50)
    right.set_ylim(0, 1.0)
    right.legend(frameon=False, loc="lower right", handlelength=1.3,
                 borderaxespad=0.4)

    for axis in (left, right):
        axis.grid(True, color=FS.GRID, lw=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    fig.tight_layout(pad=0.4, w_pad=1.6)
    out = FIGDIR / "fig_network.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
