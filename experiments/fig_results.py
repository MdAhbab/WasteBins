"""
Figures of the multi-cycle and weather results, from the raw records.
=====================================================================

``fig_waits``
    Left: the share of collections made after each wait, pooled over the
    networks of the moderate load, after the burn-in.  Right: the longest wait
    of each network under each policy, with the deadline marked.

``fig_weather``
    The rain scenario under the full rule: containers collected and the longest
    wait in each cycle, as means over networks, for plans made blind, made blind
    with the reserved stops protected, and made on current conditions.  The
    spell is shaded.

A figure whose records do not exist yet is skipped with a message.

Run:  python -m experiments.fig_results
Out:  results/figures/fig_waits.pdf, fig_weather.pdf (and .png)
"""
from __future__ import annotations

import pathlib
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.ticker import MaxNLocator              # noqa: E402

from experiments import figstyle as FS                # noqa: E402
from experiments import store as ST                   # noqa: E402

FIGDIR = ST.RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
TAU_H = 48.0

#: Policies drawn, with a fixed colour, line style and marker each.  The rule of
#: the paper is blue; the urgency rule it is set against is orange; the steps
#: between them are grey; the other planner is green.
WAIT_POLICIES = [
    ("insertion/urgency", "Urgency", FS.ORANGE, "-", "v"),
    ("insertion/ageing", "Ageing", FS.ORANGE, "--", "^"),
    ("insertion/tier_unbounded", "Tier, unbounded, no reservation", FS.GREY, "-", "o"),
    ("insertion/full", "Full rule, r = 1", FS.BLUE, "-", "s"),
    ("insertion/full_r4", "Full rule, r = 4", FS.BLUE, "--", "D"),
    ("ortools/full", "OR-Tools, full rule, r = 1", FS.GREEN, "-", "P"),
]
WEATHER_MODES = [("blind", "Planned for nominal conditions", FS.ORANGE, "-"),
                 ("protected", "Nominal plan, reserved stops protected", FS.GREY, "--"),
                 ("current", "Planned for current conditions", FS.BLUE, "-")]


def _save(fig, name: str) -> None:
    out = FIGDIR / f"{name}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")


def fig_waits(load: str = "moderate") -> None:
    records = ST.read(f"rollout-{load}")
    chains = defaultdict(dict)
    for r in records:
        chains[f"{r['planner']}/{r['rule']}"][r["network"]] = r
    shown = [p for p in WAIT_POLICIES if p[0] in chains]
    if not shown:
        print(f"fig_waits: no rollout-{load} records yet")
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(FS.FULL_W, 2.8),
                                      gridspec_kw={"width_ratios": [1.15, 1.0]})
    for policy, name, colour, style, _marker in shown:
        waits = np.array([w for r in chains[policy].values()
                          for cycle, _i, w, _h in r["services"] if cycle >= r["burn_in"]])
        if waits.size == 0:
            continue
        grid = np.arange(0.0, waits.max() + 12.0, 12.0)
        share = [(waits >= g).mean() for g in grid]
        left.step(grid, share, where="post", color=colour, ls=style, label=name)
    left.axvline(TAU_H, color=FS.TEXT, lw=0.8, ls=":", label=f"Deadline, {TAU_H:g} h")
    left.set_yscale("log")
    left.set_xlabel("Wait at collection (h)")
    left.set_ylabel("Share of collections at or\nabove this wait (log scale)")
    left.set_title("(a) Waits at collection", loc="left")
    left.legend(loc="upper right", fontsize=8.5, handlelength=2.0)

    for row, (policy, name, colour, _style, marker) in enumerate(shown):
        worst = [r["summary"]["worst_wait_any_dispatch_h"] for r in chains[policy].values()]
        jitter = np.linspace(-0.18, 0.18, len(worst)) if len(worst) > 1 else [0.0]
        right.scatter(worst, row + np.asarray(jitter), color=colour, marker=marker,
                      s=16, zorder=3, edgecolors="none")
    right.axvline(TAU_H, color=FS.TEXT, lw=0.8, ls=":")
    right.set_yticks(range(len(shown)))
    right.set_yticklabels([s[1] for s in shown])
    right.invert_yaxis()
    right.set_xlabel("Longest wait of a network (h)")
    right.set_title("(b) Longest wait, one point per network", loc="left")
    _save(fig, "fig_waits")


def fig_weather(scenario: str = "rain", policy: str = "insertion/full") -> None:
    records = [r for r in ST.read(f"weather-{scenario}")
               if f"{r['planner']}/{r['rule']}" == policy]
    if not records:
        print(f"fig_weather: no weather-{scenario} records for {policy} yet")
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(FS.FULL_W, 2.9))
    for mode, name, colour, style in WEATHER_MODES:
        chains = [r for r in records if r["mode"] == mode]
        if not chains:
            continue
        cycles = np.arange(len(chains[0]["per_cycle"]))
        served = np.mean([[c["served"] for c in r["per_cycle"]] for r in chains], axis=0)
        longest = np.mean([[c["max_wait_h"] for c in r["per_cycle"]] for r in chains], axis=0)
        left.plot(cycles, served, color=colour, ls=style, label=name)
        right.plot(cycles, longest, color=colour, ls=style, label=name)
    for ax in (left, right):
        ax.axvspan(15.5, 27.5, color=FS.GREY, alpha=0.12, lw=0)
        ax.set_xlim(8, len(records[0]["per_cycle"]) - 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Dispatch cycle (12 h each)")
    left.set_ylabel("Containers collected")
    right.set_ylabel("Longest wait (h)")
    left.set_title("(a) Containers collected per cycle", loc="left")
    right.set_title("(b) Longest wait at dispatch", loc="left")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8.5)
    _save(fig, "fig_weather")


def main() -> int:
    FS.apply()
    fig_waits()
    fig_weather()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
