"""
Figure of the open fill-level corpus, from its harmonised tables.
================================================================

``fig_corpus``
    (a) The relative fill rate of each container, as a cumulative share, in
    each city.  Wyndham and Mildura use the mean daily rise; Boston, which has
    no daily rise, uses yellow calls per hour below yellow (see
    `experiments.corpus`).
    (b) The share of a day's fill in each local hour: the daily rise of the
    hourly Mildura record and the first yellow calls of the Boston cycles,
    against the day and night split of the demand model.
    (c) The Boston wait at full: the share of cycles with a red call whose
    emptying came later than each wait, for the main definition of an emptying
    and for the sensitivity case in which every collection call empties.

Run:  python -m experiments.corpus, then python -m experiments.fig_corpus
Out:  results/figures/fig_corpus.pdf (and .png)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

from experiments import corpus as CO                  # noqa: E402
from experiments import exp_rollout as R              # noqa: E402
from experiments import figstyle as FS                # noqa: E402
from experiments import store as ST                   # noqa: E402

FIGDIR = ST.RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
#: One colour, line style and marker per city.  Blue and orange stay reserved
#: for the rule and its comparator in the other figures.
CITIES = [("wyndham", "Wyndham", "#6250D6", "-", "o"),
          ("mildura", "Mildura", "#199E70", "--", "s"),
          ("boston", "Boston", "#C98500", "-.", "^")]
WAIT_MARK_H = 72.0


def _ecdf(values: np.ndarray):
    x = np.sort(values)
    return x, np.arange(1, len(x) + 1) / len(x)


def main() -> int:
    daily = pd.read_csv(CO.OUT / "daily.csv", dtype={"container_id": str})
    cycles = pd.read_csv(CO.OUT / "boston_cycles.csv", dtype={"container_id": str})
    kept = cycles[cycles["kept"]]
    summary = json.loads((CO.RESULTS / "corpus.json").read_text())

    FS.apply()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(FS.FULL_W, 2.6))

    # (a) Spread of fill rates between containers.
    rates = {city: (r["mean"] / r["mean"].median()).to_numpy()
             for city, r in CO.container_rates(daily).items()}
    rates["boston"] = CO.boston_relative_rates(kept).to_numpy()
    for city, name, colour, style, _marker in CITIES:
        x, y = _ecdf(rates[city])
        a.step(x, y, where="post", color=colour, ls=style, label=f"{name} ({len(x)})")
    a.set_xscale("log")
    a.set_xlim(0.1, 30)
    a.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    a.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    a.minorticks_off()
    a.set_ylim(0, 1.02)
    a.axvline(1.0, color=FS.GREY, lw=0.6, ls=":")
    a.set_xlabel("Fill rate / city median (log)")
    a.set_ylabel("Share of containers")
    a.set_title("(a) Fill rates", loc="left")

    # (b) Share of the daily fill by local hour.
    hours = np.arange(24)
    mildura = np.asarray(summary["diurnal"]["mildura"]["share_by_hour"])
    yellow = kept["first_yellow_hour"].dropna().astype(int)
    boston = np.bincount(yellow, minlength=24) / len(yellow)
    day = np.isin(hours, list(CO.DAY_HOURS))
    factor = np.where(day, R.DAY_FACTOR, R.NIGHT_FACTOR)
    model = factor / factor.sum()
    b.step(np.r_[hours, 24], 100 * np.r_[model, model[-1]], where="post", color=FS.GREY,
           lw=1.2, ls=":", label="Demand model")
    for city, series in (("mildura", mildura), ("boston", boston)):
        _c, name, colour, style, marker = next(x for x in CITIES if x[0] == city)
        b.plot(hours + 0.5, 100 * series, color=colour, ls=style, marker=marker,
               markersize=3.5, label=name)
    b.set_xlim(0, 24)
    b.set_xticks([0, 6, 12, 18, 24])
    b.set_ylim(bottom=0)
    b.set_xlabel("Local hour")
    b.set_ylabel("Share in each hour (%)")
    b.set_title("(b) Fill by hour", loc="left")

    # (c) Boston wait from the first red call to emptying.
    readings = pd.read_csv(CO.OUT / "readings.csv", dtype={"container_id": str},
                           low_memory=False)
    bos = readings[readings["city"] == "boston"].copy()
    bos["collection_flag"] = bos["collection_flag"].astype(str).str.lower().eq("true")
    bos["time_local"] = pd.to_datetime(bos["time_local"], format="ISO8601")
    alt_cycles, _ = CO.boston_cycles(bos, any_call=True)
    alt = alt_cycles[alt_cycles["kept"]]
    grid = np.arange(0.0, 24.0 * CO.BOSTON_MAX_CYCLE_DAYS + 1.0, 1.0)
    _c, name, colour, style, _m = CITIES[2]
    for waits, label, col, ls in ((kept["red_wait_h"], "Green call", colour, "-"),
                                  (alt["red_wait_h"], "Any call", FS.GREY, "--")):
        w = waits.dropna().to_numpy()
        c.plot(grid, 100 * np.array([(w > g).mean() for g in grid]), color=col, ls=ls,
               label=label)
    c.axvline(WAIT_MARK_H, color=FS.TEXT, lw=0.8, ls=":")
    c.text(WAIT_MARK_H + 6, 30, f"{WAIT_MARK_H:g} h", fontsize=8.5, color=FS.TEXT)
    c.set_xlim(0, 336)
    c.set_xticks([0, 72, 168, 336])
    c.set_ylim(0, 100)
    c.set_xlabel("Wait from full to emptying (h)")
    c.set_ylabel("Share waiting longer (%)")
    c.set_title("(c) Boston wait at full", loc="left")
    c.legend(loc="upper right", fontsize=8.5, handlelength=1.8, title="Emptying",
             title_fontsize=8.5)

    handles, labels = a.get_legend_handles_labels()
    hb, lb = b.get_legend_handles_labels()
    handles.append(hb[0])
    labels.append(lb[0])
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8.5,
               handlelength=2.2)
    out = FIGDIR / "fig_corpus.png"
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
