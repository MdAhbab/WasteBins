"""
Multi-cycle rollout on observed demand.
=======================================

The Dhaka rollouts simulate demand: a mean rate per container, a neighbourhood
multiplier, noise and sudden loads, most of them assumed.  This study replaces
all of that with the observed record of Wyndham.  Each container fills, cycle
by cycle, at the rise its own sensor reported on that calendar day.  Only the
overall level is set, so that the fleet meets the same pressure as at the
moderate and the tight load of the Dhaka study.

Windows
-------
A window is 20 consecutive days of the record, which is 40 cycles of 12 hours.
Sixteen windows that do not overlap are drawn with a fixed seed from the days
on which at least ``MIN_REPORTING`` containers reported on at least three
quarters of the days.  Each window is one network of the paired comparison,
as R0 to R15 are in Dhaka.

Demand of a window
------------------
The daily rise of container ``i`` on day ``d`` is taken from the corpus (see
`experiments.corpus`).  A day on which it is undefined, because the container
was collected or near full, takes the container's mean rise over the window.
The rises of a window are divided by their mean, so that their average is one,
and the rate of a cycle is that relative rise times ``1 / fill_hours`` times
the day or night factor of the Dhaka model.  The heterogeneity between
containers and the variation from day to day are therefore observed; the level
and the split of a day into two cycles are not.

Fleet
-----
One vehicle with the same shift time per container as the Dhaka networks:
240 minutes for 60 containers, so 4 minutes per container.

Run:  python -m experiments.exp_replay --load moderate --windows W00,W01
Out:  results/raw/kg-replay-<load>.sNN.jsonl
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import corpus as CO               # noqa: E402
from experiments import exp_fleet as EF            # noqa: E402
from experiments import exp_rollout as R           # noqa: E402
from experiments import instances as IN            # noqa: E402
from experiments import policies as PO             # noqa: E402
from experiments import store as ST                # noqa: E402
from experiments import wyndham as WY              # noqa: E402

WINDOW_DAYS = 20
N_WINDOWS = 16
MIN_REPORTING = 28
SEED = 2026
SHIFT_MIN_PER_CONTAINER = 240.0 / 60.0
STUDY_PREFIX = "kg-replay-"

REFERENCE = ("insertion", "full")
SPECS: List[Tuple[str, str]] = [
    ("insertion", "urgency"), ("insertion", "tier_unbounded"),
    ("insertion", "full_hazard"),
    ("ortools", "urgency"), ("ortools", "tier_unbounded"), ("ortools", "full"),
    ("static_sweep", "full"),
]

_DAILY = None


def daily_rises() -> pd.DataFrame:
    """Wyndham daily rises, containers as columns and days as rows."""
    global _DAILY
    if _DAILY is None:
        path = CO.OUT / "daily.csv"
        frame = pd.read_csv(path, parse_dates=["date"], dtype={"container_id": str})
        frame = frame[frame["city"] == "wyndham"]
        rises = frame.pivot_table(index="date", columns="container_id", values="rise")
        reported = frame.pivot_table(index="date", columns="container_id",
                                     values="readings", aggfunc="sum")
        _DAILY = (rises, reported.reindex_like(rises))
    return _DAILY


def windows() -> Dict[str, pd.Timestamp]:
    """Sixteen non-overlapping 20-day windows, by name, as their first day."""
    rises, reported = daily_rises()
    days = pd.date_range(rises.index.min(), rises.index.max())
    reporting = (reported.reindex(days).fillna(0) > 0).sum(axis=1)
    starts = []
    for k in range(len(days) - WINDOW_DAYS + 1):
        span = reporting.iloc[k:k + WINDOW_DAYS]
        if (span >= MIN_REPORTING).mean() >= 0.75:
            starts.append(k)
    rng = np.random.default_rng(SEED)
    chosen: List[int] = []
    for k in rng.permutation(starts):
        if all(abs(int(k) - c) >= WINDOW_DAYS for c in chosen):
            chosen.append(int(k))
        if len(chosen) == N_WINDOWS:
            break
    if len(chosen) < N_WINDOWS:
        raise RuntimeError(f"only {len(chosen)} windows available")
    return {f"W{j:02d}": days[k] for j, k in enumerate(sorted(chosen))}


class ReplayWorld(R.World):
    """A Wyndham window with its observed demand, in the shape run_chain expects."""

    def __init__(self, window: str, load: str, n_cycles: int = R.N_CYCLES):
        self.network, self.load = window, load
        self.params = R.LOADS[load]
        self.n_cycles = n_cycles
        first = windows()[window]
        rises, reported = daily_rises()
        span = pd.date_range(first, periods=WINDOW_DAYS)
        block = rises.reindex(span)
        seen = reported.reindex(span).fillna(0) > 0
        containers, _ = WY.load()
        containers = containers.assign(container_id=containers["serial"].astype(str))
        keep = [c for c in containers["container_id"] if c in block.columns
                and seen[c].mean() >= 0.5 and block[c].notna().sum() >= 3]
        containers = containers[containers["container_id"].isin(keep)].reset_index(drop=True)
        block = block[keep]
        EF.set_study("wyndham", n_bins=len(containers), n_vehicles=1)
        self.n = len(containers)

        mean_rise = block.mean(axis=0)
        filled = block.fillna(mean_rise)
        self.imputed_share = float(block.isna().to_numpy().mean())
        relative = (filled / float(filled.to_numpy().mean())).to_numpy()   # days x n

        self.node_ids = [int(x) for x in containers["serial"]]
        self.coords = [WY.depot(containers)] + [(float(r.latitude), float(r.longitude))
                                                for r in containers.itertuples()]
        self.index_of = {nid: i + 1 for i, nid in enumerate(self.node_ids)}
        self.capacity_l = np.full(self.n, WY.CAPACITY_L)
        self.streams = ["general" if s == "general" else "recyclable"
                        for s in containers["stream"]]
        seed = [SEED, int(window[1:]), int(self.params["fill_hours"])]
        rng = np.random.default_rng(seed)
        self.organic = np.array([0.55 if s == "general" else 0.15 for s in self.streams])
        self.density = np.full(self.n, WY.DENSITY_KG_PER_M3)
        self.service_min = rng.uniform(2.5, 6.0, self.n)
        # The servicing windows of the Dhaka networks, scaled to the shorter
        # shift: they open at the start, at half or at three quarters of it.
        self.shift_min = round(SHIFT_MIN_PER_CONTAINER * self.n, 1)
        starts = rng.choice([0.0, 0.0, 0.0, 0.5, 0.75], size=self.n) * self.shift_min
        lengths = rng.choice([300.0, 360.0, 480.0], size=self.n)
        self.windows = [(float(a), float(a + b)) for a, b in zip(starts, lengths)]

        base = 1.0 / float(self.params["fill_hours"])
        per_cycle = np.repeat(relative, 2, axis=0)[:n_cycles]               # cycles x n
        factor = np.array([R.DAY_FACTOR if c % 2 == 0 else R.NIGHT_FACTOR
                           for c in range(n_cycles)])[:, None]
        self.trace = base * per_cycle * factor
        self.rate = base * relative.mean(axis=0)
        self.epsilon = relative.mean(axis=0)
        self.neighbourhood = np.zeros(self.n, dtype=int)
        self.neighbourhood_factor = np.ones(self.n)
        self.persistent_hazard = np.zeros(self.n, dtype=bool)
        shape = (n_cycles, self.n)
        self.rate_noise = np.ones(shape)
        self.sudden = np.zeros(shape, dtype=bool)
        self.sudden_time = np.zeros(shape)
        self.sudden_size = np.zeros(shape)
        self.gas_noise = rng.normal(0.0, R.GAS_NOISE, shape)
        self.temp_noise = rng.normal(0.0, R.TEMP_NOISE, shape)
        self.humidity_noise = rng.normal(0.0, R.HUMIDITY_NOISE, shape)

        probe = {"coords": self.coords, "node_ids": self.node_ids,
                 "streams": dict(zip(self.node_ids, self.streams)), "index": 0}
        self.travel = EF.travel_for(probe, IN.WHEN)
        self.fleet = EF.scarce_fleet(probe, shift_minutes=self.shift_min)
        self.first_day = str(first.date())

    def ambient(self, cycle: int):
        return 20.0, 65.0

    def cycle_rate(self, cycle: int) -> np.ndarray:
        return self.trace[cycle] * self.demand_factor(cycle)

    def describe(self) -> Dict:
        out = super().describe()
        out.update({"city": "wyndham", "first_day": self.first_day,
                    "window_days": WINDOW_DAYS, "shift_min": self.shift_min,
                    "imputed_share": round(self.imputed_share, 4)})
        return out


def run_window(load: str, window: str, shard: int, done: Dict[str, Dict]) -> int:
    study = f"{STUDY_PREFIX}{load}"
    key = lambda p, r: f"{study}|{window}|{p}|{r}"           # noqa: E731
    wanted = [s for s in SPECS if key(*s) not in done]
    if key(*REFERENCE) in done and not wanted:
        return 0
    world = ReplayWorld(window, load)
    solved = 0

    def finish(record: Dict, planner: str, rule: str, seconds: float) -> None:
        nonlocal solved
        record.update({"key": key(planner, rule), "study": study, "network": window,
                       "load": load, "world": world.describe(), "machine": ST.machine(),
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        ST.append(study, shard, record)
        done[record["key"]] = record
        solved += 1
        R._report(window, planner, rule, record, seconds)

    if key(*REFERENCE) not in done:
        t0 = time.perf_counter()
        finish(R.run_chain(world, *REFERENCE), *REFERENCE, time.perf_counter() - t0)
    budgets = [c["elapsed_s"] for c in done[key(*REFERENCE)]["per_cycle"]]
    for planner, rule in wanted:
        t0 = time.perf_counter()
        finish(R.run_chain(world, planner, rule, budgets=budgets), planner, rule,
               time.perf_counter() - t0)
    return solved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", default="moderate", choices=["moderate", "tight"])
    parser.add_argument("--windows", default=None, help="comma-separated, default all")
    parser.add_argument("--file-shard", type=int, default=0)
    parser.add_argument("--list", action="store_true", help="print the windows and stop")
    args = parser.parse_args()
    if args.list:
        for name, first in windows().items():
            world = ReplayWorld(name, args.load, n_cycles=R.N_CYCLES)
            print(name, first.date(), world.n, "containers, shift", world.shift_min,
                  "min, imputed", round(world.imputed_share, 3))
        return 0
    study = f"{STUDY_PREFIX}{args.load}"
    ST.write_meta(study, {"study": study, "load": R.LOADS[args.load],
                          "window_days": WINDOW_DAYS, "windows": {k: str(v.date())
                                                                  for k, v in windows().items()},
                          "shift_min_per_container": SHIFT_MIN_PER_CONTAINER,
                          "specs": [REFERENCE] + SPECS, "solver_config": PO.tuned_config(),
                          "environment": ST.environment()})
    done = ST.index(study)
    names = args.windows.split(",") if args.windows else list(windows())
    for name in names:
        t0 = time.perf_counter()
        n = run_window(args.load, name, args.file_shard, done)
        print(f"  {name}: {n} chains, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
