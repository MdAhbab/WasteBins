"""
Multi-cycle rollout with evolving demand.
=========================================

The single-cycle comparison cannot show what the paper is about.  Deferral is a
property of a policy applied cycle after cycle: a container is passed over
today, and again tomorrow, and the question is how long that can go on.  This
experiment applies each policy for forty consecutive dispatch cycles on the same
networks, under the same demand, and records what happens to every container.

The earlier rollout advanced only the waiting times.  Fill levels and the
hazardous set were frozen, so competition for capacity never changed and an
unserved container could not overflow.  Here every container fills at its own
rate between and during cycles, is emptied when a vehicle reaches it, and
overflows if nobody comes in time.  Hazard status follows the fill and gas
levels, so the hazard tier changes from cycle to cycle.

Demand model
------------
Each container has a mean fill rate, as a share of its capacity per hour::

    rate = (1 / fill_hours) * class_factor * epsilon * neighbourhood

``fill_hours`` sets the load level: the hours a typical large container takes to
fill.  ``class_factor`` is 1.2 for a litter bin and 1.0 for a large container.
``epsilon`` is the heterogeneity between containers.  It is not assumed: it is
resampled from the fill rates observed on the Wyndham containers, each divided
by their median, which span 0.21 to 1.84.  ``neighbourhood`` is one multiplier
per spatial cluster of containers, so that slow containers stand near other slow
containers, which is what a quiet neighbourhood is.

Within a cycle the rate is scaled by a day or night factor and by lognormal
noise, and with a small probability a container receives a sudden load, as when
a market stall is cleared.  The planner is told the current fill and the time to
overflow implied by the container's mean rate.  It is not told the noise or the
sudden loads.

Every policy faces the same draws.  The rates, the noise and the sudden loads of
a network are generated once from its seed and reused for every policy, so two
policies differ only in what they decided.

Run:  python -m experiments.exp_rollout --load moderate --shard 0 --of 4
      python -m experiments.exp_rollout --pilot
Out:  results/raw/rollout-<load>.sNN.jsonl
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import aging as AG            # noqa: E402
from wastebins_core import priority as PR         # noqa: E402
from wastebins_core import vrp as VRP             # noqa: E402
from experiments import exp_fleet as EF           # noqa: E402
from experiments import instances as IN           # noqa: E402
from experiments import policies as PO            # noqa: E402
from experiments import store as ST               # noqa: E402
from experiments import wyndham as WY             # noqa: E402

CYCLE_H = 12.0
N_CYCLES = 40
BURN_IN = 8
SHIFT_MIN = 240.0
TAU_H = AG.DEFAULT_TAU_H
#: Improvement time of the insertion planner per solve, after construction.
IMPROVE_S = 3.0

# Demand parameters that are assumptions rather than measurements.  Each is
# stated in the paper's demand table.
LITTER_BIN_FACTOR = 1.2          # a litter bin fills faster relative to its size
N_NEIGHBOURHOODS = 6
NEIGHBOURHOOD_SIGMA = 0.30       # spread of the log neighbourhood multiplier
DAY_FACTOR, NIGHT_FACTOR = 1.3, 0.7
RATE_NOISE_SIGMA = 0.15          # lognormal noise on the rate, per cycle
SUDDEN_LOAD_RATE_PER_H = 0.01    # Poisson rate of a sudden load
SUDDEN_LOAD_RANGE = (0.15, 0.45)  # its size, as a share of capacity
FILL_CAP = 1.30                  # an overflowing container still receives waste
GAS_NOISE, TEMP_NOISE, HUMIDITY_NOISE = 0.06, 1.5, 8.0

#: Load levels: hours a typical large container takes to fill, and the share of
#: containers held hazardous throughout.  The two demand levels were set on the
#: pilot network P0, never on the networks the results are reported on.  At 96
#: hours the fleet keeps up and 0.3 percent of container-cycles overflow.  At 48
#: hours it does not, and 6.9 percent do.
LOADS: Dict[str, Dict] = {
    "moderate": {"fill_hours": 96.0, "persistent_hazard_share": 0.0},
    "tight": {"fill_hours": 48.0, "persistent_hazard_share": 0.0},
    "hazard": {"fill_hours": 96.0, "persistent_hazard_share": 0.30},
}
#: Nominal ambient state.  The weather study replaces it cycle by cycle.
AMBIENT_TEMPERATURE_C, AMBIENT_HUMIDITY = 28.0, 60.0

REFERENCE = ("insertion", "full")
SPECS: List[Tuple[str, str]] = [
    ("insertion", "urgency"), ("insertion", "ageing"),
    ("insertion", "tier_bounded"), ("insertion", "tier_unbounded"),
    ("insertion", "reserve_bounded"),
    ("insertion", "full_r2"), ("insertion", "full_r4"), ("insertion", "full_r8"),
    ("insertion", "full_ageing"),
    ("ortools", "urgency"), ("ortools", "tier_unbounded"),
    ("ortools", "full"), ("ortools", "full_r4"),
    ("aco", "tier_unbounded"), ("genetic", "tier_unbounded"),
    ("risk_graph", "tier_unbounded"),
    ("static_sweep", "full"), ("threshold", "full"),
]
#: The tight load repeats the rungs that carry the argument and the two
#: scheduling baselines, on both planners.
SPECS_TIGHT: List[Tuple[str, str]] = [
    ("insertion", "urgency"), ("insertion", "ageing"),
    ("insertion", "tier_unbounded"), ("insertion", "full_r4"),
    ("ortools", "urgency"), ("ortools", "tier_unbounded"),
    ("ortools", "full"), ("ortools", "full_r4"),
    ("static_sweep", "full"), ("threshold", "full"),
]
#: The hazard run asks one question, whether a reserved head survives a hazard
#: tier that fills the shift, so it runs only the policies that answer it.
SPECS_HAZARD: List[Tuple[str, str]] = [
    ("insertion", "tier_unbounded"), ("insertion", "full_r4"),
    ("ortools", "tier_unbounded"), ("ortools", "full"),
]


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------
_RELATIVE_RATES: Optional[np.ndarray] = None


def wyndham_relative_rates() -> np.ndarray:
    """
    Observed fill rates of the Wyndham containers, each divided by their median.

    The rate of a container is its mean day-to-day rise in fill over consecutive
    days on which the fill did not fall.  A fall is a collection, and is left
    out.  A container with too few readings to estimate a rate is left out too.
    """
    global _RELATIVE_RATES
    if _RELATIVE_RATES is not None:
        return _RELATIVE_RATES
    import pandas as pd
    containers, fill = WY.load()
    wide = WY.fill_matrix(containers, fill)
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    rates = []
    for serial in wide.columns:
        series = wide[serial].dropna()
        if len(series) < 365:
            continue
        gap_days = pd.Series(series.index).diff().dt.days.values[1:]
        change = series.diff().values[1:]
        rises = change[(gap_days == 1) & (change >= 0)]
        if len(rises) and rises.mean() > 0:
            rates.append(float(rises.mean()))
    rates = np.array(sorted(rates))
    _RELATIVE_RATES = rates / np.median(rates)
    return _RELATIVE_RATES


def _neighbourhoods(lat: np.ndarray, lon: np.ndarray, k: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Spatial clusters of containers, by k-means on projected coordinates."""
    lat0 = math.radians(float(lat.mean()))
    points = np.column_stack([lon * math.cos(lat0) * 111.32, lat * 110.54])
    centres = points[rng.choice(len(points), size=k, replace=False)]
    labels = np.zeros(len(points), dtype=int)
    for iteration in range(50):
        distance = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new_labels = distance.argmin(axis=1)
        if iteration > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            if np.any(labels == c):
                centres[c] = points[labels == c].mean(axis=0)
    return labels


class World:
    """One network, its demand, and the travel model the planners share."""

    def __init__(self, network: str, load: str, n_cycles: int = N_CYCLES):
        self.network, self.load = network, load
        self.params = LOADS[load]
        frame = IN.network(network)
        EF.set_study("dhaka", n_bins=len(frame), n_vehicles=1)
        self.n = len(frame)
        self.n_cycles = n_cycles
        depot = IN.depots()[0]

        self.node_ids = [int(x) for x in frame["osm_id"]]
        self.coords = [depot] + [(float(r.latitude), float(r.longitude))
                                 for r in frame.itertuples()]
        self.index_of = {nid: i + 1 for i, nid in enumerate(self.node_ids)}
        self.capacity_l = np.array([float(x) for x in frame["capacity_l"]])
        self.streams = [str(x) for x in frame["stream"]]
        small = self.capacity_l <= 400.0

        # One generator per network and load.  It is consumed in a fixed order,
        # so the world a policy meets does not depend on which policy it is.
        seed = [IN.SEED, sum(ord(c) for c in network), len(network),
                int(self.params["fill_hours"]),
                int(100 * self.params["persistent_hazard_share"])]
        rng = np.random.default_rng(seed)

        self.organic = np.where(small, 0.22, 0.35)
        self.density = rng.uniform(180.0, 260.0, self.n)
        self.service_min = np.where(small, rng.uniform(1.5, 3.0, self.n),
                                    rng.uniform(2.5, 6.0, self.n))
        starts = rng.choice([0.0, 0.0, 0.0, 120.0, 180.0], size=self.n)
        lengths = rng.choice([300.0, 360.0, 480.0], size=self.n)
        self.windows = [(float(a), float(a + b)) for a, b in zip(starts, lengths)]

        relative = wyndham_relative_rates()
        self.epsilon = rng.choice(relative, size=self.n, replace=True)
        lat = np.array([c[0] for c in self.coords[1:]])
        lon = np.array([c[1] for c in self.coords[1:]])
        self.neighbourhood = _neighbourhoods(lat, lon, N_NEIGHBOURHOODS, rng)
        multipliers = np.exp(rng.normal(0.0, NEIGHBOURHOOD_SIGMA, N_NEIGHBOURHOODS))
        multipliers /= multipliers.mean()
        self.neighbourhood_factor = multipliers[self.neighbourhood]
        self.rate = ((1.0 / float(self.params["fill_hours"]))
                     * np.where(small, LITTER_BIN_FACTOR, 1.0)
                     * self.epsilon * self.neighbourhood_factor)

        share = float(self.params["persistent_hazard_share"])
        self.persistent_hazard = np.zeros(self.n, dtype=bool)
        if share > 0.0:
            chosen = rng.choice(self.n, size=int(round(share * self.n)), replace=False)
            self.persistent_hazard[chosen] = True

        shape = (n_cycles, self.n)
        self.rate_noise = np.exp(rng.normal(0.0, RATE_NOISE_SIGMA, shape))
        p_sudden = 1.0 - math.exp(-SUDDEN_LOAD_RATE_PER_H * CYCLE_H)
        self.sudden = rng.random(shape) < p_sudden
        self.sudden_time = rng.uniform(0.0, CYCLE_H, shape)
        self.sudden_size = rng.uniform(*SUDDEN_LOAD_RANGE, shape)
        self.gas_noise = rng.normal(0.0, GAS_NOISE, shape)
        self.temp_noise = rng.normal(0.0, TEMP_NOISE, shape)
        self.humidity_noise = rng.normal(0.0, HUMIDITY_NOISE, shape)

        probe = {"coords": self.coords, "node_ids": self.node_ids,
                 "streams": dict(zip(self.node_ids, self.streams)), "index": 0}
        self.travel = EF.travel_for(probe, IN.WHEN)
        self.fleet = EF.scarce_fleet(probe, shift_minutes=SHIFT_MIN)
        unservable = [nid for nid, (a, _b) in zip(self.node_ids, self.windows)
                      if a >= SHIFT_MIN]
        if unservable:
            raise ValueError(f"{len(unservable)} containers open after the shift ends")

    # -- ambient state: nominal here, set per cycle by the weather study -----
    def ambient(self, cycle: int) -> Tuple[float, float]:
        """Air temperature in degrees Celsius and relative humidity in percent."""
        return AMBIENT_TEMPERATURE_C, AMBIENT_HUMIDITY

    def gas_factor(self, cycle: int) -> float:
        """Factor on the gas a container gives off at a given fill."""
        return 1.0

    def demand_factor(self, cycle: int):
        """Factor on the fill rate, one number or one per container."""
        return 1.0

    def cycle_rate(self, cycle: int) -> np.ndarray:
        factor = DAY_FACTOR if cycle % 2 == 0 else NIGHT_FACTOR
        return self.rate * factor * self.rate_noise[cycle] * self.demand_factor(cycle)

    def observe(self, cycle: int, fill: np.ndarray, wait: np.ndarray) -> Dict:
        """The dispatch snapshot the planners receive at the start of a cycle."""
        seen = np.clip(fill, 0.0, FILL_CAP)
        air, moisture = self.ambient(cycle)
        gas = np.clip(self.organic * np.minimum(seen, 1.0) * self.gas_factor(cycle)
                      + self.gas_noise[cycle], 0.0, 1.0)
        temp = air + 6.0 * gas + self.temp_noise[cycle]
        humidity = np.clip(moisture + self.humidity_noise[cycle], 25.0, 100.0)
        fills, prizes, hazards, tto = {}, {}, {}, {}
        for i, nid in enumerate(self.node_ids):
            rule = PR.compute_priority({
                "waste_level": float(seen[i]), "gas_level": float(gas[i]),
                "temperature": float(temp[i]), "humidity": float(humidity[i]),
            }, policy=PR.POLICY_RENORMALISE)
            fills[nid] = float(seen[i])
            prizes[nid] = float(rule.score)
            hazards[nid] = bool(seen[i] >= 0.95 or gas[i] >= 0.62
                                or self.persistent_hazard[i])
            # Hours to overflow at the container's mean rate.  The planner is
            # not told this cycle's noise or a sudden load that is coming.
            tto[nid] = (0.0 if seen[i] >= 1.0
                        else float((1.0 - seen[i]) / max(self.rate[i], 1e-9)))
        return {
            "index": cycle, "coords": self.coords, "node_ids": self.node_ids,
            "index_of": self.index_of, "fills": fills, "prizes": prizes,
            "hazards": hazards, "tto": tto,
            "streams": dict(zip(self.node_ids, self.streams)),
            "capacities": dict(zip(self.node_ids, self.capacity_l.tolist())),
            "densities": dict(zip(self.node_ids, self.density.tolist())),
            "service": dict(zip(self.node_ids, self.service_min.tolist())),
            "windows": dict(zip(self.node_ids, self.windows)),
            "waits": {nid: float(wait[i]) for i, nid in enumerate(self.node_ids)},
        }

    def describe(self) -> Dict:
        hours_to_fill = 1.0 / self.rate
        return {
            "network": self.network, "load": self.load,
            "containers": self.n, "litter_bins": int((self.capacity_l <= 400).sum()),
            "fill_hours_parameter": float(self.params["fill_hours"]),
            "hours_to_fill": {"min": float(hours_to_fill.min()),
                              "p10": float(np.percentile(hours_to_fill, 10)),
                              "median": float(np.median(hours_to_fill)),
                              "p90": float(np.percentile(hours_to_fill, 90)),
                              "max": float(hours_to_fill.max())},
            "persistent_hazards": int(self.persistent_hazard.sum()),
            "demand_visits_per_cycle_at_70pct": float(
                (self.rate * CYCLE_H / 0.7).sum()),
        }


def _first_crossing(start: float, rate: float, duration: float,
                    sudden_at: Optional[float], sudden_size: float) -> Optional[float]:
    """
    Hours until the fill first reaches 1.0, or None if it does not in ``duration``.

    The fill rises linearly at ``rate`` and jumps by ``sudden_size`` at
    ``sudden_at`` when a sudden load falls inside the interval.
    """
    if start >= 1.0:
        return 0.0
    jump = sudden_at if sudden_at is not None and 0.0 <= sudden_at <= duration else None
    before = duration if jump is None else jump
    if rate > 0.0:
        t = (1.0 - start) / rate
        if t <= before:
            return t
    if jump is None:
        return None
    level = start + rate * jump + sudden_size
    if level >= 1.0:
        return jump
    if rate > 0.0:
        t = jump + (1.0 - level) / rate
        if t <= duration:
            return t
    return None


def _level_after(start: float, rate: float, duration: float,
                 sudden_at: Optional[float], sudden_size: float) -> float:
    level = start + rate * duration
    if sudden_at is not None and 0.0 <= sudden_at <= duration:
        level += sudden_size
    return float(min(FILL_CAP, level))


# ---------------------------------------------------------------------------
# One policy, forty cycles
# ---------------------------------------------------------------------------
def run_chain(world: World, planner: str, rule_name: str,
              budgets: Optional[List[float]] = None,
              config: Optional[Dict] = None, conduct=None) -> Dict:
    """
    Apply one policy for the whole horizon and record every container.

    ``conduct`` replaces the planning step of a cycle.  It is called with the
    cycle, the snapshot and the budget, and returns the plan as it was actually
    driven, its score, the wall and processor time of the planning, and any
    fields to add to the record of the cycle.  The weather study uses it to plan
    on one set of conditions and drive in another.  Left unset, the plan is
    driven exactly as it was made.
    """
    n, node_ids = world.n, world.node_ids
    position = {nid: i for i, nid in enumerate(node_ids)}
    weights = VRP.ObjectiveWeights()
    rule = PO.RULES[rule_name]

    fill = np.zeros(n)
    wait = np.zeros(n)
    overflowing = np.zeros(n, dtype=bool)

    per_cycle: List[Dict] = []
    services: List[List[float]] = []       # per service: cycle, container, wait, head
    wait_trace = np.zeros((world.n_cycles, n))
    overflow_events = np.zeros(n, dtype=int)
    overflow_hours = np.zeros(n)
    late_services = np.zeros(n, dtype=int)
    times_served = np.zeros(n, dtype=int)
    violations: List[str] = []
    head_failures = 0
    # Where a fixed round stopped, for a planner that reports it.  Only the
    # static sweep does, so that its round continues from shift to shift.
    round_next = None

    for cycle in range(world.n_cycles):
        snapshot = world.observe(cycle, fill, wait)
        if round_next is not None:
            snapshot["round_start"] = round_next
        wait_trace[cycle] = wait
        budget = budgets[cycle] if budgets else (
            8.0 if planner in PO.BUDGETED else None)
        if conduct is None:
            wall, cpu = time.perf_counter(), time.process_time()
            plan = PO.plan_with(planner, rule, snapshot, world.travel, world.fleet,
                                weights, budget_s=budget, improve_s=IMPROVE_S,
                                config=config, tau_h=TAU_H)
            elapsed, cpu_used = time.perf_counter() - wall, time.process_time() - cpu
            scored, extra = None, {}
        else:
            plan, scored, elapsed, cpu_used, extra = conduct(cycle, snapshot, budget)
        violations.extend(EF.audit_plan(plan, f"{planner}/{rule_name}@{cycle}"))
        round_next = plan.metrics.get("round_next")

        service_time: Dict[int, float] = {}
        for route in plan.routes:
            for stop in route.stops:
                service_time[position[stop.task.node_id]] = stop.start_service_min / 60.0

        search = plan.metrics
        reserved = int(search.get("heads_reserved", 0))
        heads = set(search.get("head_ids", []))
        if reserved and int(search.get("heads_served", 0)) != reserved:
            head_failures += 1

        rate = world.cycle_rate(cycle)
        new_events = late_now = 0
        hours_before = float(overflow_hours.sum())
        for i in range(n):
            sudden_at = float(world.sudden_time[cycle, i]) if world.sudden[cycle, i] else None
            size = float(world.sudden_size[cycle, i])
            served_at = service_time.get(i)
            first_span = CYCLE_H if served_at is None else served_at
            crossing = _first_crossing(fill[i], rate[i], first_span, sudden_at, size)
            if crossing is not None:
                if not overflowing[i]:
                    overflow_events[i] += 1
                    new_events += 1
                overflow_hours[i] += first_span - crossing
            if served_at is None:
                fill[i] = _level_after(fill[i], rate[i], CYCLE_H, sudden_at, size)
                overflowing[i] = fill[i] >= 1.0
                wait[i] += CYCLE_H
                continue
            # Reached after it overflowed, in this cycle or an earlier one.
            if crossing is not None:
                late_services[i] += 1
                late_now += 1
            times_served[i] += 1
            services.append([cycle, i, float(wait[i]),
                             1 if node_ids[i] in heads else 0])
            rest = CYCLE_H - served_at
            later = None if sudden_at is None or sudden_at <= served_at \
                else sudden_at - served_at
            again = _first_crossing(0.0, rate[i], rest, later, size)
            if again is not None:
                overflow_events[i] += 1
                new_events += 1
                overflow_hours[i] += rest - again
            fill[i] = _level_after(0.0, rate[i], rest, later, size)
            overflowing[i] = fill[i] >= 1.0
            wait[i] = CYCLE_H

        overdue_before = int((wait_trace[cycle] >= TAU_H).sum())
        if scored is None:
            scored = PO.score(plan, PO.canonical_tasks(snapshot, TAU_H),
                              world.travel, weights)
        per_cycle.append({
            "served": len(service_time),
            "km": scored["metrics"]["distance_km"],
            "co2_kg": scored["metrics"]["co2_kg"],
            "hours": round(scored["metrics"]["duration_min"] / 60.0, 3),
            "objective": round(scored["objective"], 3),
            "overdue": overdue_before,
            "hazards": int(sum(snapshot["hazards"].values())),
            "hazard_response_h": scored["metrics"]["mean_hazard_response_h"],
            "hazard_served_pct": scored["metrics"]["hazard_coverage_pct"],
            "heads_reserved": reserved,
            "heads_served": int(search.get("heads_served", 0)),
            "head_repairs": int(search.get("head_repairs", 0)),
            "max_wait_h": float(wait_trace[cycle].max()),
            "overflow_events": int(new_events),
            "overflow_hours": round(float(overflow_hours.sum()) - hours_before, 3),
            "late_services": int(late_now),
            "elapsed_s": round(elapsed, 3),
            "cpu_s": round(cpu_used, 3),
            "iterations": search.get("iterations", search.get("generations")),
            "decode_dropped": search.get("decode_dropped"),
        })
        per_cycle[-1].update(extra)

    after = slice(BURN_IN, world.n_cycles)
    counted = [s for s in services if s[0] >= BURN_IN]
    service_waits = np.array([s[2] for s in counted]) if counted else np.array([0.0])
    backlog = max(c["overdue"] for c in per_cycle)
    # The `r` of the bound is the number of heads the dispatcher is certain to
    # serve.  A cycle with fewer overdue containers than `r` reserves all of
    # them, which is the rule working and not the rate falling, so such a cycle
    # counts as `r`.  Only a cycle that reserved fewer than it was asked for
    # *and* fewer than were overdue lowers the rate.
    effective = [rule.reserve if c["heads_reserved"] >= min(rule.reserve, c["overdue"])
                 else c["heads_reserved"]
                 for c in per_cycle if c["overdue"] > 0]
    min_reserved = min(effective) if effective and rule.reserve > 0 else None
    never = [i for i in range(n) if times_served[i] == 0]
    bound_known = bound_backlog = None
    if rule.reserve > 0 and min_reserved:
        bound_known = AG.queue_wait_bound(TAU_H, CYCLE_H, n, min_reserved)
        bound_backlog = AG.queue_wait_bound(TAU_H, CYCLE_H, backlog, min_reserved)
    worst_wait = float(wait_trace.max())
    return {
        "planner": planner, "rule": rule_name,
        "n_cycles": world.n_cycles, "burn_in": BURN_IN,
        "per_cycle": per_cycle,
        "services": services,
        "final_wait_h": wait.tolist(),
        "times_served": times_served.tolist(),
        "overflow_events": overflow_events.tolist(),
        "overflow_hours": [round(float(x), 3) for x in overflow_hours],
        "late_services": late_services.tolist(),
        "violations": violations[:20],
        "summary": {
            "wait_at_service_h": {
                "mean": float(service_waits.mean()),
                "median": float(np.median(service_waits)),
                "p95": float(np.percentile(service_waits, 95)),
                "p99": float(np.percentile(service_waits, 99)),
                "max": float(service_waits.max()),
                "n": int(len(counted)),
            },
            "worst_wait_any_dispatch_h": worst_wait,
            "worst_wait_after_burn_in_h": float(wait_trace[after].max()),
            "never_collected": len(never),
            "containers_ever_overdue": int((wait_trace >= TAU_H).any(axis=0).sum()),
            "max_backlog": int(backlog),
            "min_heads_reserved": min_reserved,
            "head_failures": int(head_failures),
            "head_repairs": int(sum(c["head_repairs"] for c in per_cycle)),
            "bound_known_in_advance_h": bound_known,
            "bound_at_observed_backlog_h": bound_backlog,
            "bound_breached": (None if bound_known is None
                               else bool(worst_wait > bound_backlog + 1e-9)),
            "overflow_events_per_container_cycle": float(
                sum(c["overflow_events"] for c in per_cycle[BURN_IN:])
                / (n * (world.n_cycles - BURN_IN))),
            "overflow_hours_per_container_cycle": float(
                sum(c["overflow_hours"] for c in per_cycle[BURN_IN:])
                / (n * (world.n_cycles - BURN_IN))),
            "late_services_share": float(
                sum(c["late_services"] for c in per_cycle[BURN_IN:])
                / max(1, sum(c["served"] for c in per_cycle[BURN_IN:]))),
            "served_per_cycle": float(np.mean([c["served"] for c in per_cycle[BURN_IN:]])),
            "km_per_cycle": float(np.mean([c["km"] for c in per_cycle[BURN_IN:]])),
            "co2_per_cycle": float(np.mean([c["co2_kg"] for c in per_cycle[BURN_IN:]])),
            "shift_use": float(np.mean([c["hours"] for c in per_cycle[BURN_IN:]])
                               * 60.0 / SHIFT_MIN),
            "objective_per_cycle": float(np.mean([c["objective"] for c in per_cycle[BURN_IN:]])),
            "hazard_response_h": float(np.mean(
                [c["hazard_response_h"] for c in per_cycle[BURN_IN:] if c["hazards"]] or [0.0])),
            "hazard_served_pct": float(np.mean(
                [c["hazard_served_pct"] for c in per_cycle[BURN_IN:] if c["hazards"]] or [100.0])),
            "elapsed_per_solve_s": float(np.mean([c["elapsed_s"] for c in per_cycle])),
            "feasible": len(violations) == 0,
        },
    }


def key_of(load: str, network: str, planner: str, rule: str) -> str:
    return f"rollout-{load}|{network}|{planner}|{rule}"


def run_network(load: str, network: str, shard: int, done: Dict[str, Dict],
                specs: List[Tuple[str, str]], n_cycles: int) -> int:
    study = f"rollout-{load}"
    ref_key = key_of(load, network, *REFERENCE)
    wanted = [s for s in specs if key_of(load, network, *s) not in done]
    if ref_key in done and not wanted:
        return 0
    world = World(network, load, n_cycles=n_cycles)
    solved = 0

    def stamp(record: Dict, planner: str, rule: str) -> Dict:
        record.update({"key": key_of(load, network, planner, rule),
                       "study": study, "network": network, "load": load,
                       "world": world.describe(),
                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return record

    if ref_key not in done:
        t0 = time.perf_counter()
        record = stamp(run_chain(world, *REFERENCE), *REFERENCE)
        ST.append(study, shard, record)
        done[ref_key] = record
        solved += 1
        _report(network, *REFERENCE, record, time.perf_counter() - t0)
    # Each budgeted planner receives, in every cycle, the time the insertion
    # planner took in that cycle of this network.
    budgets = [c["elapsed_s"] for c in done[ref_key]["per_cycle"]]
    for planner, rule in wanted:
        t0 = time.perf_counter()
        record = stamp(run_chain(world, planner, rule, budgets=budgets), planner, rule)
        ST.append(study, shard, record)
        done[record["key"]] = record
        solved += 1
        _report(network, planner, rule, record, time.perf_counter() - t0)
    return solved


def _report(network: str, planner: str, rule: str, record: Dict, seconds: float) -> None:
    s = record["summary"]
    w = s["wait_at_service_h"]
    print(f"  {network} {planner:<12}{rule:<16} wait mean {w['mean']:5.1f} p95 {w['p95']:5.0f} "
          f"max {s['worst_wait_any_dispatch_h']:5.0f}  never {s['never_collected']:2d}  "
          f"overflow {100 * s['overflow_events_per_container_cycle']:5.2f}%  "
          f"served {s['served_per_cycle']:4.1f}  km {s['km_per_cycle']:5.1f}  "
          f"use {100 * s['shift_use']:3.0f}%  {seconds / 60:4.1f} min", flush=True)


def pilot(fill_hours: List[float], networks: List[str], n_cycles: int) -> None:
    """Find demand levels on the pilot networks.  Nothing is stored."""
    print(f"{'fill h':>8}{'network':>9}{'served':>8}{'use %':>7}{'overflow %':>12}"
          f"{'mean wait':>11}{'max wait':>10}{'backlog':>9}{'s/solve':>9}")
    for hours in fill_hours:
        LOADS["pilot"] = {"fill_hours": hours, "persistent_hazard_share": 0.0}
        for network in networks:
            world = World(network, "pilot", n_cycles=n_cycles)
            for planner, rule in (REFERENCE, ("insertion", "urgency")):
                s = run_chain(world, planner, rule)["summary"]
                print(f"{hours:>8.0f}{network:>9}{s['served_per_cycle']:>8.1f}"
                      f"{100 * s['shift_use']:>7.0f}"
                      f"{100 * s['overflow_events_per_container_cycle']:>12.2f}"
                      f"{s['wait_at_service_h']['mean']:>11.1f}"
                      f"{s['worst_wait_any_dispatch_h']:>10.0f}{s['max_backlog']:>9}"
                      f"{s['elapsed_per_solve_s']:>9.1f}   {rule}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", default="moderate", choices=sorted(LOADS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--networks", default=None,
                        help="comma-separated network names, default R0 to R7")
    parser.add_argument("--cycles", type=int, default=N_CYCLES)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-fill-hours", default="144,96,72,48,36")
    args = parser.parse_args()

    ST.keep_awake()
    if args.pilot:
        pilot([float(x) for x in args.pilot_fill_hours.split(",")],
              (args.networks or "P0,P1").split(","), args.cycles)
        return 0

    study = f"rollout-{args.load}"
    networks = (args.networks.split(",") if args.networks
                else [f"R{k}" for k in range(8)])
    specs = {"hazard": SPECS_HAZARD, "tight": SPECS_TIGHT}.get(args.load, SPECS)
    ST.RAW.mkdir(parents=True, exist_ok=True)
    ST.write_meta(study, {
        "study": study, "load": LOADS[args.load],
        "cycle_h": CYCLE_H, "n_cycles": args.cycles, "burn_in": BURN_IN,
        "shift_min": SHIFT_MIN, "tau_h": TAU_H, "improve_s": IMPROVE_S,
        "demand": {
            "litter_bin_factor": LITTER_BIN_FACTOR,
            "neighbourhoods": N_NEIGHBOURHOODS,
            "neighbourhood_sigma": NEIGHBOURHOOD_SIGMA,
            "day_factor": DAY_FACTOR, "night_factor": NIGHT_FACTOR,
            "rate_noise_sigma": RATE_NOISE_SIGMA,
            "sudden_load_rate_per_h": SUDDEN_LOAD_RATE_PER_H,
            "sudden_load_range": SUDDEN_LOAD_RANGE, "fill_cap": FILL_CAP,
            "wyndham_relative_rates": wyndham_relative_rates().round(4).tolist(),
        },
        "specs": [REFERENCE] + specs,
        "solver_config": PO.tuned_config(),
        "environment": ST.environment(),
    })

    done = ST.index(study)
    mine = list(ST.mine(networks, args.shard, args.of))
    print(f"[{study}] shard {args.shard}/{args.of}: networks {mine}", flush=True)
    for network in mine:
        t0 = time.perf_counter()
        solved = run_network(args.load, network, args.shard, done, specs, args.cycles)
        print(f"  {network}: {solved} chains, {(time.perf_counter() - t0) / 60:.1f} min",
              flush=True)
    print(f"[{study}] shard {args.shard} finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
