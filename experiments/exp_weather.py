"""
Adverse weather: what a spell does to deferral, and what a reading is worth.
============================================================================

The rollout applies a policy for forty cycles in nominal conditions.  This
experiment puts a six-day spell of adverse weather in the middle of the same
forty cycles, on the same networks and the same demand, and asks two things.

The first is whether the dispatch rule still does what it promises when the
fleet can reach fewer containers per shift.  Slower streets and slower handling
turn a network the fleet could keep up with into one it cannot, which is the
situation the waiting-time guarantee is for.

The second is what knowing the weather is worth.  A plan is made before the
shift and driven during it.  If it was made for dry streets and is driven in
rain, the vehicle runs out of shift before it runs out of stops, and stops are
lost whatever their priority.  Four modes are compared under the same weather:

``blind``
    The plan is made for nominal conditions.
``protected``
    The plan is made for nominal conditions, and the crew is told which stops
    are reserved.  While driving it gives up other stops as needed to keep
    those.
``current``
    The plan is made for the conditions at the dispatch instant, which is what a
    live feed of current conditions provides.
``forecast``
    The plan is made for the conditions over the coming shift, which is the most
    an hourly forecast could provide.  It differs from ``current`` only when the
    weather changes during the shift.

Whatever the plan assumed, it is then driven in the weather that actually
occurs, hour by hour.  The vehicle takes the stops in the planned order and
serves each one that still leaves time to return to the depot and tip inside
the shift.  A stop that does not is skipped.  The overtime that finishing the
whole plan would have needed is recorded beside what was skipped.

Scenarios
---------
Every scenario is a sequence of explicit conditions, so it can be rebuilt
without any weather service.  Cycles 16 to 27 carry the spell.

``rain``     Heavy rain in every shift.
``storm``    Heavy rain, with two days of violent rain and 150 mm of standing
             water in the middle.
``heat``     The mean hourly conditions of the six hottest days of April 2024 in
             Dhaka, which rise through the morning shift.
``onset``    Dry at dispatch, heavy rain from the second hour of each day shift.

Temperature and humidity come from the ERA5 reanalysis for Dhaka: July 2024 for
the rain scenarios and April 2024 for the heat scenario.  Rain is set by
intensity class and not replayed from the reanalysis, which spreads a tropical
downpour over a grid cell and holds no hour above 10 mm in that July.

What the planner is and is not told
-----------------------------------
The information levels concern travel and service time.  The gas readings and
hazard flags a planner receives are sensor readings and reflect the weather in
every mode.  The hours to overflow it is given use the mean fill rate of a
container in every mode, as in the rollout.

Run:  python -m experiments.exp_weather --scenario rain --shard 0 --of 4
      python -m experiments.exp_weather --describe
      python -m experiments.exp_weather --live
Out:  results/raw/weather-<scenario>.sNN.jsonl, results/weather_scenarios.json,
      results/weather_live.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import vrp as VRP             # noqa: E402
from wastebins_core import weather as WX          # noqa: E402
from experiments import exp_fleet as EF           # noqa: E402
from experiments import exp_rollout as RO         # noqa: E402
from experiments import instances as IN           # noqa: E402
from experiments import policies as PO            # noqa: E402
from experiments import store as ST               # noqa: E402
from experiments import weather_data as WD        # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

LOAD = "moderate"
#: All eight rollout networks.  A paired test over fewer networks cannot reach
#: the usual significance level whatever the data show.
NETWORKS = tuple(f"R{k}" for k in range(8))
SPELL = range(16, 28)             # cycles under the adverse condition
STORM_PEAK = range(20, 24)        # cycles with violent rain and standing water
SHIFT_HOURS = int(math.ceil(RO.SHIFT_MIN / 60.0))

#: Clock hours of the two shifts.  The congestion surface is evaluated at 07:30,
#: so a day shift is dispatched then and a night shift twelve hours later.
DAY_DISPATCH_HOUR, NIGHT_DISPATCH_HOUR = 7, 19

REFERENCE = ("insertion", "full")
SPECS: List[Tuple[str, str]] = [("insertion", "urgency"), ("ortools", "full")]
MODES: Dict[str, Tuple[str, ...]] = {
    "rain": ("blind", "protected", "current"),
    "storm": ("blind", "protected", "current"),
    "heat": ("blind", "protected", "current", "forecast"),
    "onset": ("blind", "protected", "current", "forecast"),
}
#: Modes in which the plan is made for nominal conditions.
PLANNED_BLIND = ("blind", "protected")

# Representative rates inside each class; only the class enters the model.
HEAVY_MM_H, VIOLENT_MM_H = 20.0, 60.0
STANDING_WATER_MM = 150.0

# Dhaka, July 2024, ERA5 through Open-Meteo.  Hours with at least 2.5 mm of
# rain: 28.2 C and 88 percent.  Dry hours of the morning shift: 30.4 C, 75
# percent, UV index 4.8.  At 07:00: 27.7 C, 87 percent.  Days with at least
# 10 mm: daily maximum 31.0 C, mean 27.9 C, mean humidity 88 percent.
MONSOON_RAIN_HOUR = (28.2, 88.0, 2.0)
MONSOON_DRY_HOUR = (30.4, 75.0, 4.8)
MONSOON_MORNING = (27.7, 87.0, 0.7)
MONSOON_DAY = {"daily_max_c": 31.0, "mean_c": 27.9, "humidity": 88.0}


@dataclass(frozen=True)
class CycleWeather:
    """The weather of one dispatch cycle."""

    #: What a feed of current conditions reports when the plan is made.
    dispatch: WX.Conditions
    #: The conditions of each hour of the shift, as they turn out.
    shift: Tuple[WX.Conditions, ...]
    #: Mean air temperature and humidity of the twelve hours before dispatch,
    #: which is what the sensors in a container have been exposed to.
    sensed_c: float = WX.REFERENCE_TEMPERATURE_C
    sensed_humidity: float = WX.REFERENCE_HUMIDITY
    #: Daily maximum temperature and whether the day is wet: the fill rate.
    daily_max_c: float = WX.REFERENCE_TEMPERATURE_C
    wet_day: bool = False

    @property
    def nominal(self) -> bool:
        return (all(WX.effects(h).neutral for h in self.shift)
                and WX.effects(self.dispatch).neutral
                and self.sensed_c == WX.REFERENCE_TEMPERATURE_C
                and self.daily_max_c == WX.REFERENCE_TEMPERATURE_C
                and not self.wet_day)


NOMINAL_CYCLE = CycleWeather(dispatch=WX.NOMINAL, shift=(WX.NOMINAL,) * SHIFT_HOURS)


def _hour(temperature: float, humidity: float, uv: float, day: bool,
          rain: float = 0.0, water: float = 0.0) -> WX.Conditions:
    return WX.Conditions(temperature_c=temperature, relative_humidity=humidity,
                         rain_mm_h=rain, uv_index=uv if day else 0.0,
                         is_daytime=day, standing_water_mm=water)


def _rain_cycle(day: bool, rate: float, water: float = 0.0) -> CycleWeather:
    hour = _hour(*MONSOON_RAIN_HOUR, day, rain=rate, water=water)
    return CycleWeather(dispatch=hour, shift=(hour,) * SHIFT_HOURS,
                        sensed_c=MONSOON_DAY["mean_c"],
                        sensed_humidity=MONSOON_DAY["humidity"],
                        daily_max_c=MONSOON_DAY["daily_max_c"], wet_day=True)


def _onset_cycle(day: bool) -> CycleWeather:
    if not day:
        return NOMINAL_CYCLE
    dry = _hour(*MONSOON_DRY_HOUR, True)
    wet = _hour(*MONSOON_RAIN_HOUR, True, rain=HEAVY_MM_H)
    return CycleWeather(dispatch=_hour(*MONSOON_MORNING, True),
                        shift=(dry,) + (wet,) * (SHIFT_HOURS - 1),
                        sensed_c=MONSOON_DAY["mean_c"],
                        sensed_humidity=MONSOON_DAY["humidity"],
                        daily_max_c=MONSOON_DAY["daily_max_c"], wet_day=True)


_HEAT: Optional[Dict[int, Dict[str, float]]] = None


def heat_profile() -> Dict[int, Dict[str, float]]:
    """Mean conditions by clock hour over the six hottest days of April 2024."""
    global _HEAT
    if _HEAT is None:
        frame = WD.hourly("dhaka", "2024-04-01", "2024-04-30")
        frame["date"] = frame["time"].dt.date
        frame["hour"] = frame["time"].dt.hour
        hottest = (frame.groupby("date")["temperature_2m"].max()
                   .sort_values(ascending=False).head(6))
        chosen = frame[frame["date"].isin(set(hottest.index))]
        by_hour = chosen.groupby("hour")[["temperature_2m", "relative_humidity_2m",
                                          "uv_index"]].mean()
        _HEAT = {int(h): {"t": float(r["temperature_2m"]),
                          "rh": float(r["relative_humidity_2m"]),
                          "uv": float(r["uv_index"])}
                 for h, r in by_hour.iterrows()}
        _HEAT[-1] = {"daily_max_c": float(hottest.mean()),
                     "days": [str(d) for d in sorted(hottest.index)]}
    return _HEAT


def _heat_cycle(day: bool) -> CycleWeather:
    profile = heat_profile()
    start = DAY_DISPATCH_HOUR if day else NIGHT_DISPATCH_HOUR

    def at(clock: int) -> WX.Conditions:
        row = profile[clock % 24]
        return _hour(row["t"], row["rh"], row["uv"], day)

    before = [profile[(start - k) % 24] for k in range(1, 13)]
    return CycleWeather(
        dispatch=at(start),
        shift=tuple(at(start + 1 + k) for k in range(SHIFT_HOURS)),
        sensed_c=float(np.mean([r["t"] for r in before])),
        sensed_humidity=float(np.mean([r["rh"] for r in before])),
        daily_max_c=profile[-1]["daily_max_c"], wet_day=False)


def sequence(scenario: str, n_cycles: int = RO.N_CYCLES) -> List[CycleWeather]:
    """The weather of every cycle of a scenario.  Even cycles are day shifts."""
    out: List[CycleWeather] = []
    for cycle in range(n_cycles):
        day = cycle % 2 == 0
        if cycle not in SPELL or scenario == "none":
            out.append(NOMINAL_CYCLE)
        elif scenario == "rain":
            out.append(_rain_cycle(day, HEAVY_MM_H))
        elif scenario == "storm":
            peak = cycle in STORM_PEAK
            out.append(_rain_cycle(day, VIOLENT_MM_H if peak else HEAVY_MM_H,
                                   STANDING_WATER_MM if peak else 0.0))
        elif scenario == "heat":
            out.append(_heat_cycle(day))
        elif scenario == "onset":
            out.append(_onset_cycle(day))
        else:
            raise KeyError(f"unknown scenario {scenario!r}")
    return out


# ---------------------------------------------------------------------------
# What the planner believes, and what happens
# ---------------------------------------------------------------------------
combine = WX.combine_effects


def believed(mode: str, weather: CycleWeather) -> WX.Effects:
    if mode in PLANNED_BLIND:
        return WX.NEUTRAL
    if mode == "current":
        return WX.effects(weather.dispatch)
    if mode == "forecast":
        return combine([WX.effects(h) for h in weather.shift])
    raise KeyError(f"unknown mode {mode!r}")


def conductor(world: "WeatherWorld", planner: str, rule_name: str, mode: str,
              config: Optional[Dict] = None):
    """The planning step of a cycle: plan on what is believed, drive in what occurs."""
    weights = VRP.ObjectiveWeights()
    rule = PO.RULES[rule_name]

    def conduct(cycle: int, snapshot: Dict, budget: Optional[float]):
        weather = world.sequence[cycle]
        actual = [WX.effects(h) for h in weather.shift]
        belief = believed(mode, weather)
        travel = WX.travel_under(world.travel, belief)
        planning = snapshot
        if belief.service_factor != 1.0:
            planning = dict(snapshot, service={
                nid: minutes * belief.service_factor
                for nid, minutes in snapshot["service"].items()})

        wall, cpu = time.perf_counter(), time.process_time()
        plan = PO.plan_with(planner, rule, planning, travel, world.fleet, weights,
                            budget_s=budget, improve_s=RO.IMPROVE_S, config=config,
                            tau_h=RO.TAU_H)
        elapsed, cpu_used = time.perf_counter() - wall, time.process_time() - cpu

        canonical = PO.canonical_tasks(snapshot, RO.TAU_H)
        heads = list(plan.metrics.get("head_ids", []))
        protect = heads if mode == "protected" else ()
        routes, dropped, overrun = [], [], 0.0
        for route in plan.routes:
            order = [canonical[s.task.node_id] for s in route.stops]
            if not order:
                continue
            driven, lost, over = WX.drive_route(order, route.vehicle, world.travel,
                                                actual, protect=protect)
            if driven.stops:
                routes.append(driven)
            dropped.extend(lost)
            overrun += over
        scored = PO.score_routes(routes, canonical, weights)

        served = set(scored["served_ids"])
        metrics = dict(plan.metrics)
        metrics["heads_served"] = sum(1 for nid in heads if nid in served)
        executed = VRP.FleetPlan(
            routes=routes,
            unserved=[t for nid, t in canonical.items() if nid not in served],
            objective=scored["objective"], metrics=metrics,
            compute_ms=plan.compute_ms, algorithm=plan.algorithm)
        mean_actual = combine(actual)
        extra = {
            "planned": int(sum(len(r.stops) for r in plan.routes)),
            "dropped": len(dropped),
            "dropped_heads": sum(1 for t in dropped if t.node_id in set(heads)),
            "dropped_hazards": sum(1 for t in dropped if t.hazard),
            "dropped_overdue": sum(1 for t in dropped if t.overdue),
            "overtime_to_finish_min": round(overrun, 2),
            "speed_factor_believed": round(belief.speed_factor, 4),
            "speed_factor_actual": round(mean_actual.speed_factor, 4),
            "service_factor_believed": round(belief.service_factor, 4),
            "service_factor_actual": round(mean_actual.service_factor, 4),
            "in_spell": cycle in SPELL,
        }
        return executed, scored, elapsed, cpu_used, extra

    return conduct


class WeatherWorld(RO.World):
    """The rollout world with the weather of a scenario laid over it."""

    def __init__(self, network: str, load: str, scenario: str,
                 n_cycles: int = RO.N_CYCLES):
        super().__init__(network, load, n_cycles=n_cycles)
        self.scenario = scenario
        self.sequence = sequence(scenario, n_cycles)
        self._litter = self.capacity_l <= 400.0

    def ambient(self, cycle: int) -> Tuple[float, float]:
        weather = self.sequence[cycle]
        return weather.sensed_c, weather.sensed_humidity

    def gas_factor(self, cycle: int) -> float:
        return WX.decomposition_factor(self.sequence[cycle].sensed_c)

    def demand_factor(self, cycle: int):
        weather = self.sequence[cycle]
        factor = WX.litter_demand_factor(WX.Conditions(
            daily_max_c=weather.daily_max_c, wet_day=weather.wet_day))
        # Measured on public litter bins; a large container keeps its rate.
        return np.where(self._litter, factor, 1.0)


def describe(scenario: str) -> Dict:
    """The factors each cycle type of a scenario produces, for the paper's table."""
    cycles = sequence(scenario)
    rows = {}
    for label, cycle in (("day", SPELL[0]), ("night", SPELL[0] + 1),
                         ("storm peak day", STORM_PEAK[0])):
        if label.startswith("storm") and scenario != "storm":
            continue
        weather = cycles[cycle]
        hours = [WX.effects(h) for h in weather.shift]
        mean = combine(hours)
        demand = WX.litter_demand_factor(WX.Conditions(
            daily_max_c=weather.daily_max_c, wet_day=weather.wet_day))
        rows[label] = {
            "dispatch": weather.dispatch.as_dict(),
            "shift_hours": [h.as_dict() for h in weather.shift],
            "rain_class": [e.rain_class for e in hours],
            "speed_factor": [round(e.speed_factor, 3) for e in hours],
            "speed_cap_kmh": [None if math.isinf(e.speed_cap_kmh)
                              else round(e.speed_cap_kmh, 2) for e in hours],
            "wbgt_c": [round(e.wbgt_c, 2) for e in hours],
            "sun_exposed": [e.sun_exposed for e in hours],
            "uv_category": [e.uv_category for e in hours],
            "work_capacity": [round(e.work_capacity, 3) for e in hours],
            "service_factor": [round(e.service_factor, 3) for e in hours],
            "service_factor_shift_mean": round(mean.service_factor, 3),
            "speed_factor_shift_mean": round(mean.speed_factor, 3),
            "service_factor_at_dispatch": round(WX.effects(weather.dispatch).service_factor, 3),
            "speed_factor_at_dispatch": round(WX.effects(weather.dispatch).speed_factor, 3),
            "sensed_temperature_c": round(weather.sensed_c, 2),
            "gas_factor": round(WX.decomposition_factor(weather.sensed_c), 3),
            "daily_max_c": round(weather.daily_max_c, 2),
            "wet_day": weather.wet_day,
            "litter_demand_factor": round(demand, 4),
        }
    return rows


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------
def key_of(scenario: str, network: str, planner: str, rule: str, mode: str) -> str:
    return f"weather-{scenario}|{network}|{planner}|{rule}|{mode}"


def spell_summary(chain: Dict) -> Dict:
    """What happened inside the spell and after it, from the per-cycle records."""
    cycles = chain["per_cycle"]
    spell = [c for i, c in enumerate(cycles) if i in SPELL]
    before = [c for i, c in enumerate(cycles) if RO.BURN_IN <= i < SPELL[0]]
    after = [c for i, c in enumerate(cycles) if i > SPELL[-1]]

    def mean(rows, field):
        return float(np.mean([r[field] for r in rows])) if rows else 0.0

    recovery = None
    level = max([c["overdue"] for c in before] or [0])
    for offset, c in enumerate(after):
        if c["overdue"] <= level:
            recovery = offset
            break
    return {
        "served_before": mean(before, "served"), "served_in_spell": mean(spell, "served"),
        "served_after": mean(after, "served"),
        "planned_in_spell": mean(spell, "planned"),
        "dropped_per_cycle_in_spell": mean(spell, "dropped"),
        "dropped_heads_in_spell": int(sum(c["dropped_heads"] for c in spell)),
        "dropped_hazards_in_spell": int(sum(c["dropped_hazards"] for c in spell)),
        "overtime_to_finish_min_in_spell": mean(spell, "overtime_to_finish_min"),
        "overflow_events_before": mean(before, "overflow_events"),
        "overflow_events_in_spell": mean(spell, "overflow_events"),
        "overflow_events_after": mean(after, "overflow_events"),
        "hazards_before": mean(before, "hazards"), "hazards_in_spell": mean(spell, "hazards"),
        "max_wait_before_h": float(max([c["max_wait_h"] for c in before] or [0.0])),
        "max_wait_in_spell_h": float(max([c["max_wait_h"] for c in spell] or [0.0])),
        "max_wait_after_h": float(max([c["max_wait_h"] for c in after] or [0.0])),
        "max_backlog_in_spell": int(max([c["overdue"] for c in spell] or [0])),
        "cycles_to_recover": recovery,
        "km_in_spell": mean(spell, "km"), "hours_in_spell": mean(spell, "hours"),
    }


def run_network(scenario: str, network: str, shard: int, done: Dict[str, Dict],
                n_cycles: int) -> int:
    study = f"weather-{scenario}"
    def specs(mode: str) -> List[Tuple[str, str]]:
        # Protection concerns reserved stops, so it is run under rules that
        # reserve some.  Under any other rule it is the blind mode again.
        return [(p, r) for p, r in [REFERENCE] + SPECS
                if mode != "protected" or PO.RULES[r].reserve > 0]

    wanted = [(p, r, m) for m in MODES[scenario] for p, r in specs(m)
              if key_of(scenario, network, p, r, m) not in done]
    if not wanted:
        return 0
    world = WeatherWorld(network, LOAD, scenario, n_cycles=n_cycles)
    solved = 0
    for mode in MODES[scenario]:
        ref_key = key_of(scenario, network, *REFERENCE, mode)
        for planner, rule in specs(mode):
            key = key_of(scenario, network, planner, rule, mode)
            if key in done:
                continue
            # Each budgeted planner receives, in every cycle, the time the
            # insertion planner took in that cycle under the same information.
            budgets = None
            if planner in PO.BUDGETED:
                budgets = [c["elapsed_s"] for c in done[ref_key]["per_cycle"]]
            t0 = time.perf_counter()
            record = RO.run_chain(world, planner, rule, budgets=budgets,
                                  conduct=conductor(world, planner, rule, mode))
            record.update({
                "key": key, "study": study, "scenario": scenario, "mode": mode,
                "network": network, "load": LOAD, "world": world.describe(),
                "spell": spell_summary(record),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            ST.append(study, shard, record)
            done[key] = record
            solved += 1
            s, w = record["summary"], record["spell"]
            print(f"  {network} {scenario:<6}{mode:<9}{planner:<10}{rule:<9} "
                  f"served {w['served_before']:4.1f}>{w['served_in_spell']:4.1f}  "
                  f"dropped {w['dropped_per_cycle_in_spell']:4.1f}  "
                  f"max wait {s['worst_wait_any_dispatch_h']:4.0f}  "
                  f"breach {s['bound_breached']}  heads lost {w['dropped_heads_in_spell']}  "
                  f"overflow {w['overflow_events_before']:.2f}>{w['overflow_events_in_spell']:.2f}  "
                  f"{(time.perf_counter() - t0) / 60:4.1f} min", flush=True)
    return solved


# ---------------------------------------------------------------------------
# Live demonstration
# ---------------------------------------------------------------------------
def live(snapshots: int = 5, budget_s: float = 30.0) -> Dict:
    """
    One dated run on live conditions.

    The conditions at the main depot are read from the Google feed and from
    Open-Meteo.  Each of the first snapshots of the primary sample is planned
    twice with OR-Tools, once for nominal conditions and once for the conditions
    read, and both plans are driven in the conditions read.

    The readings of the Google feed are not written to the result file, which
    its terms do not allow.  The file holds the class labels and the factors
    derived from them, and the Open-Meteo reading, which may be kept.
    """
    depot = IN.depots()[0]
    google = WX.GoogleWeatherProvider()
    open_meteo = WX.OpenMeteoProvider()
    readings = {"google": google.current(*depot), "open-meteo": open_meteo.current(*depot)}
    if google.fallbacks:
        print(f"Google feed unavailable: {google.last_error}")
    weights = VRP.ObjectiveWeights()
    out: Dict = {
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "site": {"latitude": depot[0], "longitude": depot[1], "name": "main depot, Dhaka"},
        "planner": "ortools", "rule": "full", "budget_s": budget_s,
        "sources": {},
    }
    for name, conditions in readings.items():
        effect = WX.effects(conditions)
        block = {
            "attribution": {"google": google.attribution,
                            "open-meteo": open_meteo.attribution}[name],
            "live": conditions.source == name,
            "effects": {
                "rain_class": effect.rain_class, "uv_category": effect.uv_category,
                "sun_exposed": effect.sun_exposed,
                "speed_factor": round(effect.speed_factor, 3),
                "service_factor": round(effect.service_factor, 2),
                "gas_factor": round(effect.gas_factor, 2),
                "litter_demand_factor": round(effect.litter_demand_factor, 2),
            },
            "snapshots": [],
        }
        if name == "open-meteo":
            block["conditions"] = conditions.as_dict()
        for index in range(snapshots):
            snapshot = IN.snapshots("S0", 25, 6)[index]
            base = EF.travel_for(snapshot, IN.WHEN)
            fleet = EF.build_fleet(snapshot)
            canonical = PO.canonical_tasks(snapshot)
            row = {"snapshot": index}
            for label, belief in (("nominal_plan", WX.NEUTRAL), ("live_plan", effect)):
                planning = dict(snapshot, service={
                    nid: m * belief.service_factor
                    for nid, m in snapshot["service"].items()})
                plan = PO.plan_with("ortools", PO.RULES["full"], planning,
                                    WX.travel_under(base, belief), fleet, weights,
                                    budget_s=budget_s, improve_s=0.0)
                routes, dropped, overrun = [], 0, 0.0
                for route in plan.routes:
                    order = [canonical[s.task.node_id] for s in route.stops]
                    if not order:
                        continue
                    driven, lost, over = WX.drive_route(order, route.vehicle,
                                                        base, [effect] * 8)
                    routes.append(driven)
                    dropped += len(lost)
                    overrun += over
                scored = PO.score_routes(routes, canonical, weights)
                row[label] = {
                    "planned": int(sum(len(r.stops) for r in plan.routes)),
                    "served": scored["metrics"]["bins_served"],
                    "dropped": dropped,
                    "overtime_to_finish_min": round(overrun, 1),
                    "km": scored["metrics"]["distance_km"],
                    "hours": round(scored["metrics"]["duration_min"] / 60.0, 2),
                    "objective": round(scored["objective"], 2),
                }
            block["snapshots"].append(row)
            print(f"  {name:<11} snapshot {index}: nominal plan served "
                  f"{row['nominal_plan']['served']} (dropped {row['nominal_plan']['dropped']}), "
                  f"live plan served {row['live_plan']['served']} "
                  f"(dropped {row['live_plan']['dropped']})", flush=True)
        out["sources"][name] = block
    out["google_requests"] = google.requests
    (RESULTS / "weather_live.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {RESULTS / 'weather_live.json'}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(MODES), default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--networks", default=None)
    parser.add_argument("--cycles", type=int, default=RO.N_CYCLES)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--live-snapshots", type=int, default=5)
    args = parser.parse_args()

    ST.keep_awake()
    if args.describe:
        payload = {
            "reference": {"temperature_c": WX.REFERENCE_TEMPERATURE_C,
                          "relative_humidity": WX.REFERENCE_HUMIDITY},
            "spell_cycles": [SPELL[0], SPELL[-1]],
            "storm_peak_cycles": [STORM_PEAK[0], STORM_PEAK[-1]],
            "heat_days": heat_profile()[-1],
            "weather_source": WD.ATTRIBUTION,
            "scenarios": {name: describe(name) for name in MODES},
        }
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "weather_scenarios.json").write_text(
            json.dumps(payload, indent=2, default=str))
        for name, rows in payload["scenarios"].items():
            for label, row in rows.items():
                print(f"{name:<6} {label:<15} speed {row['speed_factor']} cap "
                      f"{row['speed_cap_kmh'][0]}  service {row['service_factor']}  "
                      f"gas x{row['gas_factor']}  litter x{row['litter_demand_factor']}")
        print(f"wrote {RESULTS / 'weather_scenarios.json'}")
        return 0
    if args.live:
        live(args.live_snapshots)
        return 0
    if args.scenario is None:
        parser.error("give --scenario, --describe or --live")

    study = f"weather-{args.scenario}"
    networks = args.networks.split(",") if args.networks else list(NETWORKS)
    ST.RAW.mkdir(parents=True, exist_ok=True)
    (ST.RAW / f"{study}.meta.json").write_text(json.dumps({
        "study": study, "scenario": args.scenario, "modes": MODES[args.scenario],
        "load": RO.LOADS[LOAD], "networks": networks,
        "spell_cycles": [SPELL[0], SPELL[-1]],
        "cycle_h": RO.CYCLE_H, "n_cycles": args.cycles, "burn_in": RO.BURN_IN,
        "shift_min": RO.SHIFT_MIN, "tau_h": RO.TAU_H, "improve_s": RO.IMPROVE_S,
        "specs": [REFERENCE] + SPECS,
        "factors": describe(args.scenario),
        "rain_speed_factor": WX.RAIN_SPEED_FACTOR,
        "decomposition_q10": WX.DECOMPOSITION_Q10,
        "demand_coefficients": {"per_10C": WX.DEMAND_LOG_PER_10C,
                                "wet": WX.DEMAND_LOG_WET},
        "weather_source": WD.ATTRIBUTION,
        "solver_config": PO.tuned_config(),
        "environment": ST.environment(),
    }, indent=2, default=str))

    done = ST.index(study)
    mine = list(ST.mine(networks, args.shard, args.of))
    print(f"[{study}] shard {args.shard}/{args.of}: networks {mine}", flush=True)
    for network in mine:
        t0 = time.perf_counter()
        solved = run_network(args.scenario, network, args.shard, done, args.cycles)
        print(f"  {network}: {solved} chains, {(time.perf_counter() - t0) / 60:.1f} min",
              flush=True)
    print(f"[{study}] shard {args.shard} finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
