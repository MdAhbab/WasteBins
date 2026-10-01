"""
Weather as an operating condition.
==================================

A dispatch plan is built for nominal conditions: dry streets, a mild day, a crew
working at its usual pace.  Weather changes each of those, and this module is
the one place where a weather reading is turned into the quantities the planner
and the demand model actually use.

It has three parts.

``Conditions``
    What the weather is: air temperature, relative humidity, rain rate, UV
    index, daylight, and the depth of standing water on the streets.

``effects``
    What those conditions do, as five numbers: a factor on travel speed, a cap
    on travel speed, a factor on the service time at a container, a factor on
    the gas a container gives off, and a factor on the fill rate of a public
    litter bin.

Providers
    Where a reading comes from.  ``GoogleWeatherProvider`` reads current
    conditions from the Google Maps Platform Weather API and is the live input
    of a running system.  ``OpenMeteoProvider`` reads the same quantities from
    an open service whose data may be archived.  ``StaticWeatherProvider``
    returns fixed conditions and is what the experiments and the tests use.

Where each number comes from
----------------------------
Rain and travel speed.  Rain is classed by rate as slight, moderate, heavy or
violent, following the criteria of the World Meteorological Organization
(WMO-No. 8, 2023 edition, Volume I, Chapter 14, annex).  The speed factor of
each class lies inside two measured ranges.  In London, travel time rose by 0.1
to 2.1 percent in light rain, 1.5 to 3.8 percent in moderate rain and 4.0 to
6.0 percent in heavy rain (Tsapakis, Cheng and Bolbol, 2013,
doi:10.1016/j.jtrangeo.2012.11.003).  In Shenzhen, speed fell by 1.2 to 18.4
percent in rain and by 21.5 to 24.3 percent on waterlogged segments (Hu, Lin,
Xie, Dai and Qui, 2018, doi:10.1109/ITSC.2018.8569639).

Standing water and travel speed.  Pregnolato, Ford, Wilkinson and Dawson (2017,
doi:10.1016/j.trd.2017.06.020) fitted the highest speed at which a vehicle keeps
control in water of depth ``w`` millimetres, ``0.0009 w^2 - 0.5529 w + 86.9448``
kilometres per hour, and place the limit of a passable road at 300 mm.

Heat and service time.  Physical work capacity falls with the wet-bulb globe
temperature (WBGT) as ``1 / (1 + (33.63 / WBGT)^-6.33)``, fitted to 338 work
sessions (Foster et al., 2021, doi:10.1007/s00484-021-02105-0).  The same
function holds under solar load when the WBGT includes it (Foster et al., 2022,
doi:10.1007/s00484-021-02205-x).  In shade the WBGT is ``0.7 Tw + 0.3 Ta``, and
work in the sun adds about 3 C to it at the hottest hours (Hyatt, Lemke and
Kjellstrom, 2010, doi:10.3402/gha.v3i0.5715).  The wet-bulb temperature ``Tw``
follows Stull (2011, doi:10.1175/JAMC-D-11-0143.1).  The service time at a
container is the nominal time divided by the work capacity relative to the
reference condition.

UV index and solar load.  No published relation ties the UV index to work
output.  It is used here as the sign of sun-exposed work: a daytime index in the
World Health Organization category "high" or above (6 or more; Global Solar UV
Index: A Practical Guide, 2002) switches the solar addition on.

Temperature and gas.  Food waste released 28 times more volatile organic
compounds at 35 C than at 10 C (Cui, Zhang, Zhang, Lv and Xie, 2022,
doi:10.1016/j.eti.2022.102443), a factor of 3.79 per 10 C.  The gas reading of a
container scales by that factor from the reference temperature, within the
measured range of 10 to 35 C.

Weather and fill rate.  Estimated on the 966-day Wyndham record by
``experiments/exp_weather_demand.py``: with container, weekday, month and
restriction-period effects, the fill rate of a public container rises by 10.8
percent per 10 C of daily maximum temperature and falls by 8.6 percent on a wet
day.  The factor applies to public litter bins only.

What is assumed rather than measured is stated in the paper: the speed factors
are measured in other cities, the standing water is applied to the whole
network at one depth, and the work-capacity function is taken from paced work in
a climate chamber.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone as dt_timezone
from typing import Dict, List, Optional, Sequence, Tuple

from . import emissions as EM
from . import vrp as VRP

# ---------------------------------------------------------------------------
# Reference condition: the nominal ambient state of the demand model
# ---------------------------------------------------------------------------
REFERENCE_TEMPERATURE_C = 28.0
REFERENCE_HUMIDITY = 60.0

# ---------------------------------------------------------------------------
# Rain
# ---------------------------------------------------------------------------
#: Below this rate a reading is treated as dry; it is the resolution of a
#: tipping-bucket gauge.
DRY_BELOW_MM_H = 0.1
#: Lower bounds of the rain classes, in millimetres per hour (WMO-No. 8).
RAIN_CLASS_BOUNDS: Tuple[Tuple[str, float], ...] = (
    ("violent", 50.0), ("heavy", 10.0), ("moderate", 2.5), ("slight", DRY_BELOW_MM_H),
)
#: Factor on the speed of every leg, by rain class.
RAIN_SPEED_FACTOR: Dict[str, float] = {
    "dry": 1.00, "slight": 0.98, "moderate": 0.95, "heavy": 0.88, "violent": 0.82,
}
#: The severe end of the published ranges, for the sensitivity run.  The largest
#: fall in speed measured in rain is 18.4 percent, so both upper classes take it.
RAIN_SPEED_FACTOR_SEVERE: Dict[str, float] = {
    "dry": 1.00, "slight": 0.95, "moderate": 0.90, "heavy": 0.816, "violent": 0.816,
}

# ---------------------------------------------------------------------------
# Standing water
# ---------------------------------------------------------------------------
FLOOD_A, FLOOD_B, FLOOD_C = 0.0009, -0.5529, 86.9448
IMPASSABLE_DEPTH_MM = 300.0
#: A leg is never timed below this speed.
MIN_SPEED_KMH = 1.0

# ---------------------------------------------------------------------------
# Heat
# ---------------------------------------------------------------------------
PWC_WBGT_MIDPOINT_C = 33.63
PWC_WBGT_EXPONENT = -6.33
PWC_WBGT_RANGE_C = (12.0, 40.0)        # range of the fitted function
SUN_WBGT_ADDITION_C = 3.0
UV_SUN_EXPOSED = 6.0                   # WHO category "high" starts here
UV_CATEGORIES: Tuple[Tuple[str, float], ...] = (
    ("extreme", 11.0), ("very high", 8.0), ("high", 6.0), ("moderate", 3.0), ("low", 0.0),
)

# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------
#: 28-fold over 25 C, expressed per 10 C.
DECOMPOSITION_Q10 = 28.0 ** (10.0 / 25.0)
DECOMPOSITION_RANGE_C = (10.0, 35.0)   # range of the measurement

# ---------------------------------------------------------------------------
# Fill rate of a public litter bin
# ---------------------------------------------------------------------------
DEMAND_LOG_PER_10C = 0.1025            # results/weather_demand.json, month effects
DEMAND_LOG_WET = -0.0903
DEMAND_TEMPERATURE_RANGE_C = (9.5, 37.8)   # range observed in that record
WET_DAY_MM_H = DRY_BELOW_MM_H


@dataclass(frozen=True)
class Conditions:
    """The weather at one place and time."""

    temperature_c: float = REFERENCE_TEMPERATURE_C
    relative_humidity: float = REFERENCE_HUMIDITY     # percent
    rain_mm_h: float = 0.0
    uv_index: float = 0.0
    is_daytime: bool = True
    #: Depth of water standing on the streets.  No weather feed reports it; it
    #: is a scenario input or an operator's report.
    standing_water_mm: float = 0.0
    #: Daily maximum temperature and whether the day is wet, which drive the
    #: fill rate.  Left unset they are read from the fields above.
    daily_max_c: Optional[float] = None
    wet_day: Optional[bool] = None
    observed_at: Optional[datetime] = None
    source: str = "scenario"

    def as_dict(self) -> Dict:
        out = asdict(self)
        if self.observed_at is not None:
            out["observed_at"] = self.observed_at.isoformat()
        return out


NOMINAL = Conditions(source="nominal")


@dataclass(frozen=True)
class Effects:
    """What a set of conditions does to the quantities the planner uses."""

    rain_class: str = "dry"
    speed_factor: float = 1.0
    speed_cap_kmh: float = math.inf
    passable: bool = True
    wet_bulb_c: float = 0.0
    wbgt_c: float = 0.0
    sun_exposed: bool = False
    uv_category: str = "low"
    work_capacity: float = 1.0
    service_factor: float = 1.0
    gas_factor: float = 1.0
    litter_demand_factor: float = 1.0

    @property
    def neutral(self) -> bool:
        return (self.speed_factor == 1.0 and math.isinf(self.speed_cap_kmh)
                and self.service_factor == 1.0 and self.gas_factor == 1.0
                and self.litter_demand_factor == 1.0)

    def as_dict(self) -> Dict:
        out = asdict(self)
        if math.isinf(self.speed_cap_kmh):
            out["speed_cap_kmh"] = None
        return out

    def summary(self) -> Dict:
        """
        The factors a plan was built on, coarse enough to keep.

        A stored plan records these and not the reading they came from, because
        a reading from a commercial feed may not be kept.
        """
        return {
            "rain_class": self.rain_class,
            "uv_category": self.uv_category,
            "sun_exposed": self.sun_exposed,
            "speed_factor": round(self.speed_factor, 3),
            "speed_cap_kmh": (None if math.isinf(self.speed_cap_kmh)
                              else round(self.speed_cap_kmh, 1)),
            "service_factor": round(self.service_factor, 2),
            "gas_factor": round(self.gas_factor, 2),
            "litter_demand_factor": round(self.litter_demand_factor, 2),
            "passable": self.passable,
        }


NEUTRAL = Effects()


# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------
def rain_class(rain_mm_h: float) -> str:
    rate = max(0.0, float(rain_mm_h))
    for name, lower in RAIN_CLASS_BOUNDS:
        if rate >= lower:
            return name
    return "dry"


def flood_speed_cap_kmh(depth_mm: float) -> float:
    """Highest safe speed in standing water; zero when the road is impassable."""
    w = float(depth_mm)
    if w <= 0.0:
        return math.inf
    if w >= IMPASSABLE_DEPTH_MM:
        return 0.0
    return max(0.0, FLOOD_A * w * w + FLOOD_B * w + FLOOD_C)


def wet_bulb_c(temperature_c: float, relative_humidity: float) -> float:
    """Wet-bulb temperature at sea-level pressure (Stull, 2011)."""
    t = float(temperature_c)
    rh = min(99.0, max(5.0, float(relative_humidity)))
    return (t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
            + math.atan(t + rh) - math.atan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
            - 4.686035)


def uv_category(uv_index: float) -> str:
    for name, lower in UV_CATEGORIES:
        if float(uv_index) >= lower:
            return name
    return "low"


def sun_exposed(conditions: Conditions) -> bool:
    return bool(conditions.is_daytime) and float(conditions.uv_index) >= UV_SUN_EXPOSED


def wbgt_c(conditions: Conditions) -> float:
    """Wet-bulb globe temperature: the shade value, plus the solar addition."""
    shade = (0.7 * wet_bulb_c(conditions.temperature_c, conditions.relative_humidity)
             + 0.3 * float(conditions.temperature_c))
    return shade + (SUN_WBGT_ADDITION_C if sun_exposed(conditions) else 0.0)


def work_capacity(wbgt: float) -> float:
    """Physical work capacity as a share of that in cool conditions (Foster et al., 2021)."""
    x = min(PWC_WBGT_RANGE_C[1], max(PWC_WBGT_RANGE_C[0], float(wbgt)))
    return 1.0 / (1.0 + (PWC_WBGT_MIDPOINT_C / x) ** PWC_WBGT_EXPONENT)


def work_capacity_guideline(wbgt: float) -> float:
    """
    Labour capacity under rest guidelines (Dunne, Stouffer and John, 2013,
    doi:10.1038/nclimate1827).  It describes how much work the guidelines allow,
    which is less than a crew delivers, and is kept for the sensitivity run.
    """
    return max(0.10, 1.0 - 0.25 * max(0.0, float(wbgt) - 25.0) ** (2.0 / 3.0))


def decomposition_factor(temperature_c: float, q10: float = DECOMPOSITION_Q10) -> float:
    t = min(DECOMPOSITION_RANGE_C[1], max(DECOMPOSITION_RANGE_C[0], float(temperature_c)))
    return float(q10) ** ((t - REFERENCE_TEMPERATURE_C) / 10.0)


def litter_demand_factor(conditions: Conditions) -> float:
    """Factor on the fill rate of a public litter bin, from the Wyndham record."""
    tmax = conditions.daily_max_c
    if tmax is None:
        tmax = conditions.temperature_c
    tmax = min(DEMAND_TEMPERATURE_RANGE_C[1], max(DEMAND_TEMPERATURE_RANGE_C[0], float(tmax)))
    wet = conditions.wet_day
    if wet is None:
        wet = float(conditions.rain_mm_h) >= WET_DAY_MM_H
    return math.exp(DEMAND_LOG_PER_10C * (tmax - REFERENCE_TEMPERATURE_C) / 10.0
                    + (DEMAND_LOG_WET if wet else 0.0))


def effects(conditions: Conditions, reference: Conditions = NOMINAL,
            capacity=work_capacity, q10: float = DECOMPOSITION_Q10,
            rain_speed: Optional[Dict[str, float]] = None) -> Effects:
    """Turn a weather reading into the factors the planner and the demand model use."""
    klass = rain_class(conditions.rain_mm_h)
    cap = flood_speed_cap_kmh(conditions.standing_water_mm)
    wbgt = wbgt_c(conditions)
    capacity_now = capacity(wbgt)
    capacity_ref = capacity(wbgt_c(reference))
    return Effects(
        rain_class=klass,
        speed_factor=float((rain_speed or RAIN_SPEED_FACTOR)[klass]),
        speed_cap_kmh=cap,
        passable=cap > 0.0,
        wet_bulb_c=wet_bulb_c(conditions.temperature_c, conditions.relative_humidity),
        wbgt_c=wbgt,
        sun_exposed=sun_exposed(conditions),
        uv_category=uv_category(conditions.uv_index),
        work_capacity=capacity_now,
        service_factor=capacity_ref / capacity_now,
        gas_factor=(decomposition_factor(conditions.temperature_c, q10)
                    / decomposition_factor(reference.temperature_c, q10)),
        litter_demand_factor=litter_demand_factor(conditions) / litter_demand_factor(reference),
    )


def combine_effects(hourly: Sequence[Effects]) -> Effects:
    """
    One set of effects standing for several hours, for a plan that has one.

    Travel time adds up over the hours, so the speed factor is the harmonic
    mean.  Service time is the plain mean.  A cap that applies in any hour is
    kept, which is the cautious reading.
    """
    if not hourly:
        return NEUTRAL
    n = float(len(hourly))

    def mean(values: List[float], harmonic: bool = False) -> float:
        # Hours that agree give their common value exactly, so that a plan made
        # for unchanging weather is driven exactly as it was made.
        if all(v == values[0] for v in values):
            return values[0]
        return n / sum(1.0 / v for v in values) if harmonic else sum(values) / n

    return replace(
        hourly[0],
        speed_factor=mean([e.speed_factor for e in hourly], harmonic=True),
        speed_cap_kmh=min(e.speed_cap_kmh for e in hourly),
        passable=all(e.passable for e in hourly),
        service_factor=mean([e.service_factor for e in hourly]),
        work_capacity=mean([e.work_capacity for e in hourly]),
        wbgt_c=mean([e.wbgt_c for e in hourly]),
        sun_exposed=any(e.sun_exposed for e in hourly),
    )


def mean_conditions(hours: Sequence[Conditions]) -> Conditions:
    """One set of conditions standing for several hours: means, and the worst water."""
    if not hours:
        return NOMINAL
    n = float(len(hours))
    tmax = [h.daily_max_c for h in hours if h.daily_max_c is not None]
    wet = [h.wet_day for h in hours if h.wet_day is not None]
    return Conditions(
        temperature_c=sum(h.temperature_c for h in hours) / n,
        relative_humidity=sum(h.relative_humidity for h in hours) / n,
        rain_mm_h=sum(h.rain_mm_h for h in hours) / n,
        uv_index=sum(h.uv_index for h in hours) / n,
        is_daytime=sum(1 for h in hours if h.is_daytime) * 2 >= len(hours),
        standing_water_mm=max(h.standing_water_mm for h in hours),
        daily_max_c=max(tmax) if tmax else None,
        wet_day=any(wet) if wet else None,
        observed_at=hours[0].observed_at,
        source=hours[0].source,
    )


# ---------------------------------------------------------------------------
# The planner's view
# ---------------------------------------------------------------------------
def leg_speed_kmh(base_speed_kmh: float, effect: Effects) -> float:
    """Speed of one leg under the given effects."""
    if not effect.passable:
        raise ValueError("standing water of 300 mm or more: the streets are impassable")
    speed = min(float(base_speed_kmh) * effect.speed_factor, effect.speed_cap_kmh)
    return max(MIN_SPEED_KMH, speed)


class WeatherTravelModel(VRP.TravelModel):
    """
    A travel model seen through one set of weather effects.

    Distances are unchanged.  Speed on every leg is scaled by the rain factor
    and limited by the standing-water cap, and travel time and emissions follow
    from the speed, as they do in the model it wraps.  The congestion level of a
    leg is left as the base model has it: the sources give the fall in speed and
    not a change in stop-and-go driving.
    """

    def __init__(self, base: VRP.TravelModel, effect: Effects):
        super().__init__(base.distance_m, travel_context=base.ctx,
                         default_speed_kmh=base.default_speed_kmh,
                         freeflow_kmh=base.freeflow_kmh)
        self.base = base
        self.effect = effect
        self._speed: Dict[Tuple[int, int], float] = {}

    def speed_kmh(self, i: int, j: int) -> float:
        key = (i, j)
        speed = self._speed.get(key)
        if speed is None:
            speed = leg_speed_kmh(self.base.speed_kmh(i, j), self.effect)
            self._speed[key] = speed
        return speed

    def friction(self, i: int, j: int) -> float:
        return self.base.friction(i, j)


def travel_under(base: VRP.TravelModel, effect: Effects) -> VRP.TravelModel:
    """The base model itself when the weather changes nothing about travel."""
    if effect.speed_factor == 1.0 and math.isinf(effect.speed_cap_kmh):
        return base
    return WeatherTravelModel(base, effect)


def tasks_under(tasks: Sequence[VRP.BinTask], effect: Effects) -> List[VRP.BinTask]:
    """Copies of the tasks with the service time the weather implies."""
    if effect.service_factor == 1.0:
        return list(tasks)
    return [replace(t, service_minutes=t.service_minutes * effect.service_factor)
            for t in tasks]


def drive_route(order: Sequence[VRP.BinTask], vehicle: VRP.VehicleSpec,
                travel: VRP.TravelModel, hourly: Sequence[Effects],
                protect: Sequence[int] = ()
                ) -> Tuple[VRP.VehicleRoute, List[VRP.BinTask], float]:
    """
    Follow a planned sequence in the weather that actually occurs.

    ``travel`` is the nominal model and ``hourly`` the effects of each hour of
    the shift.  A leg takes the effects of the hour it starts in, and a stop the
    effects of the hour its service starts in.

    The vehicle takes the stops in the planned order.  It serves a stop when it
    can afterwards still return to the depot and tip inside the shift, and
    skips it otherwise and goes on to the next.  A stop reached after its window
    has closed is skipped.  The crew has no forecast.  It judges what still fits
    from the conditions of the hour in which the stop would end.

    ``protect`` names stops that must not be lost, by container id: the reserved
    heads of the overdue queue.  A stop that is not protected is then served
    only if the protected stops still ahead of it in the sequence can be served
    afterwards and the vehicle can return.  A protected stop that could not be
    reached even if the vehicle went straight to it holds nothing back.  A
    protected stop is skipped only when it cannot itself be served with a
    return inside the shift.

    Returns the route as driven, the stops that were skipped, and the minutes by
    which the whole sequence would have overrun the shift had none been skipped.
    """
    depot = vehicle.depot_index
    shift = float(vehicle.shift_minutes)
    protected = set(protect)

    def effect_at(clock: float) -> Effects:
        return hourly[min(len(hourly) - 1, max(0, int(clock // 60.0)))]

    def minutes(i: int, j: int, effect: Effects) -> float:
        if i == j:
            return 0.0
        return 60.0 * (travel.distance(i, j) / 1000.0) / leg_speed_kmh(
            travel.speed_kmh(i, j), effect)

    def co2(i: int, j: int, effect: Effects, payload: float, idle: float = 0.0,
            lifts: int = 0, lifted: float = 0.0) -> float:
        return EM.leg_emissions(
            distance_m=travel.distance(i, j),
            speed_kmh=leg_speed_kmh(travel.speed_kmh(i, j), effect),
            payload_kg=payload, friction=travel.friction(i, j), idle_minutes=idle,
            lifts=lifts, lifted_kg=lifted, profile=vehicle.profile,
            freeflow_kmh=travel.freeflow_kmh).co2_kg(vehicle.profile)

    def serve(state: Dict, task: VRP.BinTask, frozen: Optional[Effects] = None):
        """
        The state after serving ``task`` from ``state``, and the stop record.

        With ``frozen`` set, every leg and the service take those effects: this
        is the crew looking ahead from where it stands.  Without it, each takes
        the effects of the hour it starts in: this is what happens.
        """
        def now(clock: float) -> Effects:
            return frozen if frozen is not None else effect_at(clock)

        clock, load, volume = state["clock"], state["load"], state["volume"]
        position, trip = state["position"], state["trip"]
        distance, emitted = state["distance"], state["co2"]
        added_mass = VRP._finite(task.load_kg)
        added_volume = task.loose_volume_m3 / max(vehicle.compaction_ratio, 1e-9)
        if VRP._exceeds(load + added_mass, volume + added_volume, vehicle):
            effect = now(clock)
            emitted += co2(position, depot, effect, load)
            distance += travel.distance(position, depot)
            clock += minutes(position, depot, effect) + vehicle.tipping_minutes
            load = volume = 0.0
            position = depot
            trip += 1
        effect = now(clock)
        leg_dist = travel.distance(position, task.index)
        arrival = clock + minutes(position, task.index, effect)
        if arrival > task.window_end_min + 1e-9:
            return None, None
        wait = max(0.0, task.window_start_min - arrival)
        begin = arrival + wait
        service = task.service_minutes * now(begin).service_factor
        leg_co2 = co2(position, task.index, effect, load, idle=service,
                      lifts=1, lifted=added_mass)
        after = {"clock": begin + service, "load": load + added_mass,
                 "volume": volume + added_volume, "position": task.index,
                 "trip": trip, "distance": distance + leg_dist,
                 "co2": emitted + leg_co2}
        stop = VRP.Stop(task=task, arrival_min=arrival, start_service_min=begin,
                        departure_min=begin + service, wait_min=wait,
                        leg_distance_m=leg_dist, leg_co2_kg=leg_co2,
                        load_after_kg=after["load"], trip_index=trip)
        return after, stop

    def home(state: Dict, effect: Effects) -> float:
        """Clock at which the vehicle has returned and tipped."""
        return (state["clock"] + minutes(state["position"], depot, effect)
                + vehicle.tipping_minutes)

    def finish(state: Dict, ahead: Sequence[VRP.BinTask], effect: Effects) -> float:
        """Clock after the protected stops still ahead and the return, as judged now."""
        for task in ahead:
            state, _stop = serve(state, task, frozen=effect)
            if state is None:
                return math.inf
        return home(state, effect)

    def attainable(state: Dict, ahead: Sequence[VRP.BinTask],
                   effect: Effects) -> List[VRP.BinTask]:
        """The protected stops ahead that can still be reached from here."""
        kept: List[VRP.BinTask] = []
        for task in ahead:
            if finish(state, kept + [task], effect) <= shift + 1e-9:
                kept.append(task)
        return kept

    empty = {"clock": 0.0, "load": 0.0, "volume": 0.0, "position": depot,
             "trip": 0, "distance": 0.0, "co2": 0.0}

    # What finishing the whole plan would have taken, for the record.
    whole = dict(empty)
    for task in order:
        after, _stop = serve(whole, task)
        if after is not None:
            whole = after
    overrun = (max(0.0, home(whole, effect_at(whole["clock"])) - shift)
               if order else 0.0)

    state = dict(empty)
    stops: List[VRP.Stop] = []
    skipped: List[VRP.BinTask] = []
    for k, task in enumerate(order):
        after, stop = serve(state, task)
        if after is None:
            skipped.append(task)
            continue
        judged = effect_at(after["clock"])
        fits = home(after, judged) <= shift + 1e-9
        if fits and protected and task.node_id not in protected:
            ahead = attainable(state, [t for t in order[k + 1:]
                                       if t.node_id in protected],
                               effect_at(state["clock"]))
            fits = finish(after, ahead, judged) <= shift + 1e-9
        if fits:
            state = after
            stops.append(stop)
        else:
            skipped.append(task)

    route = VRP.VehicleRoute(vehicle=vehicle, trips=1)
    if stops:
        effect = effect_at(state["clock"])
        route.stops = stops
        route.distance_m = state["distance"] + travel.distance(state["position"], depot)
        route.co2_kg = state["co2"] + co2(state["position"], depot, effect, state["load"])
        route.duration_min = home(state, effect)
        route.trips = state["trip"] + 1
        route.load_kg = sum(VRP._finite(s.task.load_kg) for s in stops)
    return route, skipped, overrun


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class WeatherProvider:
    """Interface every provider implements."""

    name = "base"
    attribution = ""

    def current(self, lat: float, lng: float) -> Conditions:
        raise NotImplementedError

    def next_hours(self, lat: float, lng: float, hours: int) -> List[Conditions]:
        """Conditions for each of the coming hours; the present, repeated, by default."""
        return [self.current(lat, lng)] * int(hours)

    def describe(self) -> Dict:
        return {"provider": self.name, "attribution": self.attribution}


class StaticWeatherProvider(WeatherProvider):
    """Fixed conditions: the scenarios, the tests, and the fallback of a live feed."""

    name = "static"

    def __init__(self, conditions: Conditions = NOMINAL):
        self.conditions = conditions

    def current(self, lat: float, lng: float) -> Conditions:
        return self.conditions

    def describe(self) -> Dict:
        return {"provider": self.name, "conditions": self.conditions.as_dict()}


def _http_json(url: str, timeout: float, secret: str = "") -> Dict:
    """GET a JSON document.  An error message never carries the key."""
    import urllib.error
    import urllib.request

    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            return json.loads(handle.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        message = f"HTTP {exc.code}: {detail}"
    except Exception as exc:                        # network, timeout, parse
        message = f"{type(exc).__name__}: {exc}"
    if secret:
        message = message.replace(secret, "***")
    raise RuntimeError(message)


def _celsius(block: Optional[Dict]) -> Optional[float]:
    if not isinstance(block, dict) or block.get("degrees") is None:
        return None
    value = float(block["degrees"])
    if str(block.get("unit", "CELSIUS")).upper() == "FAHRENHEIT":
        value = (value - 32.0) * 5.0 / 9.0
    return value


def _millimetres(block: Optional[Dict]) -> float:
    if not isinstance(block, dict) or block.get("quantity") is None:
        return 0.0
    value = float(block["quantity"])
    if str(block.get("unit", "MILLIMETERS")).upper() == "INCHES":
        value *= 25.4
    return value


def parse_google(document: Dict) -> Conditions:
    """
    Read one current-conditions or hourly document of the Google Weather API.

    The precipitation amount of a current-conditions document is the liquid
    water that accumulated over the last hour, which is a rate in millimetres
    per hour.  An hourly document carries the amount for its own hour.
    """
    temperature = _celsius(document.get("temperature"))
    if temperature is None or document.get("relativeHumidity") is None:
        raise ValueError("document carries no temperature or humidity")
    precipitation = document.get("precipitation") or {}
    stamp = document.get("currentTime") or (document.get("interval") or {}).get("startTime")
    observed = None
    if stamp:
        try:
            # The feed prints nanoseconds, which the standard parser rejects.
            head, _, tail = str(stamp).rstrip("Z").partition(".")
            observed = datetime.fromisoformat(
                head + ("." + tail[:6] if tail else "")).replace(tzinfo=dt_timezone.utc)
        except ValueError:
            observed = None
    return Conditions(
        temperature_c=temperature,
        relative_humidity=float(document["relativeHumidity"]),
        rain_mm_h=_millimetres(precipitation.get("qpf")),
        uv_index=float(document.get("uvIndex") or 0.0),
        is_daytime=bool(document.get("isDaytime", True)),
        observed_at=observed,
        source="google",
    )


class GoogleWeatherProvider(WeatherProvider):
    """
    Live conditions from the Google Maps Platform Weather API.

    Three properties of the service are built in, and none of them is a setting
    a caller may relax.

    The key is a secret.  It is read from the environment, sent only to the
    service, and removed from any error message before the message is kept.

    A response is not kept.  The service terms allow current conditions to be
    cached for one hour and require deletion after that, so a reading lives in
    memory for at most ``MAX_CACHE_SECONDS`` and is never written to disk.  An
    experiment that must be reproducible uses archived weather instead.

    The data carries an attribution, ``attribution``, which any display of it
    must show.

    When the service cannot be reached the provider answers from ``fallback``,
    nominal conditions unless told otherwise, and counts the event, so a
    dispatcher always gets a plan and the plan can be audited for which
    conditions it was built on.
    """

    name = "google"
    attribution = "Source: Includes weather data from Google"
    BASE_URL = "https://weather.googleapis.com/v1"
    MAX_CACHE_SECONDS = 3600.0

    def __init__(self, api_key: Optional[str] = None, cache_seconds: float = 900.0,
                 timeout_seconds: float = 5.0,
                 fallback: Optional[WeatherProvider] = None,
                 grid_deg: float = 0.05, fetch=None):
        self.api_key = api_key if api_key is not None else api_key_from_environment()
        self.cache_seconds = min(self.MAX_CACHE_SECONDS, max(0.0, float(cache_seconds)))
        self.timeout_seconds = float(timeout_seconds)
        self.fallback = fallback or StaticWeatherProvider(NOMINAL)
        self.grid_deg = float(grid_deg)
        # Injectable so a test never touches the network.
        self._fetch = fetch or (lambda url: _http_json(url, self.timeout_seconds,
                                                       self.api_key))
        self._cache: Dict[Tuple[str, int, int, int], Tuple[float, object]] = {}
        self._lock = threading.Lock()
        self.requests = 0
        self.cache_hits = 0
        self.fallbacks = 0
        self.last_error: Optional[str] = None

    def _url(self, path: str, lat: float, lng: float, **extra) -> str:
        import urllib.parse

        query = {"key": self.api_key, "location.latitude": f"{lat:.5f}",
                 "location.longitude": f"{lng:.5f}", **extra}
        return f"{self.BASE_URL}/{path}?" + urllib.parse.urlencode(query)

    def _cached(self, kind: str, lat: float, lng: float, hours: int, build):
        key = (kind, int(round(lat / self.grid_deg)), int(round(lng / self.grid_deg)),
               int(hours))
        now = time.monotonic()
        with self._lock:
            # Expired readings are deleted, not merely ignored.
            for stale in [k for k, (at, _v) in self._cache.items()
                          if now - at >= self.cache_seconds]:
                del self._cache[stale]
            hit = self._cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit[1]
        if not self.api_key:
            self.last_error = "no API key in the environment"
            self.fallbacks += 1
            return None
        try:
            self.requests += 1
            value = build()
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc).replace(self.api_key, "***")
            self.fallbacks += 1
            return None
        if self.cache_seconds > 0.0:
            with self._lock:
                self._cache[key] = (now, value)
        return value

    def current(self, lat: float, lng: float) -> Conditions:
        value = self._cached(
            "current", lat, lng, 0,
            lambda: parse_google(self._fetch(
                self._url("currentConditions:lookup", lat, lng))))
        return value if value is not None else self.fallback.current(lat, lng)

    def next_hours(self, lat: float, lng: float, hours: int) -> List[Conditions]:
        hours = max(1, int(hours))

        def build() -> List[Conditions]:
            document = self._fetch(self._url("forecast/hours:lookup", lat, lng,
                                             hours=hours, pageSize=hours))
            return [parse_google(h) for h in document.get("forecastHours", [])[:hours]]

        value = self._cached("hours", lat, lng, hours, build)
        return value if value else self.fallback.next_hours(lat, lng, hours)

    def describe(self) -> Dict:
        return {"provider": self.name, "attribution": self.attribution,
                "cache_seconds": self.cache_seconds, "requests": self.requests,
                "cache_hits": self.cache_hits, "fallbacks": self.fallbacks,
                "last_error": self.last_error,
                "fallback_provider": self.fallback.name,
                "key_configured": bool(self.api_key)}


class OpenMeteoProvider(WeatherProvider):
    """
    Current conditions from Open-Meteo, an open service that needs no key.

    Its data is published under CC BY 4.0 and may be stored, so it is the
    provider to use when a reading has to go into a result file.
    """

    name = "open-meteo"
    attribution = "Weather data by Open-Meteo.com (CC BY 4.0)"
    URL = "https://api.open-meteo.com/v1/forecast"
    FIELDS = "temperature_2m,relative_humidity_2m,precipitation,uv_index,is_day"

    def __init__(self, timeout_seconds: float = 8.0,
                 fallback: Optional[WeatherProvider] = None, fetch=None):
        self.timeout_seconds = float(timeout_seconds)
        self.fallback = fallback or StaticWeatherProvider(NOMINAL)
        self._fetch = fetch or (lambda url: _http_json(url, self.timeout_seconds))
        self.fallbacks = 0
        self.last_error: Optional[str] = None

    def _url(self, lat: float, lng: float, **extra) -> str:
        import urllib.parse

        return self.URL + "?" + urllib.parse.urlencode(
            {"latitude": f"{lat:.5f}", "longitude": f"{lng:.5f}",
             "timezone": "UTC", **extra})

    @staticmethod
    def _row(row: Dict, stamp: str, interval_s: float = 3600.0) -> Conditions:
        return Conditions(
            temperature_c=float(row["temperature_2m"]),
            relative_humidity=float(row["relative_humidity_2m"]),
            # A current reading sums precipitation over its interval.
            rain_mm_h=float(row.get("precipitation") or 0.0) * 3600.0 / interval_s,
            uv_index=float(row.get("uv_index") or 0.0),
            is_daytime=bool(row.get("is_day", 1)),
            observed_at=datetime.fromisoformat(stamp).replace(tzinfo=dt_timezone.utc),
            source="open-meteo",
        )

    def current(self, lat: float, lng: float) -> Conditions:
        try:
            document = self._fetch(self._url(lat, lng, current=self.FIELDS))
            row = document["current"]
            value = self._row(row, row["time"], float(row.get("interval") or 3600.0))
            self.last_error = None
            return value
        except Exception as exc:
            self.last_error = str(exc)
            self.fallbacks += 1
            return self.fallback.current(lat, lng)

    def next_hours(self, lat: float, lng: float, hours: int) -> List[Conditions]:
        hours = max(1, int(hours))
        try:
            document = self._fetch(self._url(lat, lng, hourly=self.FIELDS,
                                             forecast_hours=hours))
            block = document["hourly"]
            value = [self._row({k: v[i] for k, v in block.items() if k != "time"},
                               block["time"][i])
                     for i in range(min(hours, len(block["time"])))]
            self.last_error = None
            return value
        except Exception as exc:
            self.last_error = str(exc)
            self.fallbacks += 1
            return self.fallback.next_hours(lat, lng, hours)

    def describe(self) -> Dict:
        return {"provider": self.name, "attribution": self.attribution,
                "fallbacks": self.fallbacks, "last_error": self.last_error}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KEY_VARIABLE = "GOOGLE_MAPS_API_KEY"


def api_key_from_environment(dotenv: Optional[pathlib.Path] = None) -> str:
    """
    The Google Maps Platform key, from the environment or from ``.env``.

    The file is the git-ignored one in the repository root, the same file the
    web service loads.  The key is returned to the caller and stored nowhere.
    """
    value = os.environ.get(KEY_VARIABLE, "").strip()
    if value:
        return value
    path = dotenv or pathlib.Path(__file__).resolve().parent.parent / ".env"
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith(KEY_VARIABLE + "="):
                return line.partition("=")[2].strip().strip("'").strip('"')
    return ""


def make_provider(config: Optional[Dict] = None) -> WeatherProvider:
    """Build a provider from a plain mapping (Django ``settings.WEATHER``)."""
    config = config or {}
    kind = str(config.get("PROVIDER", "none")).lower()
    if kind == "google":
        return GoogleWeatherProvider(
            api_key=config.get("API_KEY") or None,
            cache_seconds=float(config.get("CACHE_SECONDS", 900.0)),
            timeout_seconds=float(config.get("TIMEOUT_SECONDS", 5.0)))
    if kind in ("open-meteo", "openmeteo"):
        return OpenMeteoProvider(timeout_seconds=float(config.get("TIMEOUT_SECONDS", 8.0)))
    return StaticWeatherProvider(NOMINAL)
