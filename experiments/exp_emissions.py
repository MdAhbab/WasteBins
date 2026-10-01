"""
The emission model at stated settings, and against a measured truck.
====================================================================

The paper gives the emission model as an equation.  This script evaluates it at
settings that are written out here, so that a reader can reproduce each row of
the emission table from the equation and the parameter table alone.

Three things are reported.

``duty``
    Emissions per kilometre and the share of fuel in each of the four terms, on
    a transfer leg and on a collection leg, at two congestion levels.

``range``
    The lowest and highest emissions per kilometre over every leg of the
    primary sample, driven empty and driven full, with no stop.  This is the
    spread a single factor per kilometre would have to stand for.

``measured``
    The fuel economy that Sandhu et al. measured on roll-off refuse trucks,
    converted to the same units, beside the emissions per kilometre of the
    plans of the main study.  The plan figures are added by `analyze.py` once
    that study has run.

Run:  python -m experiments.exp_emissions
Out:  results/emissions.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import emissions as EM        # noqa: E402
from wastebins_core import traffic as TR          # noqa: E402
from experiments import exp_fleet as EF           # noqa: E402
from experiments import instances as IN           # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

FREEFLOW_KMH = TR.FREEFLOW_KMH
PROFILE = EM.DEFAULT_PROFILE

#: The rows of the emission table.  Every input of the equation is listed.
DUTY = {
    "transfer_empty": {
        "label": "Transfer leg, empty, no congestion",
        "distance_km": 5.0, "payload_kg": 0.0, "congestion": 0.0,
        "service_min": 0.0, "lifts": 0, "lifted_kg": 0.0},
    "transfer_full": {
        "label": "Transfer leg, full, no congestion",
        "distance_km": 5.0, "payload_kg": 6000.0, "congestion": 0.0,
        "service_min": 0.0, "lifts": 0, "lifted_kg": 0.0},
    "collection_light": {
        "label": "Collection leg, half load, light congestion",
        "distance_km": 0.5, "payload_kg": 3000.0, "congestion": 0.2,
        "service_min": 4.0, "lifts": 1, "lifted_kg": 150.0},
    "collection_heavy": {
        "label": "Collection leg, half load, heavy congestion",
        "distance_km": 0.5, "payload_kg": 3000.0, "congestion": 0.8,
        "service_min": 4.0, "lifts": 1, "lifted_kg": 150.0},
}

#: Sandhu, Frey, Bartelt-Hunt and Jones (2015), roll-off refuse trucks,
#: doi:10.1080/10962247.2014.990587: six trucks, 870 miles, mean speed 16 mph,
#: mean fuel economy 4.4 miles per US gallon, half of the time at idle.
MEASURED = {
    "source": "Sandhu et al. (2015), doi:10.1080/10962247.2014.990587",
    "vehicle": "roll-off refuse truck",
    "fuel_economy_mpg": 4.4,
    "mean_speed_mph": 16.0,
    "idle_time_share": 0.50,
}
LITRES_PER_US_GALLON = 3.785411784
KM_PER_MILE = 1.609344


def duty_row(spec: Dict) -> Dict:
    x = float(spec["congestion"])
    speed = TR.bpr_speed_kmh(x, FREEFLOW_KMH)
    leg = EM.leg_emissions(
        distance_m=1000.0 * spec["distance_km"], speed_kmh=speed,
        payload_kg=spec["payload_kg"], friction=x,
        idle_minutes=spec["service_min"], lifts=spec["lifts"],
        lifted_kg=spec["lifted_kg"], profile=PROFILE, freeflow_kmh=FREEFLOW_KMH)
    total = leg.fuel_total_l
    co2 = leg.co2_kg(PROFILE)
    return {
        **spec,
        "speed_kmh": round(speed, 2),
        "fuel_l": round(total, 5),
        "fuel_l_per_100km": round(100.0 * total / spec["distance_km"], 2),
        "co2_kg": round(co2, 5),
        "co2_kg_per_km": round(co2 / spec["distance_km"], 3),
        "share_cruise": round(leg.fuel_cruise_l / total, 4),
        "share_stop_go": round(leg.fuel_stopgo_l / total, 4),
        "share_idle": round(leg.fuel_idle_l / total, 4),
        "share_compaction": round(leg.fuel_pto_l / total, 4),
    }


def leg_range() -> Dict:
    """Emissions per kilometre over every leg of the primary sample, no stop."""
    snapshot = IN.snapshots("S0", 25, 6)[0]
    travel = EF.travel_for(snapshot, IN.WHEN)
    profile = EF.build_fleet(snapshot)[0].profile
    n = len(snapshot["coords"])
    rates = {"empty": [], "full": []}
    speeds = []
    for i in range(n):
        for j in range(n):
            if i == j or travel.distance(i, j) < 100.0:
                continue
            km = travel.distance(i, j) / 1000.0
            speeds.append(travel.speed_kmh(i, j))
            rates["empty"].append(travel.leg_co2(i, j, 0.0, profile) / km)
            rates["full"].append(travel.leg_co2(i, j, profile.capacity_kg, profile) / km)
    both = np.array(rates["empty"] + rates["full"])
    return {
        "legs": len(speeds),
        "legs_excluded": "pairs less than 100 m apart by road",
        "speed_kmh": {"min": round(float(np.min(speeds)), 2),
                      "median": round(float(np.median(speeds)), 2),
                      "max": round(float(np.max(speeds)), 2)},
        "co2_kg_per_km": {
            "min": round(float(both.min()), 3), "max": round(float(both.max()), 3),
            "empty_median": round(float(np.median(rates["empty"])), 3),
            "full_median": round(float(np.median(rates["full"])), 3),
            "ratio_max_to_min": round(float(both.max() / both.min()), 2),
        },
    }


def measured() -> Dict:
    km_per_l = MEASURED["fuel_economy_mpg"] * KM_PER_MILE / LITRES_PER_US_GALLON
    return {
        **MEASURED,
        "fuel_l_per_100km": round(100.0 / km_per_l, 2),
        "co2_kg_per_km": round(EM.DIESEL_KG_CO2_PER_L / km_per_l, 3),
        "mean_speed_kmh": round(MEASURED["mean_speed_mph"] * KM_PER_MILE, 1),
    }


def main() -> int:
    payload = {
        "co2_kg_per_litre": round(EM.DIESEL_KG_CO2_PER_L, 4),
        "co2_kg_per_mj": EM.DIESEL_KG_CO2_PER_MJ,
        "fuel_mj_per_litre": EM.DIESEL_LHV_MJ_PER_L,
        "freeflow_kmh": FREEFLOW_KMH,
        "profile": {
            "kerb_mass_kg": PROFILE.kerb_mass_kg, "capacity_kg": PROFILE.capacity_kg,
            "frontal_area_m2": PROFILE.frontal_area_m2,
            "drag_coefficient": PROFILE.drag_coefficient,
            "rolling_resistance": PROFILE.rolling_resistance,
            "engine_efficiency": PROFILE.engine_efficiency,
            "driveline_efficiency": PROFILE.driveline_efficiency,
            "idle_fuel_l_per_h": PROFILE.idle_fuel_l_per_h,
            "pto_fuel_l_per_lift": PROFILE.pto_fuel_l_per_lift,
            "low_speed_penalty_ref_kmh": PROFILE.low_speed_penalty_ref_kmh,
            "low_speed_penalty_max": PROFILE.low_speed_penalty_max,
        },
        "duty": {name: duty_row(spec) for name, spec in DUTY.items()},
        "range": leg_range(),
        "measured": measured(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "emissions.json").write_text(json.dumps(payload, indent=2))
    for name, row in payload["duty"].items():
        print(f"{row['label']:<46}{row['speed_kmh']:>6.1f} km/h "
              f"{row['co2_kg_per_km']:>7.3f} kg/km  {row['fuel_l_per_100km']:>6.1f} L/100km  "
              f"cruise {100 * row['share_cruise']:4.0f}%  stop-go {100 * row['share_stop_go']:4.0f}%  "
              f"idle {100 * row['share_idle']:4.0f}%  compaction {100 * row['share_compaction']:4.0f}%")
    print("range over legs:", payload["range"])
    print("measured:", payload["measured"])
    print(f"wrote {RESULTS / 'emissions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
