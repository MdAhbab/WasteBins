"""
The open fill-level corpus.
===========================

The paper assembles the public fill-level records it can find into one corpus
with one schema, so that the demand assumptions of the simulation can be set
against observed demand from several places, and so that other studies can use
the same records without repeating the cleaning.

Sources
-------
``wyndham``
    Wyndham City Council, Victoria, Australia: 33 public containers in Werribee
    and Point Cook, daily readings on a six-level scale, June 2018 to May 2021.
    CC BY 2.5 AU.
``mildura``
    Mildura Rural City Council, Victoria, Australia: 7 street bins, ultrasonic
    distance from the sensor to the waste surface, about hourly, December 2021
    to September 2023.  CC BY 3.0 AU.
``boston``
    City of Boston: the calls of 519 Big Belly compacting receptacles in 2014.
    Each call gives one of three LED fill levels (green, yellow, red, about 20,
    60 and 100 percent full) and whether the collection door was opened.  PDDL.
    Read when its files are present in ``data/corpus/raw/boston``.
``ai4waste``
    AI4WASTE annotated collection events (Tarabek et al., 2026): lift times of
    eight collection vehicles in Slovakia.  CC BY 4.0.  It has no fill levels
    and enters only the service-time comparison.

Harmonised tables (``data/corpus/processed``)
---------------------------------------------
``containers.csv``  one row per container: city, id, position, stream, sensor.
``readings.csv``    one row per reading: city, id, local time, fill in [0, 1],
                    the raw value and its unit.
``daily.csv``       one row per container and local day: the last fill of the
                    day, whether a collection was detected since the previous
                    day, and the daily rise when it is defined.
``collections.csv`` one row per detected collection: the drop and its time.
``boston_cycles.csv`` one row per Boston emptying cycle (see below).

Definitions, applied to every city with a fill scale
----------------------------------------------------
A *collection* is a fall in fill of at least ``DROP`` between two consecutive
valid readings.  The *daily rise* of a container on day ``d`` is its last fill
of day ``d`` less its last fill of day ``d - 1``.  It is defined when both
days have readings, no collection was detected between the two readings, and
the earlier fill is at most ``MAX_START_FILL``, so that a container near the
top of its scale cannot hide a rise.  These are the rules of the weather
regression of the paper, now written once for every city.

Boston has three levels and calls only when something changes, so a daily
rise is not defined there.  Its record is read as *emptying cycles* instead.
An emptying is a collection call that reports green; calls of one unit less
than ``BOSTON_MERGE_H`` apart are one visit.  A cycle runs from one visit to
the next.  Within it the record gives the time to the first yellow and to the
first red call, and the *wait at full*, the time from the first red call to
the visit that ends the cycle.  A cycle longer than ``BOSTON_MAX_CYCLE_DAYS``
is taken as a gap in the record and left out.

Run:  python -m experiments.corpus
Out:  data/corpus/processed/*.csv, results/corpus.json
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import exp_weather_demand as WDM   # noqa: E402
from experiments import weather_data as WD          # noqa: E402
from experiments import wyndham as WY               # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "corpus" / "raw"
OUT = ROOT / "data" / "corpus" / "processed"
RESULTS = pathlib.Path(__file__).resolve().parent / "results"

DROP = 0.25                 # fall in fill that marks a collection
MAX_START_FILL = 0.6        # as in the weather regression
MIN_DAYS_FOR_RATE = 60      # daily rises needed to estimate a container's rate
#: Ultrasonic distances at or below this, and the code 9999, are sensor errors.
MILDURA_MIN_MM, MILDURA_ERROR_MM = 100.0, 9999.0
#: Local hours that count as the day half of a 12-hour cycle pair.
DAY_HOURS = range(6, 18)
#: Fill of the three LED levels of a Big Belly, as the city's notes define them.
BOSTON_LEVEL = {"GREEN": 0.2, "YELLOW": 0.6, "RED": 1.0}
BOSTON_MERGE_H = 1.0
BOSTON_MAX_CYCLE_DAYS = 28
#: Emptying cycles a unit needs before its fill speed is estimated.
BOSTON_MIN_CYCLES = 10


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def wyndham() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    containers, fill = WY.load()
    c = pd.DataFrame({
        "city": "wyndham", "container_id": containers["serial"].astype(str),
        "latitude": containers["latitude"], "longitude": containers["longitude"],
        "stream": containers["stream"], "area": containers["suburb"],
        "sensor": "fill level, six classes", "timezone": "Australia/Melbourne"})
    r = pd.DataFrame({
        "city": "wyndham", "container_id": fill["serial"].astype(str),
        "time_local": pd.to_datetime(fill["date"]),
        "fill": (fill["fullness"] / WY.FULLNESS_MAX).clip(0.0, 1.0),
        "raw_value": fill["fullness"], "raw_unit": "level of 10"})
    quality = {"rows_raw": int(len(fill)), "rows_invalid": 0,
               "duplicates": int(fill.duplicated(["serial", "date"]).sum())}
    return c, r, quality


def mildura() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    path = RAW / "mildura" / "mildura-bin-sensor-data.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["t"] = (pd.to_datetime(frame["Timestamp (UTC)"], utc=True)
                  .dt.tz_convert("Australia/Melbourne").dt.tz_localize(None))
    value = frame["Calibrated Value"].astype(float)
    valid = (value > MILDURA_MIN_MM) & (value < MILDURA_ERROR_MM)
    quality = {"rows_raw": int(len(frame)), "rows_invalid": int((~valid).sum()),
               "duplicates": int(frame.duplicated(["Device Tag", "t"]).sum())}
    frame = frame[valid].drop_duplicates(["Device Tag", "t"]).sort_values(["Device Tag", "t"])
    rows, meta = [], []
    for tag, g in frame.groupby("Device Tag"):
        d = g["Calibrated Value"].astype(float)
        # The sensor reads the distance down to the waste.  The empty distance
        # is taken as the 99th percentile of a bin's readings; full is a
        # distance of zero.  A rolling median of three readings removes single
        # echoes off the bin wall before the fill is formed.
        empty = float(np.percentile(d, 99))
        smooth = d.rolling(3, center=True, min_periods=1).median()
        fill = (1.0 - smooth / empty).clip(0.0, 1.0)
        rows.append(pd.DataFrame({"city": "mildura", "container_id": tag,
                                  "time_local": g["t"].values, "fill": fill.values,
                                  "raw_value": d.values, "raw_unit": "mm to surface"}))
        meta.append({"city": "mildura", "container_id": tag,
                     "latitude": float(g["Location Lat"].median()),
                     "longitude": float(g["Location Long"].median()),
                     "stream": "general", "area": "Mildura district",
                     "sensor": "ultrasonic distance", "timezone": "Australia/Melbourne",
                     "empty_distance_mm": round(empty, 1)})
    return pd.DataFrame(meta), pd.concat(rows, ignore_index=True), quality


def boston() -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
    """Read the Boston files when they are present.  See `_boston_parse`."""
    folder = RAW / "boston"
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    if not files:
        return None
    return _boston_parse(files)


def _boston_parse(files: List[pathlib.Path]):
    alerts = next(p for p in files if "alert" in p.name)
    frame = pd.read_csv(alerts)
    frame["t"] = pd.to_datetime(frame["timestamp"], format="%m/%d/%Y %I:%M:%S %p")
    frame["container_id"] = frame["description"].str.strip()
    position = frame["Location"].str.strip("()").str.split(",", expand=True).astype(float)
    frame["latitude"], frame["longitude"] = position[0], position[1]
    valid = frame["fullness"].isin(list(BOSTON_LEVEL)) & frame["t"].notna()
    exact = frame.duplicated(["container_id", "t", "fullness", "collection"])
    quality = {"rows_raw": int(len(frame)), "rows_invalid": int((~valid).sum()),
               "duplicates": int(exact.sum())}
    frame = frame[valid & ~exact].sort_values(["container_id", "t"], kind="stable")
    listed = next((p for p in files if "location" in p.name), None)
    if listed is not None:
        names = set(pd.read_csv(listed)["description"].str.strip())
        quality["units_in_locations_file"] = int(frame["container_id"].drop_duplicates()
                                                 .isin(names).sum())
    quality["collection_calls_not_green"] = int((frame["collection"]
                                                 & (frame["fullness"] != "GREEN")).sum())
    r = pd.DataFrame({"city": "boston", "container_id": frame["container_id"],
                      "time_local": frame["t"], "fill": frame["fullness"].map(BOSTON_LEVEL),
                      "raw_value": frame["fullness"], "raw_unit": "LED level",
                      "collection_flag": frame["collection"].astype(bool)})
    pos = frame.groupby("container_id")[["latitude", "longitude"]].median()
    c = pd.DataFrame({"city": "boston", "container_id": pos.index,
                      "latitude": pos["latitude"].values, "longitude": pos["longitude"].values,
                      "stream": "not stated", "area": "Boston",
                      "sensor": "compactor fill, three levels",
                      "timezone": "America/New_York"})
    return c, r.reset_index(drop=True), quality


def boston_cycles(readings: pd.DataFrame,
                  any_call: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Emptying cycles and visits of the Boston units, by the module notes.

    With ``any_call`` every collection call empties the unit, including the
    few that still report yellow or red; this is the sensitivity check.
    """
    cycles, visits = [], []
    hour = np.timedelta64(1, "h")
    for cid, g in readings.groupby("container_id", sort=True):
        t = pd.to_datetime(g["time_local"]).to_numpy()
        level = g["raw_value"].to_numpy()
        flag = g["collection_flag"].to_numpy(bool)
        emptied = np.flatnonzero(flag if any_call else flag & (level == "GREEN"))
        starts: List[int] = []
        for k in emptied:
            if not starts or (t[k] - t[last]) / hour > BOSTON_MERGE_H:
                starts.append(k)
            last = k
        for k in starts:
            visits.append({"city": "boston", "container_id": cid,
                           "time_local": pd.Timestamp(t[k]), "fill_before": np.nan,
                           "fill_after": BOSTON_LEVEL["GREEN"], "source": "collection call"})
        for a, b in zip(starts[:-1], starts[1:]):
            length = (t[b] - t[a]) / hour
            inside = np.arange(a + 1, b)
            inside = inside[~flag[inside]] if any_call else inside
            yellow = inside[np.isin(level[inside], ("YELLOW", "RED"))]
            red = inside[level[inside] == "RED"]
            to_yellow = (t[yellow[0]] - t[a]) / hour if len(yellow) else np.nan
            to_red = (t[red[0]] - t[a]) / hour if len(red) else np.nan
            cycles.append({"container_id": cid, "start": pd.Timestamp(t[a]),
                           "end": pd.Timestamp(t[b]), "length_h": length,
                           "to_yellow_h": to_yellow, "to_red_h": to_red,
                           "yellow_wait_h": length - to_yellow, "red_wait_h": length - to_red,
                           "first_yellow_hour": (pd.Timestamp(t[yellow[0]]).hour
                                                 if len(yellow) else np.nan),
                           "kept": length <= 24.0 * BOSTON_MAX_CYCLE_DAYS})
    return pd.DataFrame(cycles), pd.DataFrame(visits)


def boston_units(kept: pd.DataFrame) -> pd.DataFrame:
    """Cycles, yellow calls, hours below yellow and longest wait at full, by unit."""
    # Fill speed of a unit: the yellow calls per hour spent below yellow, so
    # that a cycle emptied before it reached yellow still counts its time.
    below = kept["to_yellow_h"].fillna(kept["length_h"])
    return (kept.assign(below=below, reached=kept["to_yellow_h"].notna())
            .groupby("container_id").agg(cycles=("length_h", "size"),
                                         reached=("reached", "sum"), below=("below", "sum"),
                                         longest_red_wait=("red_wait_h", "max")))


def boston_relative_rates(kept: pd.DataFrame) -> pd.Series:
    """Fill speed of each unit with enough cycles, divided by the median unit."""
    per_unit = boston_units(kept)
    rated = per_unit[(per_unit["cycles"] >= BOSTON_MIN_CYCLES) & (per_unit["reached"] > 0)]
    rate = rated["reached"] / rated["below"]
    return rate / rate.median()


def boston_summary(cycles: pd.DataFrame, visits: pd.DataFrame) -> Dict:
    """Waits at full, fill speed and its spread between units, from the cycles."""
    kept = cycles[cycles["kept"]]

    def spread(x: pd.Series) -> Dict:
        x = x.dropna()
        return {"n": int(len(x)), "median": round(float(x.median()), 2),
                "p75": round(float(x.quantile(0.75)), 2),
                "p90": round(float(x.quantile(0.90)), 2),
                "p99": round(float(x.quantile(0.99)), 2), "max": round(float(x.max()), 2),
                **{f"share_over_{h}h": round(float((x > h).mean()), 4)
                   for h in (12, 24, 48, 72)}}

    per_unit = boston_units(kept)
    rel = boston_relative_rates(kept).to_numpy()
    hours = kept["first_yellow_hour"].dropna()
    longest = per_unit["longest_red_wait"].dropna()
    return {
        "units": int(cycles["container_id"].nunique()), "visits": int(len(visits)),
        "cycles": int(len(cycles)), "cycles_kept": int(len(kept)),
        "cycle_length_h": spread(kept["length_h"]),
        "share_reaching_yellow": round(float(kept["to_yellow_h"].notna().mean()), 4),
        "share_reaching_red": round(float(kept["to_red_h"].notna().mean()), 4),
        "time_to_yellow_h": spread(kept["to_yellow_h"]),
        "time_to_red_h": spread(kept["to_red_h"]),
        "wait_from_yellow_h": spread(kept["yellow_wait_h"]),
        "wait_at_full_h": spread(kept["red_wait_h"]),
        "units_with_red": int(len(longest)),
        "unit_longest_wait_at_full_h": spread(longest),
        "heterogeneity": {"containers": int(len(rel)), "method": "yellow calls per hour below yellow",
                          "relative_min": round(float(rel.min()), 3),
                          "relative_p10": round(float(np.percentile(rel, 10)), 3),
                          "relative_p90": round(float(np.percentile(rel, 90)), 3),
                          "relative_max": round(float(rel.max()), 3),
                          "relative_cv": round(float(rel.std(ddof=1) / rel.mean()), 3)},
        "yellow_day_share": round(float(hours.isin(list(DAY_HOURS)).mean()), 4),
    }


# ---------------------------------------------------------------------------
# Derived tables
# ---------------------------------------------------------------------------
def daily_table(readings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Daily rises and detected collections, by the rules in the module notes."""
    days, events = [], []
    for (city, cid), g in readings.sort_values("time_local").groupby(["city", "container_id"]):
        t = pd.to_datetime(g["time_local"]).values
        f = g["fill"].values.astype(float)
        fall = np.r_[False, (f[:-1] - f[1:]) >= DROP]
        for k in np.flatnonzero(fall):
            events.append({"city": city, "container_id": cid,
                           "time_local": pd.Timestamp(t[k]),
                           "fill_before": float(f[k - 1]), "fill_after": float(f[k])})
        frame = pd.DataFrame({"t": t, "fill": f, "fall": fall})
        frame["date"] = frame["t"].dt.normalize()
        per_day = frame.groupby("date").agg(last=("fill", "last"), n=("fill", "size"),
                                            fell=("fall", "any"))
        per_day = per_day.reindex(pd.date_range(per_day.index.min(), per_day.index.max()))
        prev_last = per_day["last"].shift(1)
        rise = per_day["last"] - prev_last
        ok = (per_day["n"].notna() & prev_last.notna() & ~per_day["fell"].fillna(True)
              .astype(bool) & (prev_last <= MAX_START_FILL + 1e-9) & (rise >= -1e-9))
        out = pd.DataFrame({"city": city, "container_id": cid, "date": per_day.index,
                            "readings": per_day["n"].fillna(0).astype(int),
                            "fill_last": per_day["last"],
                            "collected": per_day["fell"].fillna(False).astype(bool),
                            "rise": np.where(ok, rise.clip(lower=0.0), np.nan)})
        days.append(out[out["readings"] > 0])
    return pd.concat(days, ignore_index=True), pd.DataFrame(events)


def container_rates(daily: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Mean daily rise of each container with enough rises, by city."""
    out = {}
    for city, g in daily.groupby("city"):
        rates = g.dropna(subset=["rise"]).groupby("container_id")["rise"].agg(["mean", "size"])
        rates = rates[(rates["size"] >= MIN_DAYS_FOR_RATE) & (rates["mean"] > 0)]
        if len(rates) >= 3:
            out[city] = rates
    return out


def heterogeneity(daily: pd.DataFrame) -> Dict:
    """Mean daily rise of each container, divided by the median of its city."""
    out = {}
    for city, rates in container_rates(daily).items():
        rel = (rates["mean"] / rates["mean"].median()).values
        out[city] = {"containers": int(len(rel)),
                     "mean_daily_rise": round(float(rates["mean"].mean()), 4),
                     "relative_min": round(float(rel.min()), 3),
                     "relative_p10": round(float(np.percentile(rel, 10)), 3),
                     "relative_p90": round(float(np.percentile(rel, 90)), 3),
                     "relative_max": round(float(rel.max()), 3),
                     "relative_cv": round(float(rel.std(ddof=1) / rel.mean()), 3)}
    return out


def diurnal(readings: pd.DataFrame, max_gap_h: float = 1.5) -> Optional[Dict]:
    """
    Share of the daily rise that falls in each local hour, from hourly records.

    Only consecutive readings at most ``max_gap_h`` apart, on the same local
    day, with no collection between them, are used.  The increment is assigned
    to the hour of the later reading.  Negative increments are kept, so that
    sensor noise averages out rather than being folded into the rise.
    """
    parts = []
    for (_city, _cid), g in readings.sort_values("time_local").groupby(["city", "container_id"]):
        t = pd.to_datetime(g["time_local"])
        gap = t.diff().dt.total_seconds() / 3600.0
        inc = g["fill"].diff()
        same_day = t.dt.normalize() == t.shift(1).dt.normalize()
        keep = (gap <= max_gap_h) & same_day & (inc > -DROP)
        parts.append(pd.DataFrame({"hour": t[keep].dt.hour, "inc": inc[keep]}))
    if not parts:
        return None
    frame = pd.concat(parts)
    if len(frame) < 1000:
        return None
    by_hour = frame.groupby("hour")["inc"].mean().reindex(range(24), fill_value=0.0)
    total = float(by_hour.clip(lower=0.0).sum())
    share = (by_hour.clip(lower=0.0) / total) if total > 0 else by_hour * 0.0
    day = float(share[list(DAY_HOURS)].sum())
    return {"increments": int(len(frame)),
            "share_by_hour": [round(float(x), 4) for x in share.values],
            "day_share": round(day, 4), "night_share": round(1.0 - day, 4),
            # The simulation scales the rate by 1.3 by day and 0.7 by night,
            # which puts 65 percent of a day's rise in the day half.
            "model_day_share": 0.65,
            "implied_day_factor": round(2.0 * day, 3),
            "implied_night_factor": round(2.0 * (1.0 - day), 3)}


def weather_response(daily: pd.DataFrame, containers: pd.DataFrame, city: str) -> Optional[Dict]:
    """The weather regression of the paper, refitted on one city of the corpus."""
    g = daily[(daily["city"] == city)].dropna(subset=["rise"]).copy()
    if len(g) < 200:
        return None
    pos = containers[containers["city"] == city].set_index("container_id")
    start = (g["date"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = g["date"].max().strftime("%Y-%m-%d")
    frames = []
    # One weather series per grid cell of the archive, about 0.25 degrees.
    pos = pos.assign(cell=[f"{round(a * 4) / 4:.2f}_{round(b * 4) / 4:.2f}"
                           for a, b in zip(pos["latitude"], pos["longitude"])])
    for cell, members in pos.groupby("cell"):
        site = f"{city}_{cell}".replace("-", "m").replace(".", "p")
        WD.SITES.setdefault(site, {"latitude": float(members["latitude"].mean()),
                                   "longitude": float(members["longitude"].mean()),
                                   "timezone": members["timezone"].iloc[0]})
        weather = WD.daily(site, start, end).set_index("date")
        sub = g[g["container_id"].isin(members.index)].copy()
        prev = sub["date"] - pd.Timedelta(days=1)
        ok = sub["date"].isin(weather.index) & prev.isin(weather.index)
        sub, prev = sub[ok], prev[ok]
        a, b = weather.loc[prev.values], weather.loc[sub["date"].values]
        sub["tmax_both"] = 0.5 * (a["temperature_2m_max"].values + b["temperature_2m_max"].values)
        sub["rain_both"] = a["precipitation_sum"].values + b["precipitation_sum"].values
        frames.append(sub)
    frame = pd.concat(frames, ignore_index=True)
    frame["serial"] = frame["container_id"]
    frame["dow"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    # The Wyndham record spans the pandemic restrictions from 16 March 2020,
    # which the paper models with their own effect; the other records do not.
    frame["restrictions"] = ((frame["date"] >= WDM.RESTRICTIONS_FROM).astype(float)
                             if city == "wyndham" else 0.0)
    months = frame["month"].nunique() == 12
    result = WDM.fit(frame, month_effects=months)
    if result is not None:
        result["month_effects"] = bool(months)
    return result


def service_times() -> Optional[Dict]:
    """Lift and between-lift times of the AI4WASTE collection vehicles."""
    files = sorted((RAW / "ai4waste_events").glob("*_collection_events.csv"))
    if not files:
        return None
    parts = []
    for path in files:
        e = pd.read_csv(path)
        for col in ("timestamp", "timestamp_start", "timestamp_end"):
            e[col] = pd.to_datetime(e[col], format="mixed")
        e = e.sort_values("timestamp")
        e["lift_s"] = (e["timestamp_end"] - e["timestamp_start"]).dt.total_seconds()
        e["gap_s"] = e["timestamp"].diff().dt.total_seconds()
        e.loc[e["timestamp"].dt.date != e["timestamp"].shift(1).dt.date, "gap_s"] = np.nan
        parts.append(e)
    e = pd.concat(parts)
    gap = e["gap_s"].dropna()
    gap = gap[gap > 0]
    return {"events": int(len(e)), "vehicles": int(e["car_id"].nunique()),
            "days": int(e["timestamp"].dt.date.nunique()),
            "lift_s_median": float(e["lift_s"].median()),
            "lift_s_p95": float(e["lift_s"].quantile(0.95)),
            "gap_s_median": float(gap.median()),
            "gap_s_p90": float(gap.quantile(0.90)),
            "gap_share_below_60s": float((gap < 60).mean())}


def summarise(containers, readings, daily, events, quality) -> Dict:
    out = {}
    for city, r in readings.groupby("city"):
        d = daily[daily["city"] == city]
        t = pd.to_datetime(r["time_local"])
        span_days = (t.max() - t.min()).days + 1
        cover = d.groupby("container_id")["date"].nunique()
        out[city] = {
            "containers": int(r["container_id"].nunique()),
            "readings": int(len(r)),
            "first": str(t.min().date()), "last": str(t.max().date()),
            "span_days": int(span_days),
            "container_days": int(len(d)),
            "median_days_per_container": float(cover.median()) if len(cover) else None,
            "median_reading_gap_h": float(r.sort_values("time_local").groupby("container_id")
                                          ["time_local"].apply(lambda s: pd.to_datetime(s)
                                                               .diff().dt.total_seconds()
                                                               .median() / 3600.0).median()),
            "collections_detected": int((events["city"] == city).sum()) if len(events) else 0,
            "daily_rises": int(d["rise"].notna().sum()),
            "mean_daily_rise": float(d["rise"].mean()) if d["rise"].notna().any() else None,
            "quality": quality.get(city, {}),
        }
    return out


def main() -> int:
    sources = {"wyndham": wyndham(), "mildura": mildura()}
    extra = boston()
    if extra is not None:
        sources["boston"] = extra
    containers = pd.concat([s[0] for s in sources.values()], ignore_index=True)
    readings = pd.concat([s[1] for s in sources.values()], ignore_index=True)
    quality = {city: s[2] for city, s in sources.items()}
    # Boston has no fill scale to take a daily rise from; it enters as cycles.
    scaled = readings[readings["city"] != "boston"]
    daily, events = daily_table(scaled)
    events["source"] = "fill drop"
    cycles = visits = None
    if "boston" in sources:
        cycles, visits = boston_cycles(readings[readings["city"] == "boston"])
        events = pd.concat([events, visits], ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    containers.to_csv(OUT / "containers.csv", index=False)
    readings.to_csv(OUT / "readings.csv", index=False)
    daily.to_csv(OUT / "daily.csv", index=False)
    events.to_csv(OUT / "collections.csv", index=False)
    if cycles is not None:
        cycles.to_csv(OUT / "boston_cycles.csv", index=False)

    level_cities = list(scaled["city"].unique())
    payload = {
        "definitions": {"collection_drop": DROP, "max_start_fill": MAX_START_FILL,
                        "min_days_for_rate": MIN_DAYS_FOR_RATE,
                        "mildura_valid_mm": [MILDURA_MIN_MM, MILDURA_ERROR_MM],
                        "day_hours": [DAY_HOURS.start, DAY_HOURS.stop],
                        "boston_levels": BOSTON_LEVEL, "boston_merge_h": BOSTON_MERGE_H,
                        "boston_max_cycle_days": BOSTON_MAX_CYCLE_DAYS,
                        "boston_min_cycles": BOSTON_MIN_CYCLES},
        "cities": summarise(containers, readings, daily, events, quality),
        "heterogeneity": heterogeneity(daily),
        "diurnal": {city: diurnal(readings[readings["city"] == city])
                    for city in level_cities},
        "weather_response": {city: weather_response(daily, containers, city)
                             for city in level_cities},
        "service_times": service_times(),
    }
    if cycles is not None:
        payload["boston"] = boston_summary(cycles, visits)
        payload["heterogeneity"]["boston"] = payload["boston"]["heterogeneity"]
        alt = boston_summary(*boston_cycles(readings[readings["city"] == "boston"],
                                            any_call=True))
        payload["boston"]["sensitivity_any_collection_call"] = {
            k: alt[k] for k in ("cycles_kept", "wait_at_full_h", "unit_longest_wait_at_full_h",
                                "heterogeneity")}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "corpus.json").write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps({k: payload[k] for k in ("cities", "heterogeneity")}, indent=1,
                     default=float)[:3000])
    for city, d in payload["diurnal"].items():
        if d:
            print(city, "day share", d["day_share"], "implied factors",
                  d["implied_day_factor"], d["implied_night_factor"])
    for city, w in payload["weather_response"].items():
        if w:
            print(city, "temp per 10C", round(w["temperature_per_10C"]["percent_change"], 1),
                  "p", round(w["temperature_per_10C"]["p"], 4), "wet",
                  round(w["wet"]["percent_change"], 1), "p", round(w["wet"]["p"], 4),
                  "n", w["observations"])
    print("service", payload["service_times"])
    if "boston" in payload:
        print("boston", json.dumps({k: v for k, v in payload["boston"].items()
                                    if k != "heterogeneity"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
