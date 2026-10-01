"""
Historical weather for the two study areas, from an open archive.
=================================================================

The experiments need weather that can be archived and redistributed with the
results, so that a reader can rebuild every number.  A commercial live feed
cannot supply that: its terms limit how long a response may be kept.  The
historical series therefore come from the Open-Meteo archive, which serves the
ERA5 reanalysis of the European Centre for Medium-Range Weather Forecasts under
CC BY 4.0, and the ultraviolet index from the Copernicus Atmosphere Monitoring
Service through the same provider.

Each series is fetched once and written to ``data/weather/`` as a CSV with a
sidecar file recording the request, the grid cell that answered it and the
retrieval time.  Later runs read the CSV and make no request.

What the archive is good for and what it is not
-----------------------------------------------
Air temperature and humidity are well represented.  Rainfall is not: a
reanalysis reports the mean over a grid cell some 25 km across, which spreads a
tropical downpour thin.  On the evening of 21 September 2023 the press reported
122 mm in Dhaka between 18:00 and midnight, citing the Bangladesh Meteorological
Department, and the archive holds 5.8 mm in its wettest hour that day.  The
series is therefore used for daily rain occurrence and for heat, and never for
hourly rain intensity.  The rain scenarios are defined by intensity class
instead.

Run:  python -m experiments.weather_data
Out:  data/weather/*.csv, data/weather/*.meta.json
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, Sequence

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "weather"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ATTRIBUTION = ("Weather data by Open-Meteo.com (CC BY 4.0); ERA5 reanalysis, "
               "Copernicus Climate Change Service; UV index, Copernicus "
               "Atmosphere Monitoring Service")

#: Where each series is read.  One point per area: the areas are smaller than a
#: reanalysis grid cell, so a second point would return the same cell.
SITES: Dict[str, Dict] = {
    "dhaka": {"latitude": 23.7504, "longitude": 90.3785, "timezone": "Asia/Dhaka"},
    "wyndham": {"latitude": -37.8900, "longitude": 144.7000,
                "timezone": "Australia/Melbourne"},
}

DAILY = ("temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
         "precipitation_sum", "precipitation_hours")
HOURLY = ("temperature_2m", "relative_humidity_2m", "precipitation",
          "shortwave_radiation", "cloud_cover", "wind_speed_10m")


def _get(url: str, query: Dict) -> Dict:
    request = urllib.request.Request(
        url + "?" + urllib.parse.urlencode(query),
        headers={"User-Agent": "wastebins-research/1.0 (academic)"})
    with urllib.request.urlopen(request, timeout=120) as handle:
        return json.loads(handle.read().decode("utf-8"))


def _store(name: str, frame: pd.DataFrame, request: Dict, answer: Dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_DIR / f"{name}.csv", index=False)
    (DATA_DIR / f"{name}.meta.json").write_text(json.dumps({
        "name": name, "request": request, "rows": len(frame),
        "grid_latitude": answer.get("latitude"),
        "grid_longitude": answer.get("longitude"),
        "elevation_m": answer.get("elevation"),
        "units": answer.get("daily_units") or answer.get("hourly_units"),
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attribution": ATTRIBUTION,
    }, indent=2))


def daily(site: str, start: str, end: str,
          variables: Sequence[str] = DAILY) -> pd.DataFrame:
    """Daily series for one site, one row per local calendar day."""
    name = f"{site}_daily_{start}_{end}"
    path = DATA_DIR / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    spec = SITES[site]
    query = {"latitude": spec["latitude"], "longitude": spec["longitude"],
             "start_date": start, "end_date": end,
             "daily": ",".join(variables), "timezone": spec["timezone"]}
    answer = _get(ARCHIVE_URL, query)
    frame = pd.DataFrame(answer["daily"]).rename(columns={"time": "date"})
    _store(name, frame, {"url": ARCHIVE_URL, **query}, answer)
    return pd.read_csv(path, parse_dates=["date"])


def hourly(site: str, start: str, end: str,
           variables: Sequence[str] = HOURLY, uv: bool = True) -> pd.DataFrame:
    """Hourly series for one site in local time, with the UV index when held."""
    name = f"{site}_hourly_{start}_{end}"
    path = DATA_DIR / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["time"])
    spec = SITES[site]
    query = {"latitude": spec["latitude"], "longitude": spec["longitude"],
             "start_date": start, "end_date": end,
             "hourly": ",".join(variables), "timezone": spec["timezone"]}
    answer = _get(ARCHIVE_URL, query)
    frame = pd.DataFrame(answer["hourly"])
    if uv:
        # The reanalysis carries no UV index.  The atmospheric-composition
        # service does, from August 2022 onwards.
        try:
            extra = _get(AIR_QUALITY_URL, {**query, "hourly": "uv_index"})
            frame = frame.merge(pd.DataFrame(extra["hourly"]), on="time", how="left")
        except Exception as exc:                       # outside its coverage
            print(f"  no UV index for {site} {start}..{end}: {exc}")
    _store(name, frame, {"url": ARCHIVE_URL, **query}, answer)
    return pd.read_csv(path, parse_dates=["time"])


def main() -> int:
    # The whole Wyndham fill record, for the demand regression.
    w = daily("wyndham", "2018-06-25", "2021-05-03")
    print(f"wyndham daily: {len(w)} days, max temperature "
          f"{w['temperature_2m_max'].min():.1f} to {w['temperature_2m_max'].max():.1f} C, "
          f"{int((w['precipitation_sum'] >= 1.0).sum())} days with at least 1 mm")
    # The hottest month on record for Dhaka in the archive period used here.
    d = hourly("dhaka", "2024-04-01", "2024-04-30")
    print(f"dhaka hourly April 2024: {len(d)} hours, temperature "
          f"{d['temperature_2m'].min():.1f} to {d['temperature_2m'].max():.1f} C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
