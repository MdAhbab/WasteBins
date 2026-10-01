"""
Does weather move the fill rate of a public container?  A measurement.
======================================================================

The weather scenarios change how fast vehicles travel and how long a stop
takes, and both effects come from the literature.  Whether weather also changes
how fast a container fills is a question about demand, and the study holds one
record that can answer it: 966 days of daily fill readings from 33 public
containers in Wyndham, Victoria.  This script joins that record to the daily
weather of the same place and estimates the effect.

What is estimated
-----------------
The unit is one container on one pair of consecutive days.  The outcome is the
rise in fill between the two readings, as a share of the container.  A pair is
kept when the fill did not fall, since a fall is a collection, and when the
first reading was at most 0.6, so that the top of the scale does not censor the
rise.  This is the definition of a fill rate that the demand model of the
rollout already uses.

The readings carry a date and no time of day, so it is not known how the
interval between two readings divides between the two calendar days.  The
weather attached to a pair is therefore the mean over both days: the mean of
the two daily maximum temperatures, and whether the two days together received
at least 2 mm of rain.  That choice was made before any estimate was computed.
Results under the other two alignments are printed beside it and are not used.

The model is a Poisson regression with a log link, fitted by pseudo maximum
likelihood, so a coefficient is a proportional change in the fill rate and the
many zero rises need no special treatment.  It includes a fixed effect for each
container, the day of the week, and an indicator for the period of pandemic
restrictions in Victoria from 16 March 2020.  Standard errors are clustered by
date, because every container shares the weather of a day.

Run:  python -m experiments.exp_weather_demand
Out:  results/weather_demand.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import weather_data as WD        # noqa: E402
from experiments import wyndham as WY             # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parent / "results"

MAX_START_FILL = 0.6          # room for two steps of the six-level scale
WET_MM = 2.0                  # two-day rain total that makes a pair "wet"
TEMPERATURE_REF_C = 20.0
RESTRICTIONS_FROM = "2020-03-16"


def panel() -> pd.DataFrame:
    """One row per container and pair of consecutive days, with its weather."""
    containers, fill = WY.load()
    wide = WY.fill_matrix(containers, fill)
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    weather = WD.daily("wyndham", "2018-06-25", "2021-05-03").set_index("date")

    stream = dict(zip(containers["serial"], containers["stream"]))
    rows: List[Dict] = []
    for serial in wide.columns:
        series = wide[serial].dropna()
        dates = series.index
        values = series.values
        for k in range(1, len(series)):
            if (dates[k] - dates[k - 1]).days != 1:
                continue
            rise = float(values[k] - values[k - 1])
            if rise < 0 or values[k - 1] > MAX_START_FILL + 1e-9:
                continue
            first, second = dates[k - 1], dates[k]
            if first not in weather.index or second not in weather.index:
                continue
            a, b = weather.loc[first], weather.loc[second]
            rows.append({
                "serial": int(serial), "date": second, "rise": rise,
                "stream": stream[serial],
                "tmax_both": 0.5 * (a["temperature_2m_max"] + b["temperature_2m_max"]),
                "rain_both": float(a["precipitation_sum"] + b["precipitation_sum"]),
                "tmax_first": float(a["temperature_2m_max"]),
                "rain_first": float(a["precipitation_sum"]),
                "tmax_second": float(b["temperature_2m_max"]),
                "rain_second": float(b["precipitation_sum"]),
            })
    frame = pd.DataFrame(rows)
    frame["dow"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    frame["restrictions"] = (frame["date"] >= RESTRICTIONS_FROM).astype(float)
    return frame


def _design(frame: pd.DataFrame, temperature: str, wet: np.ndarray,
            month_effects: bool) -> (np.ndarray, List[str]):
    columns = [np.ones(len(frame))]
    names = ["intercept"]
    columns.append((frame[temperature].values - TEMPERATURE_REF_C) / 10.0)
    names.append("temperature_per_10C")
    columns.append(wet.astype(float))
    names.append("wet")
    if frame["restrictions"].nunique() > 1:
        columns.append(frame["restrictions"].values)
        names.append("restrictions")
    for d in range(1, 7):
        columns.append((frame["dow"].values == d).astype(float))
        names.append(f"dow_{d}")
    if month_effects:
        for m in range(2, 13):
            columns.append((frame["month"].values == m).astype(float))
            names.append(f"month_{m}")
    serials = sorted(frame["serial"].unique())
    for s in serials[1:]:
        columns.append((frame["serial"].values == s).astype(float))
        names.append(f"container_{s}")
    return np.column_stack(columns), names


def poisson_clustered(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict:
    """Poisson pseudo maximum likelihood with a cluster-robust covariance."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-9))
    for _ in range(200):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        XtW = X.T * mu
        new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    mu = np.exp(np.clip(X @ beta, -30.0, 30.0))
    bread = np.linalg.inv((X.T * mu) @ X)
    score = X * (y - mu)[:, None]
    order = np.argsort(groups, kind="stable")
    _, starts = np.unique(groups[order], return_index=True)
    sums = np.add.reduceat(score[order], starts, axis=0)
    g = len(starts)
    meat = sums.T @ sums * (g / (g - 1.0))
    cov = bread @ meat @ bread
    return {"beta": beta, "se": np.sqrt(np.diag(cov)), "clusters": int(g), "mu": mu}


def fit(frame: pd.DataFrame, temperature: str = "tmax_both", rain: str = "rain_both",
        wet_mm: float = WET_MM, month_effects: bool = False) -> Optional[Dict]:
    # A container that never rose carries no information and its fixed effect
    # has no finite estimate.
    active = frame.groupby("serial")["rise"].transform("sum") > 0
    frame = frame[active].reset_index(drop=True)
    if len(frame) < 200:
        return None
    wet = frame[rain].values >= wet_mm
    X, names = _design(frame, temperature, wet, month_effects)
    y = frame["rise"].values
    groups = frame["date"].values.astype("datetime64[D]").astype(np.int64)
    result = poisson_clustered(X, y, groups)

    def effect(name: str) -> Dict:
        k = names.index(name)
        b, s = float(result["beta"][k]), float(result["se"][k])
        zstat = b / s
        return {
            "coefficient": b, "standard_error": s, "z": zstat,
            "p": float(2.0 * stats.norm.sf(abs(zstat))),
            "percent_change": 100.0 * (np.exp(b) - 1.0),
            "percent_change_ci95": [100.0 * (np.exp(b - 1.959964 * s) - 1.0),
                                    100.0 * (np.exp(b + 1.959964 * s) - 1.0)],
        }

    out = {
        "observations": int(len(frame)),
        "containers": int(frame["serial"].nunique()),
        "dates": result["clusters"],
        "mean_rise_per_day": float(y.mean()),
        "share_zero_rise": float((y == 0).mean()),
        "share_wet": float(wet.mean()),
        "temperature_range_C": [float(frame[temperature].min()),
                                float(frame[temperature].max())],
        "temperature_per_10C": effect("temperature_per_10C"),
        "wet": effect("wet"),
    }
    if "restrictions" in names:
        out["restrictions"] = effect("restrictions")
    return out


def main() -> int:
    frame = panel()
    print(f"{len(frame)} container-day pairs, {frame['serial'].nunique()} containers, "
          f"{frame['date'].nunique()} dates, mean rise {frame['rise'].mean():.4f} per day")

    before = frame[frame["date"] < RESTRICTIONS_FROM]
    variants = {
        "primary": fit(frame),
        "with_month_effects": fit(frame, month_effects=True),
        "before_restrictions": fit(before),
        "before_restrictions_month_effects": fit(before, month_effects=True),
        "general_stream": fit(frame[frame["stream"] == "general"]),
        "recycling_stream": fit(frame[frame["stream"] == "recycling"]),
        "werribee": None, "point_cook": None,
        # The two alignments that were not chosen, for transparency only.
        "alignment_first_day": fit(frame, "tmax_first", "rain_first", wet_mm=1.0),
        "alignment_second_day": fit(frame, "tmax_second", "rain_second", wet_mm=1.0),
    }
    containers, _ = WY.load()
    suburb = dict(zip(containers["serial"], containers["suburb"]))
    frame["suburb"] = frame["serial"].map(suburb)
    variants["werribee"] = fit(frame[frame["suburb"] == "Werribee"])
    variants["point_cook"] = fit(frame[frame["suburb"] == "Point Cook"])

    print(f"\n{'variant':<36}{'n':>7}{'per +10 C':>12}{'95% CI':>20}{'p':>9}"
          f"{'wet':>10}{'95% CI':>20}{'p':>9}")
    for name, v in variants.items():
        if v is None:
            print(f"{name:<36} too few observations")
            continue
        t, w = v["temperature_per_10C"], v["wet"]
        print(f"{name:<36}{v['observations']:>7}{t['percent_change']:>11.1f}%"
              f"{'[%.1f, %.1f]' % tuple(t['percent_change_ci95']):>20}{t['p']:>9.4f}"
              f"{w['percent_change']:>9.1f}%"
              f"{'[%.1f, %.1f]' % tuple(w['percent_change_ci95']):>20}{w['p']:>9.4f}")

    payload = {
        "source": {
            "fill": "Wyndham City Council smart-bin fill level, daily, "
                    "Creative Commons Attribution 2.5 Australia",
            "weather": WD.ATTRIBUTION,
            "weather_site": WD.SITES["wyndham"],
        },
        "definition": {
            "outcome": "rise in fill between two consecutive daily readings, share of container",
            "kept": f"pairs with no fall and a first reading of at most {MAX_START_FILL}",
            "temperature": "mean of the two daily maximum temperatures, per 10 C, "
                           f"centred at {TEMPERATURE_REF_C} C",
            "wet": f"the two days together received at least {WET_MM} mm",
            "model": "Poisson pseudo maximum likelihood, log link, container fixed "
                     "effects, day of week, restriction period; standard errors "
                     "clustered by date",
            "restrictions_from": RESTRICTIONS_FROM,
        },
        "variants": variants,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "weather_demand.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {RESULTS / 'weather_demand.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
