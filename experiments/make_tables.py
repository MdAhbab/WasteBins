"""
The tables and the quoted numbers of the paper, written from the results.
=========================================================================

`analyze.py` turns the raw records into ``results/summary.json``.  This script
turns that file, and the few result files that are already summaries, into two
kinds of LaTeX input:

``tables/tab_*.tex``
    One file per results table, included by the manuscript with ``\\input``.

``tables/numbers.tex``
    One macro per number quoted in the running text.  The manuscript writes
    ``\\wdTempPct`` and not ``10.8``, so a number in the text cannot drift from
    the result it reports.

A quantity that has no result yet is written as ``n/a``, so the manuscript
compiles at any stage of the runs and an unfinished study is visible in the PDF.

Run:  python -m experiments.analyze && python -m experiments.make_tables
Out:  ../Paper/tables/  (or the folder given with --out)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, Optional, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import store as ST               # noqa: E402

RESULTS = ST.RESULTS
ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT.parent / "Paper" / "tables"
NA = "n/a"

MACROS: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def load(name: str) -> Optional[Dict]:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def dig(data, *keys, default=None):
    for key in keys:
        if data is None:
            return default
        try:
            data = data[key]
        except (KeyError, IndexError, TypeError):
            return default
    return default if data is None else data


def num(x, digits: int = 1, signed: bool = False) -> str:
    if x is None:
        return NA
    try:
        x = float(x)
    except (TypeError, ValueError):
        return NA
    if not math.isfinite(x):
        return NA
    body = f"{abs(x):,.{digits}f}"
    zero = float(body.replace(",", "")) == 0.0       # never print "-0.0"
    body = body.replace(",", "{,}")
    if x < 0 and not zero:
        return "\\ensuremath{-}" + body              # a true minus in text and in math
    return ("+" if signed and not zero else "") + body


def whole(x) -> str:
    return num(x, 0)


def pvalue(p, symbol: str = "p") -> str:
    """A p-value with its relation, for use inside math mode."""
    if p is None:
        return f"{symbol} = \\text{{{NA}}}"
    p = float(p)
    if p < 0.001:
        return f"{symbol} < 0.001"
    return f"{symbol} = {p:.3f}"


def macro(name: str, value: str) -> None:
    if not name.isalpha():
        raise ValueError(f"a macro name holds letters only: {name!r}")
    if name in MACROS and MACROS[name] != value:
        raise ValueError(f"macro {name} defined twice with different values")
    MACROS[name] = value


def table(path: pathlib.Path, caption: str, label: str, columns: str,
          header: Sequence[str], rows: Sequence[Sequence[str]],
          notes: str = "", small: bool = True, tabcolsep: Optional[str] = None,
          fontsize: Optional[str] = None) -> None:
    lines = ["\\begin{table}[!t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\centering"]
    if fontsize:
        lines.append(f"\\{fontsize}")
    elif small:
        lines.append("\\small")
    if tabcolsep:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    # Wrapped columns are set ragged right, so short lines are not stretched.
    columns = columns.replace("p{", ">{\\raggedright\\arraybackslash}p{")
    lines += [f"\\begin{{tabular}}{{{columns}}}", "\\toprule"]
    lines += list(header)
    lines.append("\\midrule")
    for row in rows:
        lines.append(row if isinstance(row, str) else " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines.append(f"\\par\\smallskip\\footnotesize {notes}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Weather and fill rate
# ---------------------------------------------------------------------------
def weather_demand() -> None:
    data = load("weather_demand.json")
    used = dig(data, "variants", "with_month_effects")
    plain = dig(data, "variants", "primary")
    t, w = dig(used, "temperature_per_10C"), dig(used, "wet")
    macro("wdPairs", whole(dig(used, "observations")))
    macro("wdDates", whole(dig(used, "dates")))
    macro("wdContainers", whole(dig(used, "containers")))
    macro("wdTempPct", num(dig(t, "percent_change"), 1))
    macro("wdTempLo", num(dig(t, "percent_change_ci95", 0), 1))
    macro("wdTempHi", num(dig(t, "percent_change_ci95", 1), 1))
    macro("wdTempZ", num(dig(t, "z"), 2))
    macro("wdTempP", pvalue(dig(t, "p")))
    macro("wdTempCoef", num(dig(t, "coefficient"), 4))
    macro("wdWetPct", num(abs(dig(w, "percent_change", default=float("nan"))), 1))
    macro("wdWetLo", num(abs(dig(w, "percent_change_ci95", 1, default=float("nan"))), 1))
    macro("wdWetHi", num(abs(dig(w, "percent_change_ci95", 0, default=float("nan"))), 1))
    macro("wdWetZ", num(dig(w, "z"), 2))
    macro("wdWetP", pvalue(dig(w, "p")))
    coef = dig(w, "coefficient")
    macro("wdWetCoefSigned",
          NA if coef is None else f"{'-' if coef < 0 else '+'}\\, {abs(coef):.4f}")
    macro("wdTempPctNoMonth", num(dig(plain, "temperature_per_10C", "percent_change"), 1))
    for tag, key in (("General", "general_stream"), ("Recycling", "recycling_stream"),
                     ("Before", "before_restrictions_month_effects")):
        sub = dig(data, "variants", key, "temperature_per_10C")
        macro(f"wd{tag}Pct", num(dig(sub, "percent_change"), 1))
        macro(f"wd{tag}P", pvalue(dig(sub, "p")))
    macro("wdWetPctNoMonth",
          num(abs(dig(plain, "wet", "percent_change", default=float("nan"))), 1))


def weather_demand_table(out: pathlib.Path) -> None:
    data = load("weather_demand.json")
    names = [
        ("with_month_effects", "All, month effects (used)"),
        ("primary", "All, no month effects"),
        ("before_restrictions_month_effects", "Pre-restriction, month effects"),
        ("before_restrictions", "Pre-restriction, no month effects"),
        ("general_stream", "General waste only"),
        ("recycling_stream", "Recycling only"),
        ("werribee", "Werribee only"),
        ("point_cook", "Point Cook only"),
        ("alignment_first_day", "First-day weather only"),
        ("alignment_second_day", "Second-day weather only"),
    ]

    def bare_p(p) -> str:
        # The column is headed $p$, so the cell holds the value alone.
        text = pvalue(p)
        return "$" + text.replace("p = ", "").replace("p ", "") + "$"

    rows = []
    for key, label in names:
        v = dig(data, "variants", key)
        t, w = dig(v, "temperature_per_10C"), dig(v, "wet")
        rows.append([
            label, whole(dig(v, "observations")),
            num(dig(t, "percent_change"), 1, signed=True),
            f"[{num(dig(t, 'percent_change_ci95', 0), 1, signed=True)}, "
            f"{num(dig(t, 'percent_change_ci95', 1), 1, signed=True)}]",
            bare_p(dig(t, "p")),
            num(dig(w, "percent_change"), 1, signed=True),
            f"[{num(dig(w, 'percent_change_ci95', 0), 1, signed=True)}, "
            f"{num(dig(w, 'percent_change_ci95', 1), 1, signed=True)}]",
            bare_p(dig(w, "p")),
        ])
    table(out / "tab_weather_demand.tex",
          "Change in fill rate with weather, Wyndham record.",
          "tab:weatherdemand", "lrrlrrlr",
          ["& & \\multicolumn{3}{c}{Per 10\\,$^\\circ$C of daily maximum} & "
           "\\multicolumn{3}{c}{Wet pair of days} \\\\",
           "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
           "Sample and model & Pairs & Change & Interval & $p$ & Change & Interval & "
           "$p$ \\\\"],
          rows, tabcolsep="3.5pt", fontsize="footnotesize",
          notes="Changes in percent with 95 percent intervals; standard errors "
                "clustered by date. Pre-restriction: before the pandemic "
                "restrictions in Victoria from 16 March 2020.")


def weather_live() -> None:
    """The dated demonstration on a live reading: derived factors and plan outcome."""
    data = load("weather_live.json")
    google = dig(data, "sources", "google", default={}) or {}
    effects = google.get("effects", {})
    stamp = str(dig(data, "retrieved_utc", default="") or "")
    months = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
    if len(stamp) >= 16:
        macro("liveDate", f"{int(stamp[8:10])} {months[int(stamp[5:7]) - 1]} {stamp[:4]}")
        macro("liveTime", stamp[11:16])
    else:
        macro("liveDate", NA)
        macro("liveTime", NA)
    macro("liveRain", str(effects.get("rain_class", NA)))
    macro("liveService", num(effects.get("service_factor"), 2))
    macro("liveGas", num(effects.get("gas_factor"), 2))
    macro("liveLitter", num(effects.get("litter_demand_factor"), 2))
    snaps = google.get("snapshots", [])
    macro("liveSnapshots", whole(len(snaps)))
    macro("liveNominalDropped", whole(sum(s["nominal_plan"]["dropped"] for s in snaps)))
    macro("liveLiveDropped", whole(sum(s["live_plan"]["dropped"] for s in snaps)))


def weather_scenarios(out: pathlib.Path) -> None:
    """The factors each cycle type of each scenario produces."""
    data = load("weather_scenarios.json")
    rows = []
    names = {"rain": "Rain", "storm": "Storm", "heat": "Heat", "onset": "Onset"}
    labels = {"day": "day shift", "night": "night shift", "storm peak day": "peak day shift"}
    for scenario in ("rain", "storm", "heat", "onset"):
        for label, r in dig(data, "scenarios", scenario, default={}).items():
            cap = (r.get("speed_cap_kmh") or [None])[0]
            classes = r.get("rain_class") or [NA]
            rain = (classes[0] if len(set(classes)) == 1
                    else f"{classes[0]}, then {classes[-1]}")
            rows.append([
                f"{names[scenario]}, {labels.get(label, label)}",
                rain,
                f"{num(r.get('speed_factor_at_dispatch'), 2)} / "
                f"{num(r.get('speed_factor_shift_mean'), 2)}",
                "none" if cap is None else num(cap, 1),
                f"{num(r.get('service_factor_at_dispatch'), 2)} / "
                f"{num(r.get('service_factor_shift_mean'), 2)}",
                num(r.get("gas_factor"), 2), num(r.get("litter_demand_factor"), 2)])
    if data:
        days = dig(data, "heat_days", "days", default=[])
        macro("heatDailyMax", num(dig(data, "heat_days", "daily_max_c"), 1))
        macro("heatDays", whole(len(days)))
    table(out / "tab_scenarios.tex",
          "Weather scenario factors during the spell.",
          "tab:scenarios", "p{0.24\\textwidth}lrrrrr",
          ["Scenario and cycle & Rain & Speed & Speed limit & Service time & Gas & "
           "Litter-bin \\\\",
           " & class & factor & (km\\,h$^{-1}$) & factor & factor & fill rate \\\\"],
          rows, tabcolsep="4pt",
          notes="Factors at dispatch and as a mean over the shift, relative to "
                "28\\,$^\\circ$C, 60 percent humidity and dry streets.")


# ---------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------
def emissions(out: pathlib.Path) -> None:
    data = load("emissions.json")
    macro("speedMin", num(dig(data, "range", "speed_kmh", "min"), 1))
    macro("speedMax", num(dig(data, "range", "speed_kmh", "max"), 1))
    macro("emisLegMin", num(dig(data, "range", "co2_kg_per_km", "min"), 2))
    macro("emisLegMax", num(dig(data, "range", "co2_kg_per_km", "max"), 2))
    macro("emisLegRatio", num(dig(data, "range", "co2_kg_per_km", "ratio_max_to_min"), 1))
    macro("emisMeasured", num(dig(data, "measured", "co2_kg_per_km"), 2))
    macro("emisMeasuredFuel", num(dig(data, "measured", "fuel_l_per_100km"), 1))
    macro("emisMeasuredSpeed", num(dig(data, "measured", "mean_speed_kmh"), 1))
    rows = []
    for key in ("transfer_empty", "transfer_full", "collection_light", "collection_heavy"):
        r = dig(data, "duty", key, default={})
        setting = (f"{num(r.get('distance_km'), 1)} km, {whole(r.get('payload_kg'))} kg, "
                   f"$x = {num(r.get('congestion'), 1)}$"
                   + (f", {whole(r.get('service_min'))} min stop" if r.get("lifts") else ""))
        rows.append([
            r.get("label", NA), setting, num(r.get("speed_kmh"), 1),
            num(r.get("co2_kg_per_km"), 2), num(r.get("fuel_l_per_100km"), 1),
            whole(100 * r.get("share_cruise", float("nan"))),
            whole(100 * r.get("share_stop_go", float("nan"))),
            whole(100 * r.get("share_idle", float("nan"))),
            whole(100 * r.get("share_compaction", float("nan"))),
        ])
    table(out / "tab_emissions.tex",
          "Emission model at stated settings.",
          "tab:emissions", "p{0.25\\textwidth}p{0.23\\textwidth}rrrrrrr",
          ["Leg & Settings & km\\,h$^{-1}$ & kg\\,km$^{-1}$ & L per 100 km & "
           "Cruise & Stop-go & Idle & Lift \\\\",
           " & & & CO$_2$ & & \\% & \\% & \\% & \\% \\\\"],
          rows, tabcolsep="3pt",
          notes="Shares are of the fuel used on the leg. A stop is one lift of 150~kg.")


# ---------------------------------------------------------------------------
# Planner settings
# ---------------------------------------------------------------------------
PRETTY = {
    "first_solution": "first solution", "metaheuristic": "metaheuristic",
    "n_ants": "ants", "beta": "$\\beta$", "polish_fraction": "local-search share",
    "population_size": "population", "mutation_rate": "mutation rate",
    "PATH_CHEAPEST_ARC": "path cheapest arc",
    "PARALLEL_CHEAPEST_INSERTION": "parallel cheapest insertion",
    "GUIDED_LOCAL_SEARCH": "guided local search", "TABU_SEARCH": "tabu search",
    "SIMULATED_ANNEALING": "simulated annealing",
}
TUNED = {"ortools": ("first_solution", "metaheuristic"),
         "aco": ("n_ants", "beta", "polish_fraction"),
         "genetic": ("population_size", "mutation_rate", "polish_fraction")}
FIXED = {"ortools": "arc emissions at mean load",
         "aco": "$\\alpha = 1$, evaporation 0.10, $q_0 = 0.25$",
         "genetic": "crossover 0.85, elite 4, tournament 3"}
NAMES = {"ortools": "OR-Tools", "aco": "Ant colony", "genetic": "Genetic algorithm"}


def _setting(config: Dict, keys: Sequence[str]) -> str:
    parts = []
    for key in keys:
        value = config.get(key)
        shown = PRETTY.get(value, f"{value:g}" if isinstance(value, (int, float)) else str(value))
        parts.append(f"{PRETTY.get(key, key)} {shown}")
    return ", ".join(parts)


def planner_config(out: pathlib.Path) -> None:
    data = load("tuning.json")
    rows = []
    for solver in ("ortools", "aco", "genetic"):
        grid = dig(data, "grid", solver, default=[])
        if grid:
            best, worst = grid[0], grid[-1]
            chosen = _setting(best["config"], TUNED[solver])
            spread = (f"{num(best['mean_objective'], 0)} to {num(worst['mean_objective'], 0)}")
            count = str(len(grid))
        else:
            chosen, spread, count = NA, NA, NA
        rows.append([NAMES[solver], count, chosen, spread, FIXED[solver]])
        key = {"ortools": "Ort", "aco": "Aco", "genetic": "Gen"}[solver]
        macro(f"tune{key}Best", num(grid[0]["mean_objective"], 0) if grid else NA)
        macro(f"tune{key}Worst", num(grid[-1]["mean_objective"], 0) if grid else NA)
        macro(f"tune{key}Settings", count)
    insertion = dig(data, "insertion_objective", default=[])
    budgets = dig(data, "matched_budget_s", default=[])
    macro("tuneInsObjective", num(sum(insertion) / len(insertion), 0) if insertion else NA)
    macro("tuneBudget", num(sum(budgets) / len(budgets), 0) if budgets else NA)
    table(out / "tab_config.tex",
          "Planner settings and tuning grids.",
          "tab:config", "lrp{0.28\\textwidth}p{0.13\\textwidth}p{0.21\\textwidth}",
          ["Planner & Grid & Setting selected & Mean objective & Fixed parameters \\\\",
           " & size & & best to worst & \\\\"],
          rows, tabcolsep="4pt",
          notes="Each grid was run on three tuning instances at the matched time; the "
                "setting with the lowest mean objective was kept.")


def tuning_grid(out: pathlib.Path) -> None:
    """Every setting of every grid with its objective on each tuning instance."""
    data = load("tuning.json")
    rows = []
    for solver in ("ortools", "aco", "genetic"):
        grid = dig(data, "grid", solver, default=[]) or []
        if rows and grid:
            rows.append("\\addlinespace")
        for index, entry in enumerate(grid):
            rows.append([NAMES[solver] if index == 0 else "",
                         _setting(entry["config"], TUNED[solver]),
                         *[num(x, 0) for x in entry.get("objectives", [])],
                         num(entry["mean_objective"], 0)])
    if not rows:
        rows = [["\\multicolumn{6}{l}{\\emph{The tuning runs have not been made.}}"]]
    table(out / "tab_tuning.tex", "Objective of every setting on the tuning instances.",
          "tab:tuning", "lp{0.38\\textwidth}rrrr",
          ["Planner & Setting & Instance 1 & Instance 2 & Instance 3 & Mean \\\\"],
          rows, tabcolsep="4pt", fontsize="footnotesize",
          notes="Set T160 at the matched time; settings in order of mean objective, "
                "the first one kept.")


def prize_weight(out: pathlib.Path) -> None:
    """The prize-weight check on the held-out set T60, from the raw records."""
    records = ST.read("tune-lambda")
    values = sorted({r["lambda_prize"] for r in records})
    rows, previous = [], None
    for value in values:
        group = [r for r in records if r["lambda_prize"] == value]
        served = sum(r["served"] for r in group) / len(group)
        km = sum(r["km"] for r in group) / len(group)
        marginal = ("" if previous is None
                    else NA if served <= previous[0]
                    else num((km - previous[1]) / (served - previous[0]), 2))
        rows.append([num(value, 0), num(served, 1), num(km, 1),
                     num(km / served, 2), marginal])
        previous = (served, km)
    snapshots = len({r["snapshot"] for r in records}) if records else 0
    macro("lambdaSnapshots", whole(snapshots) if records else NA)
    table(out / "tab_lambda.tex",
          "Prize weight $\\lambda$ on the held-out set T60.",
          "tab:lambda", "rrrrr",
          ["$\\lambda$ & Served & km & km per container & Extra km per extra \\\\",
           " & & & served & container served \\\\"],
          rows,
          notes=f"Insertion planner, full rule; means over {snapshots} snapshots of 60 "
                f"containers and four vehicles.")


# ---------------------------------------------------------------------------
# Street network
# ---------------------------------------------------------------------------
def network(out: pathlib.Path) -> None:
    data = load("network.json")
    d, w = dig(data, "areas", "dhaka"), dig(data, "areas", "wyndham")

    def pct(area, *keys, digits=1, signed=True):
        value = dig(area, *keys)
        return num(None if value is None else 100.0 * value, digits, signed=signed)

    rows = [
        ["Ordered pairs", whole(dig(d, "pairs_kept")), whole(dig(w, "pairs_kept"))],
        ["Circuity, mean", num(dig(d, "circuity", "mean"), 3), num(dig(w, "circuity", "mean"), 3)],
        ["Circuity, median", num(dig(d, "circuity", "median"), 3),
         num(dig(w, "circuity", "median"), 3)],
        ["Circuity, 90th percentile", num(dig(d, "circuity", "p90"), 3),
         num(dig(w, "circuity", "p90"), 3)],
        ["Circuity, 99th percentile", num(dig(d, "circuity", "p99"), 3),
         num(dig(w, "circuity", "p99"), 3)],
        "\\midrule",
        ["Error of 1.30, mean (\\%)", pct(d, "constant_factor_error", "relative", "mean"),
         pct(w, "constant_factor_error", "relative", "mean")],
        ["Error, 10th percentile (\\%)", pct(d, "constant_factor_error", "relative", "p10"),
         pct(w, "constant_factor_error", "relative", "p10")],
        ["Error, 90th percentile (\\%)", pct(d, "constant_factor_error", "relative", "p90"),
         pct(w, "constant_factor_error", "relative", "p90")],
        ["Pairs understated (\\%)",
         pct(d, "constant_factor_error", "share_underestimated", digits=0, signed=False),
         pct(w, "constant_factor_error", "share_underestimated", digits=0, signed=False)],
        "\\midrule",
        ["Direction asymmetry, mean (\\%)", pct(d, "asymmetry", "mean", signed=False),
         pct(w, "asymmetry", "mean", signed=False)],
        ["Direction asymmetry, 90th percentile (\\%)", pct(d, "asymmetry", "p90", signed=False),
         pct(w, "asymmetry", "p90", signed=False)],
    ]
    table(out / "tab_circuity.tex",
          "Circuity and error of a constant detour factor of 1.30.",
          "tab:circuity", "lrr", [" & Dhaka & Wyndham \\\\"], rows,
          notes="Ordered pairs at least 100~m apart. A negative error means the "
                "constant understates road distance.")

    for tag, area in (("Dhaka", d), ("Wyndham", w)):
        macro(f"net{tag}CircMean", num(dig(area, "circuity", "mean"), 3))
        mean_error = dig(area, "constant_factor_error", "relative", "mean")
        macro(f"net{tag}ErrMeanAbs",
              num(None if mean_error is None else abs(100.0 * mean_error), 1))
        macro(f"net{tag}ErrLow", pct(area, "constant_factor_error", "relative", "p10"))
        macro(f"net{tag}ErrHigh", pct(area, "constant_factor_error", "relative", "p90"))
        macro(f"net{tag}Under", pct(area, "constant_factor_error", "share_underestimated",
                                    digits=0, signed=False))
        macro(f"net{tag}Asym", pct(area, "asymmetry", "mean", signed=False))
        macro(f"net{tag}ShortPairs", whole(dig(area, "short_pairs_under_100m", "pairs")))
        macro(f"net{tag}ShortRoad", whole(dig(area, "short_pairs_under_100m", "mean_road_m")))
        macro(f"net{tag}ShortMax", whole(dig(area, "short_pairs_under_100m", "max_road_m")))
        macro(f"net{tag}Snap", whole(dig(area, "snap_distance_m", "max")))
        macro(f"net{tag}Tagged", pct(area, "graph", "tagged_maxspeed_share", signed=False))
    def box(area, axis):
        b = dig(area, "graph", "bbox")
        if not b:
            return NA
        south, west, north, east = b
        if axis == "lat":
            low, high = sorted((abs(south), abs(north)))
            return f"{low:.3f} to {high:.3f}\\,{'N' if south >= 0 else 'S'}"
        return f"{west:.3f} to {east:.3f}\\,E"

    def fetched(area):
        stamp = dig(area, "graph", "fetched_utc")
        return stamp[:10] if stamp else NA

    rows = [
        ["Bounding box, latitude", box(d, "lat"), box(w, "lat")],
        ["Bounding box, longitude", box(d, "lon"), box(w, "lon")],
        ["Retrieved (UTC date)", fetched(d), fetched(w)],
        ["Ways parsed", whole(dig(d, "graph", "ways_parsed")),
         whole(dig(w, "graph", "ways_parsed"))],
        ["Strongly connected components before reduction",
         whole(dig(d, "graph", "strong_components_before_trim")),
         whole(dig(w, "graph", "strong_components_before_trim"))],
        ["Nodes after reduction", whole(dig(d, "graph", "nodes")),
         whole(dig(w, "graph", "nodes"))],
        ["Directed arcs after reduction", whole(dig(d, "graph", "arcs")),
         whole(dig(w, "graph", "arcs"))],
        ["One-way arcs (\\%)", pct(d, "graph", "oneway_arc_share", signed=False),
         pct(w, "graph", "oneway_arc_share", signed=False)],
        ["Ways with a speed tag (\\%)", pct(d, "graph", "tagged_maxspeed_share", signed=False),
         pct(w, "graph", "tagged_maxspeed_share", signed=False)],
        ["Factor on class default speeds", num(dig(d, "graph", "speed_scale"), 2),
         num(dig(w, "graph", "speed_scale"), 2)],
        ["Free-flow leg speed, mean (km\\,h$^{-1}$)", num(dig(d, "freeflow_kmh", "mean"), 1),
         num(dig(w, "freeflow_kmh", "mean"), 1)],
        ["Leg speed at 07:30 on a Tuesday, mean (km\\,h$^{-1}$)",
         num(dig(d, "leg_speed_kmh", "mean"), 1), num(dig(w, "leg_speed_kmh", "mean"), 1)],
        ["Leg speed at 07:30 on a Tuesday, range (km\\,h$^{-1}$)",
         f"{num(dig(d, 'leg_speed_kmh', 'min'), 1)} to {num(dig(d, 'leg_speed_kmh', 'max'), 1)}",
         f"{num(dig(w, 'leg_speed_kmh', 'min'), 1)} to {num(dig(w, 'leg_speed_kmh', 'max'), 1)}"],
        ["Containers routed", whole((dig(d, "n_stops") or 1) - 1),
         whole((dig(w, "n_stops") or 1) - 1)],
        ["Largest snap distance (m)", whole(dig(d, "snap_distance_m", "max")),
         whole(dig(w, "snap_distance_m", "max"))],
        ["Arc-list fingerprint", f"\\texttt{{{dig(d, 'graph', 'fingerprint', default=NA)}}}",
         f"\\texttt{{{dig(w, 'graph', 'fingerprint', default=NA)}}}"],
    ]
    table(out / "tab_graphs.tex",
          "Street graphs of the two study areas.",
          "tab:graphs", "p{0.36\\textwidth}p{0.26\\textwidth}p{0.26\\textwidth}",
          [" & Dhaka & Wyndham \\\\"], rows,
          notes="After reduction to the largest strongly connected component.")

    macro("netDhakaCircMin", num(dig(d, "circuity", "min"), 2))
    macro("netDhakaCircPninetynine", num(dig(d, "circuity", "p99"), 2))
    macro("netDhakaLongestLeg", num(dig(d, "longest_leg", "road_km"), 2))
    touches = dig(d, "longest_leg", "touches_depot")
    macro("netDhakaLongestLegDepot", NA if touches is None else ("touches" if touches
                                                                 else "does not touch"))
    trips = dig(d, "round_trips", default={})
    macro("tripNotServable", whole(dig(trips, "cost", "not_servable_alone", "street")))
    macro("tripLongest", num(dig(trips, "longest", "road_km"), 2))
    macro("tripLongestOut", num(dig(trips, "longest", "out_km"), 2))
    macro("tripLongestBack", num(dig(trips, "longest", "back_km"), 2))
    macro("tripLongestErr", num(abs(100.0 * dig(trips, "longest", "relative_error",
                                                default=float("nan"))), 1))
    macro("tripCostMax", num(dig(trips, "cost", "largest", "street"), 1))
    macro("tripCostMaxConstant", num(dig(trips, "cost", "largest", "constant"), 1))
    macro("tripCostMaxWait", num(dig(trips, "cost", "largest", "price_wait_h"), 1))
    macro("tripCostMaxWaitConstant",
          num(dig(trips, "cost", "largest", "price_wait_constant_h"), 1))
    macro("tripCostMedian", num(dig(trips, "cost", "street", "median"), 1))
    macro("tripAboveHalf", whole(dig(trips, "cost", "above_half_ceiling", "street")))
    macro("tripUnderstated", pct(trips, "share_understated", digits=0, signed=False))
    macro("tripErrLow", pct(trips, "constant_relative_error", "p10"))
    macro("tripErrHigh", pct(trips, "constant_relative_error", "p90"))


# ---------------------------------------------------------------------------
# Controlled validation of the propositions
# ---------------------------------------------------------------------------
def wait_bound(out: pathlib.Path) -> None:
    data = load("wait_bound.json")
    sweep = dig(data, "price_sweep", default=[])
    rows = [[num(r["distance_km"], 0), num(r["round_trip_cost"], 1),
             num(r["price_threshold_h"], 1), num(r["predicted_h"], 0),
             num(r["served_at_h_price_only"], 0), num(r["served_at_h_reserved"], 0)]
            for r in sweep]
    table(out / "tab_pricesweep.tex",
          "Price sweep: one overdue container at eight distances.",
          "tab:pricesweep", "rrrrrr",
          ["Distance & Round-trip & Threshold of & Predicted & "
           "\\multicolumn{2}{c}{Observed wait at collection} \\\\",
           "\\cmidrule(lr){5-6}",
           "(km) & cost $C_i$ & $2\\tau C_i/(\\lambda\\mu)$ (h) & (h) & "
           "No reservation & Reserved \\\\"],
          rows,
          notes="One free vehicle. Waits in hours, with and without a reserved head.")
    macro("sweepExact", whole(dig(data, "n_price_exact")))
    macro("sweepCases", whole(dig(data, "n_price_cases")))
    macro("sweepCeilingKm", num(dig(data, "bounded_penalty", "first_unaffordable_km"), 0))
    macro("sweepLongest", num(max([r["served_at_h_price_only"] for r in sweep] or [float("nan")]), 0))
    macro("tightCases", whole(dig(data, "n_tight_cases")))
    macro("tightAttained", whole(dig(data, "n_tight")))
    tight = dig(data, "tightness", default=[])
    rows = [[whole(r["containers"]), whole(r["reserved_per_cycle"]), whole(r["shift_min"]),
             num(r["bound_h"], 0), num(r["worst_wait_at_service_h"], 0)] for r in tight]
    table(out / "tab_tightness.tex",
          "Tightness of the waiting-time bound.",
          "tab:tightness", "rrrrr",
          ["Containers & Reserved $r$ & Shift (min) & Bound $W$ (h) & Longest wait (h) \\\\"],
          rows,
          notes="Each instance lets the fleet serve exactly $r$ containers per cycle.")


# ---------------------------------------------------------------------------
# Software and hardware
# ---------------------------------------------------------------------------
def environment() -> None:
    """Versions and processor, as stored with the runs, main comparison first."""
    metas = sorted(ST.RAW.glob("*.meta.json"),
                   key=lambda p: (p.name != "main.meta.json", p.name))
    env = None
    for path in metas:
        env = dig(json.loads(path.read_text()), "environment")
        if env:
            break
    versions = dig(env, "versions", default={})
    for name, module in (("Python", "python"), ("Ortools", "ortools"),
                         ("Numpy", "numpy"), ("Scipy", "scipy")):
        macro(f"env{name}", str(versions.get(module) or NA))
    macro("envCpus", whole(dig(env, "logical_cpus")))
    # A second machine, where some studies ran.  Each study ran on one machine.
    machine = (dig(env, "processor"), dig(env, "logical_cpus"))
    other, studies = None, []
    for path in metas:
        e = dig(json.loads(path.read_text()), "environment")
        if e and (e.get("processor"), e.get("logical_cpus")) != machine:
            other = other or e
            studies.append(path.name.replace(".meta.json", ""))
    macro("envOtherPython", str(dig(other, "versions", "python") or NA))
    macro("envOtherOrtools", str(dig(other, "versions", "ortools") or NA))
    macro("envOtherCpus", whole(dig(other, "logical_cpus")))
    macro("envOtherStudies", whole(len(studies)) if other else "0")


def write_macros(out: pathlib.Path) -> None:
    lines = ["%% Written by experiments/make_tables.py. Do not edit by hand.",
             "%% Each macro is one number quoted in the text."]
    for name in sorted(MACROS):
        lines.append(f"\\newcommand{{\\{name}}}{{{MACROS[name]}}}")
    (out / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    weather_demand()
    weather_demand_table(out)
    weather_scenarios(out)
    weather_live()
    emissions(out)
    planner_config(out)
    tuning_grid(out)
    prize_weight(out)
    network(out)
    wait_bound(out)
    environment()
    from experiments import make_results_tables
    make_results_tables.build(out, load("summary.json"), macro, table, num, whole,
                              pvalue, dig, NA)
    write_macros(out)
    missing = sorted(k for k, v in MACROS.items() if NA in v)
    print(f"wrote {len(MACROS)} macros and the tables to {out}")
    if missing:
        print(f"{len(missing)} macros have no result yet: {', '.join(missing[:12])}"
              + (" ..." if len(missing) > 12 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
