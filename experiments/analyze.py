"""
From raw records to the numbers the paper reports.
==================================================

Every experiment writes one record per solve or per chain to ``results/raw/``.
This script reads those records and nothing else, and writes
``results/summary.json``.  The tables, the figures and the quoted figures in the
text are all functions of that one file, so a number in the paper can be traced
to the records it came from.

Comparisons are paired.  Two planners are compared on the same snapshots and two
policies on the same networks, so every test is on within-pair differences:
the Wilcoxon signed-rank test, with an exact sign-permutation test when there
are fewer than six pairs.  The p-values of a family of comparisons are adjusted
by the Holm step-down procedure.  The interval on a mean difference is the
bias-corrected and accelerated bootstrap, and the interval on a relative
difference is the percentile bootstrap of the ratio of means, both from 10,000
resamples.  Cliff's delta is reported as the effect size.

A study that has not been run, or has not finished, is reported as far as its
records go, with the number of units it rests on.

Run:  python -m experiments.analyze
Out:  results/summary.json
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import aging as AG            # noqa: E402
from wastebins_core import stats as S             # noqa: E402
from experiments import store as ST               # noqa: E402

RESULTS = ST.RESULTS
REFERENCE = ("insertion", "full")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _num(x, digits: int = 4):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, digits)


def spread(values: Sequence[float]) -> Dict:
    x = np.asarray([v for v in values if v is not None and math.isfinite(float(v))],
                   dtype=float)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size), "mean": _num(x.mean()), "sd": _num(x.std(ddof=1) if x.size > 1 else 0.0),
        "median": _num(np.median(x)), "q25": _num(np.quantile(x, 0.25)),
        "q75": _num(np.quantile(x, 0.75)), "min": _num(x.min()), "max": _num(x.max()),
    }


def paired(a: Sequence[float], b: Sequence[float]) -> Dict:
    """
    ``a`` against ``b`` on the same units, lower being better.

    The relative difference is ``(mean a - mean b) / mean b`` in percent, so a
    negative figure means ``a`` is lower.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = int(min(a.size, b.size))
    if n == 0:
        return {"n_pairs": 0}
    a, b = a[:n], b[:n]
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(10_000, n))
    den = b[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = 100.0 * (a[idx].mean(axis=1) - den) / den
    ratio = ratio[np.isfinite(ratio)]
    test = S.paired_test(a, b)
    difference = S.bootstrap_ci(a - b)
    delta = S.cliffs_delta(a, b)
    mean_b = float(b.mean())
    return {
        "n_pairs": n,
        "mean_a": _num(a.mean()), "mean_b": _num(mean_b),
        "difference": _num(difference["estimate"]),
        "difference_ci": [_num(difference["ci_low"]), _num(difference["ci_high"])],
        "relative_pct": _num(100.0 * (a.mean() - mean_b) / mean_b, 3) if abs(mean_b) > 1e-12 else None,
        "relative_ci": ([_num(np.quantile(ratio, 0.025), 3), _num(np.quantile(ratio, 0.975), 3)]
                        if ratio.size else None),
        "a_lower_in": int((a < b - 1e-9).sum()), "a_higher_in": int((a > b + 1e-9).sum()),
        "test": test["test"], "statistic": _num(test.get("statistic")),
        "p": float(test["p_value"]),
        "cliffs_delta": delta["delta"], "magnitude": delta["magnitude"],
    }


def holm(family: Dict[str, Dict]) -> None:
    """Add the Holm-adjusted p-value to each comparison of a family, in place."""
    raw = {k: v["p"] for k, v in family.items() if v.get("n_pairs")}
    for name, row in S.holm_bonferroni(raw).items():
        family[name]["p_holm"] = row["p_adjusted"]


def label(planner: str, rule: str) -> str:
    return f"{planner}/{rule}"


# ---------------------------------------------------------------------------
# Single-cycle comparisons
# ---------------------------------------------------------------------------
COMPARE_FIELDS = {
    "objective": lambda r: r["objective"],
    "km": lambda r: r["metrics"]["distance_km"],
    "co2_kg": lambda r: r["metrics"]["co2_kg"],
    "hours": lambda r: r["metrics"]["duration_min"] / 60.0,
    "served": lambda r: r["metrics"]["bins_served"],
    "unserved": lambda r: r["unserved"],
    "late_served": lambda r: r["late_served"],
    "overflow_unserved": lambda r: r["overflow_unserved"],
    "overdue_unserved": lambda r: r["overdue_unserved"],
    "hazard_response_h": lambda r: r["metrics"]["mean_hazard_response_h"],
    "hazard_coverage_pct": lambda r: r["metrics"]["hazard_coverage_pct"],
    "vehicles_used": lambda r: r["metrics"]["vehicles_used"],
    "elapsed_s": lambda r: r["elapsed_s"],
}


def by_unit(records: Iterable[Dict]) -> Dict[Tuple[str, int], Dict[str, Dict]]:
    """Records of a study grouped by (network, snapshot), then by planner/rule."""
    units: Dict[Tuple[str, int], Dict[str, Dict]] = defaultdict(dict)
    for r in records:
        if not r.get("available", True) or "objective" not in r:
            continue
        units[(r["network"], int(r["snapshot"]))][label(r["planner"], r["rule"])] = r
    return units


def compare_study(study: str, reference_from: Optional[str] = None,
                  against: Sequence[str] = ()) -> Optional[Dict]:
    records = ST.read(study)
    if not records:
        return None
    units = by_unit(records)
    if reference_from:
        # The reference solve of these units was made in another study.
        borrowed = by_unit(ST.read(reference_from))
        for unit, row in units.items():
            ref = borrowed.get(unit, {}).get(label(*REFERENCE))
            if ref is not None and label(*REFERENCE) not in row:
                row[label(*REFERENCE)] = ref
    policies = sorted({p for row in units.values() for p in row})
    out: Dict = {"study": study, "units": len(units), "policies": {}}
    for policy in policies:
        rows = [row[policy] for unit, row in sorted(units.items()) if policy in row]
        out["policies"][policy] = {
            name: spread([get(r) for r in rows]) for name, get in COMPARE_FIELDS.items()}
        out["policies"][policy]["violations"] = int(sum(len(r.get("violations", [])) for r in rows))
        out["policies"][policy]["infeasible_on_rescoring"] = int(
            sum(r.get("infeasible_on_rescoring", 0) for r in rows))
        terms = defaultdict(list)
        for r in rows:
            for k, v in r.get("terms", {}).items():
                terms[k].append(v)
        out["policies"][policy]["terms"] = {k: _num(np.mean(v)) for k, v in terms.items()}
        if rows and "flat_emission" in rows[0]:
            out["policies"][policy]["objective_flat_emission"] = spread(
                [r["flat_emission"]["objective"] for r in rows])
        searches = [r.get("search", {}) for r in rows]
        for name in ("construct_s", "improve_s", "descents", "ruin_rounds", "iterations",
                     "generations", "decode_dropped", "heads_reserved", "head_repairs"):
            values = [s[name] for s in searches if s.get(name) is not None]
            if values:
                out["policies"][policy][name] = spread(values)

    # Paired comparisons of the objective against each anchor policy.
    out["against"] = {}
    for anchor in list(against) or [label(*REFERENCE)]:
        family: Dict[str, Dict] = {}
        for policy in policies:
            if policy == anchor:
                continue
            common = [u for u, row in sorted(units.items()) if policy in row and anchor in row]
            if not common:
                continue
            a = [units[u][policy]["objective"] for u in common]
            b = [units[u][anchor]["objective"] for u in common]
            family[policy] = paired(a, b)
        holm(family)
        out["against"][anchor] = family

    # The same comparison by sample, to show the spread between networks.
    samples = sorted({u[0] for u in units})
    if len(samples) > 1:
        out["by_sample"] = {}
        for sample in samples:
            out["by_sample"][sample] = {
                policy: _num(np.mean([row[policy]["objective"]
                                      for u, row in units.items()
                                      if u[0] == sample and policy in row]))
                for policy in policies
                if any(u[0] == sample and policy in row for u, row in units.items())}
    return out


def flat_emission_check(study: str = "main") -> Optional[Dict]:
    """
    How much of a gap between planners is the emission model each plans on.

    OR-Tools plans on arc emissions at mean load.  Scoring every plan under a
    flat emission factor removes that difference from the comparison.
    """
    units = by_unit(ST.read(study))
    rows = [row for _, row in sorted(units.items())
            if "ortools/full" in row and label(*REFERENCE) in row
            and "flat_emission" in row["ortools/full"]]
    if not rows:
        return None
    a = [r["ortools/full"]["flat_emission"]["objective"] for r in rows]
    b = [r[label(*REFERENCE)]["flat_emission"]["objective"] for r in rows]
    return {"ortools_against_insertion_flat_emission": paired(a, b),
            "co2_per_km": rows[0]["ortools/full"]["flat_emission"]["co2_per_km"]}


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------
ROLLOUT_FIELDS = {
    "wait_mean_h": lambda s: s["wait_at_service_h"]["mean"],
    "wait_p95_h": lambda s: s["wait_at_service_h"]["p95"],
    "wait_p99_h": lambda s: s["wait_at_service_h"]["p99"],
    "wait_max_at_service_h": lambda s: s["wait_at_service_h"]["max"],
    "worst_wait_h": lambda s: s["worst_wait_any_dispatch_h"],
    "never_collected": lambda s: s["never_collected"],
    "ever_overdue": lambda s: s["containers_ever_overdue"],
    "max_backlog": lambda s: s["max_backlog"],
    "overflow_pct": lambda s: 100.0 * s["overflow_events_per_container_cycle"],
    "overflow_hours": lambda s: s["overflow_hours_per_container_cycle"],
    "late_share_pct": lambda s: 100.0 * s["late_services_share"],
    "served_per_cycle": lambda s: s["served_per_cycle"],
    "km_per_cycle": lambda s: s["km_per_cycle"],
    # Distance per container collected, the measure sensor-driven studies
    # report against a timetable.
    "km_per_served": lambda s: s["km_per_cycle"] / max(s["served_per_cycle"], 1e-9),
    "co2_per_cycle": lambda s: s["co2_per_cycle"],
    "shift_use_pct": lambda s: 100.0 * s["shift_use"],
    "objective_per_cycle": lambda s: s["objective_per_cycle"],
    "hazard_response_h": lambda s: s["hazard_response_h"],
    "hazard_served_pct": lambda s: s["hazard_served_pct"],
    "elapsed_per_solve_s": lambda s: s["elapsed_per_solve_s"],
}
#: The paper's claim is a bounded wait without giving up urgent service, so the
#: family tested for each policy holds the wait, overflow, the share of
#: hazardous containers collected in the cycle and distance per container.
ROLLOUT_TESTED = ("worst_wait_h", "wait_p95_h", "wait_mean_h", "overflow_pct",
                  "hazard_served_pct", "served_per_cycle", "km_per_cycle",
                  "km_per_served", "objective_per_cycle")


def _percentiles(values: Sequence[float]) -> List[float]:
    x = np.asarray(values, dtype=float)
    return [float(v) for v in np.percentile(x, np.arange(0, 101))] if x.size else []


def rollout_study(load: str) -> Optional[Dict]:
    records = ST.read(f"rollout-{load}")
    if not records:
        return None
    chains: Dict[str, Dict[str, Dict]] = defaultdict(dict)      # policy -> network -> record
    for r in records:
        chains[label(r["planner"], r["rule"])][r["network"]] = r
    networks = sorted({n for per in chains.values() for n in per})
    out: Dict = {"load": load, "networks": networks, "policies": {}, "bound": {},
                 "against": {}, "wait_percentiles": {}}
    for policy, per in sorted(chains.items()):
        summaries = [per[n]["summary"] for n in networks if n in per]
        row = {name: spread([get(s) for s in summaries])
               for name, get in ROLLOUT_FIELDS.items()}
        row["networks"] = len(summaries)
        row["worst_wait_over_networks_h"] = _num(max(s["worst_wait_any_dispatch_h"]
                                                     for s in summaries))
        row["never_collected_total"] = int(sum(s["never_collected"] for s in summaries))
        row["infeasible_chains"] = int(sum(0 if s["feasible"] else 1 for s in summaries))
        row["head_failures"] = int(sum(s["head_failures"] for s in summaries))
        row["head_repairs"] = int(sum(s["head_repairs"] for s in summaries))
        out["policies"][policy] = row

        # The bound of the queueing proposition, where a head is reserved.
        checked = [s for s in summaries if s.get("bound_known_in_advance_h") is not None]
        if checked:
            out["bound"][policy] = {
                "networks": len(checked),
                "min_heads_reserved": int(min(s["min_heads_reserved"] for s in checked)),
                "bound_known_in_advance_h": _num(max(s["bound_known_in_advance_h"]
                                                     for s in checked)),
                "bound_at_observed_backlog_h": spread(
                    [s["bound_at_observed_backlog_h"] for s in checked]),
                "worst_wait_h": _num(max(s["worst_wait_any_dispatch_h"] for s in checked)),
                "max_backlog": int(max(s["max_backlog"] for s in checked)),
                "breaches": int(sum(1 for s in checked if s["bound_breached"])),
                "slack_h": spread([s["bound_at_observed_backlog_h"]
                                   - s["worst_wait_any_dispatch_h"] for s in checked]),
            }

        # The distribution of the wait at the moment of collection, pooled over
        # networks, after the burn-in.
        waits = [w for n in networks if n in per
                 for cycle, _c, w, _h in per[n]["services"] if cycle >= per[n]["burn_in"]]
        out["wait_percentiles"][policy] = _percentiles(waits)
        row["services"] = len(waits)
        row["share_served_overdue_pct"] = _num(
            100.0 * float(np.mean(np.asarray(waits) >= AG.DEFAULT_TAU_H)) if waits else 0.0)

    # Each rule against the full rule on the same planner, paired by network.
    for planner in sorted({p.split("/")[0] for p in chains}):
        anchor = f"{planner}/full"
        if anchor not in chains:
            continue
        for policy in sorted(chains):
            if policy == anchor or not policy.startswith(planner + "/"):
                continue
            common = [n for n in networks if n in chains[policy] and n in chains[anchor]]
            family = {}
            for name in ROLLOUT_TESTED:
                get = ROLLOUT_FIELDS[name]
                family[name] = paired([get(chains[policy][n]["summary"]) for n in common],
                                      [get(chains[anchor][n]["summary"]) for n in common])
            holm(family)
            out["against"][f"{policy} vs {anchor}"] = family
    # Every policy against the reference policy, the insertion planner under the
    # full rule, so that baselines on other planners are tested as well.
    reference = label(*REFERENCE)
    if reference in chains:
        out["against_reference"] = {}
        for policy in sorted(chains):
            if policy == reference:
                continue
            common = [n for n in networks if n in chains[policy] and n in chains[reference]]
            family = {name: paired([ROLLOUT_FIELDS[name](chains[policy][n]["summary"])
                                    for n in common],
                                   [ROLLOUT_FIELDS[name](chains[reference][n]["summary"])
                                    for n in common])
                      for name in ROLLOUT_TESTED}
            holm(family)
            out["against_reference"][policy] = family
    # The two planners under the full rule.
    if "ortools/full" in chains and "insertion/full" in chains:
        common = [n for n in networks if n in chains["ortools/full"]
                  and n in chains["insertion/full"]]
        family = {name: paired(
            [ROLLOUT_FIELDS[name](chains["ortools/full"][n]["summary"]) for n in common],
            [ROLLOUT_FIELDS[name](chains["insertion/full"][n]["summary"]) for n in common])
            for name in ROLLOUT_TESTED}
        holm(family)
        out["against"]["ortools/full vs insertion/full"] = family
    out["world"] = {n: next(iter(per[n]["world"] for per in chains.values() if n in per))
                    for n in networks}
    return out


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
WEATHER_FIELDS = (
    "served_before", "served_in_spell", "served_after", "planned_in_spell",
    "dropped_per_cycle_in_spell", "overtime_to_finish_min_in_spell",
    "overflow_events_before", "overflow_events_in_spell", "overflow_events_after",
    "hazards_before", "hazards_in_spell", "max_wait_before_h", "max_wait_in_spell_h",
    "max_wait_after_h", "max_backlog_in_spell", "km_in_spell", "hours_in_spell",
)
WEATHER_TESTED = ("served_in_spell", "dropped_per_cycle_in_spell",
                  "overflow_events_in_spell", "max_wait_in_spell_h")


def weather_study(scenario: str) -> Optional[Dict]:
    records = ST.read(f"weather-{scenario}")
    if not records:
        return None
    chains: Dict[Tuple[str, str], Dict[str, Dict]] = defaultdict(dict)
    for r in records:
        chains[(r["mode"], label(r["planner"], r["rule"]))][r["network"]] = r
    networks = sorted({n for per in chains.values() for n in per})
    out: Dict = {"scenario": scenario, "networks": networks, "rows": {}, "against": {}}
    for (mode, policy), per in sorted(chains.items()):
        spells = [per[n]["spell"] for n in networks if n in per]
        summaries = [per[n]["summary"] for n in networks if n in per]
        row = {name: spread([s[name] for s in spells]) for name in WEATHER_FIELDS}
        row["networks"] = len(spells)
        row["dropped_heads_total"] = int(sum(s["dropped_heads_in_spell"] for s in spells))
        row["dropped_hazards_total"] = int(sum(s["dropped_hazards_in_spell"] for s in spells))
        recoveries = [s["cycles_to_recover"] for s in spells]
        row["recovered_networks"] = int(sum(1 for c in recoveries if c is not None))
        row["cycles_to_recover"] = spread([c for c in recoveries if c is not None])
        row["worst_wait_h"] = _num(max(s["worst_wait_any_dispatch_h"] for s in summaries))
        row["never_collected_total"] = int(sum(s["never_collected"] for s in summaries))
        row["head_failures"] = int(sum(s["head_failures"] for s in summaries))
        row["bound_breaches"] = int(sum(1 for s in summaries if s.get("bound_breached")))
        row["bound_checked"] = int(sum(1 for s in summaries
                                       if s.get("bound_breached") is not None))
        row["overflow_pct_whole_run"] = spread(
            [100.0 * s["overflow_events_per_container_cycle"] for s in summaries])
        out["rows"][f"{mode}|{policy}"] = row
    # Planning on what the feed reports against planning blind, by network.
    modes = sorted({m for m, _ in chains})
    for better, worse in (("current", "blind"), ("protected", "blind"),
                          ("forecast", "current")):
        if better not in modes or worse not in modes:
            continue
        for policy in sorted({p for _, p in chains}):
            a, b = chains.get((better, policy)), chains.get((worse, policy))
            if not a or not b:
                continue
            common = [n for n in networks if n in a and n in b]
            family = {name: paired([a[n]["spell"][name] for n in common],
                                   [b[n]["spell"][name] for n in common])
                      for name in WEATHER_TESTED}
            holm(family)
            out["against"][f"{better} vs {worse}|{policy}"] = family
    return out


# ---------------------------------------------------------------------------
# Distance model, scale, tuning
# ---------------------------------------------------------------------------
def distance_study() -> Optional[Dict]:
    records = ST.read("distance")
    if not records:
        return None
    street = by_unit(ST.read("main"))
    out: Dict = {"planners": {}}
    for planner in sorted({r["planner"] for r in records}):
        rows = sorted((r for r in records if r["planner"] == planner),
                      key=lambda r: r["snapshot"])
        reported = np.array([r["reported_km"] for r in rows])
        driven = np.array([r["driven_km"] for r in rows])
        routes = sum(r["routes"] for r in rows)
        block = {
            "snapshots": len(rows),
            "reported_km": spread(reported), "driven_km": spread(driven),
            "understatement_pct": _num(100.0 * (driven.sum() - reported.sum()) / driven.sum(), 3),
            "understatement": paired(reported, driven),
            "routes": int(routes),
            "routes_over_shift": int(sum(r["routes_over_shift"] for r in rows)),
            "routes_over_shift_pct": _num(100.0 * sum(r["routes_over_shift"] for r in rows)
                                          / max(1, routes), 2),
            "overrun_min_worst": _num(max(r["overrun_min_worst"] for r in rows), 1),
            "overrun_min_mean_over_route": _num(
                sum(r["overrun_min_total"] for r in rows)
                / max(1, sum(r["routes_over_shift"] for r in rows)), 1),
            "late_arrivals": int(sum(r["late_arrivals"] for r in rows)),
        }
        # What choosing the sequence on the wrong distances costs: the same
        # planner on the street graph, on the same snapshots.
        key = label(planner, "full")
        pairs = [(r["driven_km"], street[("S0", r["snapshot"])][key]["metrics"]["distance_km"],
                  r["served"], street[("S0", r["snapshot"])][key]["metrics"]["bins_served"])
                 for r in rows if ("S0", r["snapshot"]) in street
                 and key in street[("S0", r["snapshot"])]]
        if pairs:
            block["against_street_plan_km"] = paired([p[0] for p in pairs], [p[1] for p in pairs])
            block["served_constant_factor"] = _num(np.mean([p[2] for p in pairs]))
            block["served_street"] = _num(np.mean([p[3] for p in pairs]))
        out["planners"][planner] = block
    return out


def scale_study() -> Optional[Dict]:
    records = ST.read("scale")
    if not records:
        return None
    sizes = sorted({r["n_containers"] for r in records})
    out: Dict = {"sizes": {}, "fit": None}
    points = []
    for n in sizes:
        row = {}
        for planner in ("insertion", "ortools"):
            rows = [r for r in records if r["n_containers"] == n and r["planner"] == planner]
            if not rows:
                continue
            row[planner] = {
                "snapshots": len(rows),
                "objective": spread([r["objective"] for r in rows]),
                "served_pct": spread([100.0 * r["metrics"]["bins_served"] / n for r in rows]),
                "km": spread([r["metrics"]["distance_km"] for r in rows]),
                "elapsed_s": spread([r["elapsed_s"] for r in rows]),
            }
            if planner == "insertion":
                row[planner]["construct_s"] = spread(
                    [r["search"]["construct_s"] for r in rows])
                row[planner]["vehicles"] = rows[0]["n_vehicles"]
                points.extend((n, r["search"]["construct_s"]) for r in rows
                              if r["search"]["construct_s"] > 0)
        out["sizes"][str(n)] = row
    if len({p[0] for p in points}) >= 3:
        x = np.log([p[0] for p in points])
        y = np.log([p[1] for p in points])
        slope, intercept = np.polyfit(x, y, 1)
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(10_000):
            i = rng.integers(0, len(x), len(x))
            if len(set(x[i])) < 2:
                continue
            boot.append(np.polyfit(x[i], y[i], 1)[0])
        residual = y - (slope * x + intercept)
        out["fit"] = {
            "exponent": _num(slope, 3),
            "exponent_ci": [_num(np.quantile(boot, 0.025), 3), _num(np.quantile(boot, 0.975), 3)],
            "points": len(points), "sizes": len({p[0] for p in points}),
            "r_squared": _num(1.0 - residual.var() / y.var(), 4),
        }
    return out


def lambda_study() -> Optional[Dict]:
    records = ST.read("tune-lambda")
    if not records:
        return None
    out = {"snapshots": len({r["snapshot"] for r in records}), "values": {}}
    for value in sorted({r["lambda_prize"] for r in records}):
        rows = [r for r in records if r["lambda_prize"] == value]
        out["values"][f"{value:g}"] = {
            "served": spread([r["served"] for r in rows]),
            "km": spread([r["km"] for r in rows]),
            "missed": spread([r["missed"] for r in rows]),
            "km_per_served": _num(np.mean([r["km"] / max(1, r["served"]) for r in rows])),
        }
    return out


def tuning() -> Optional[Dict]:
    path = RESULTS / "tuning.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    out = {"tuning_set": data["tuning_set"], "snapshots": data["snapshots"],
           "matched_budget_s": data["matched_budget_s"],
           "insertion_objective": data["insertion_objective"],
           "selected": data["selected"], "grid": {}}
    for solver, rows in data["grid"].items():
        out["grid"][solver] = [{"config": r["config"],
                                "mean_objective": _num(r["mean_objective"], 2),
                                "objectives": r["objectives"]} for r in rows]
        out["grid"][solver + "_range_pct"] = _num(
            100.0 * (rows[-1]["mean_objective"] - rows[0]["mean_objective"])
            / rows[0]["mean_objective"], 2)
    return out


def variants(prefix: str) -> Dict:
    """Studies named ``prefix-<tag>``: each planner against the reference, per tag."""
    out = {}
    names = sorted({p.name.split(".s")[0] for p in ST.RAW.glob(f"{prefix}-*.s*.jsonl")})
    for name in names:
        reference_from = "main" if prefix == "budget" else None
        study = compare_study(name, reference_from=reference_from)
        if study:
            out[name.split("-", 1)[1]] = study
    return out


def load_json(name: str) -> Optional[Dict]:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def environment() -> Dict:
    """The software and hardware recorded with the runs, and when they were made."""
    out = {}
    for path in sorted(ST.RAW.glob("*.meta.json")):
        meta = json.loads(path.read_text())
        env = meta.get("environment", {})
        out[path.name.replace(".meta.json", "")] = {
            "utc": env.get("utc"), "versions": env.get("versions"),
            "processor": env.get("processor"), "logical_cpus": env.get("logical_cpus"),
            "platform": env.get("platform"),
        }
    return out


def main() -> int:
    summary: Dict = {
        "protocol": {
            "pairing": "planners on the same snapshots, policies on the same networks",
            "test": "Wilcoxon signed-rank, exact sign permutation below six pairs",
            "family_correction": "Holm step-down within each family of comparisons",
            "difference_interval": "BCa bootstrap, 10,000 resamples, 95 percent",
            "relative_interval": "percentile bootstrap of the ratio of means, "
                                 "10,000 resamples, 95 percent",
            "effect_size": "Cliff's delta",
        },
        "main": compare_study("main", against=[label(*REFERENCE), "ortools/full"]),
        "main_flat_emission": flat_emission_check("main"),
        "wyndham": compare_study("wyndham", against=[label(*REFERENCE), "ortools/full"]),
        "ablation": compare_study("ablation", reference_from="main"),
        "rollout": {load: rollout_study(load) for load in ("moderate", "tight", "hazard")},
        "weather": {name: weather_study(name) for name in ("rain", "storm", "heat", "onset")},
        "weather_scenarios": load_json("weather_scenarios.json"),
        "weather_demand": load_json("weather_demand.json"),
        "weather_live": load_json("weather_live.json"),
        "distance": distance_study(),
        "scale": scale_study(),
        "weights": variants("weights"),
        "depot": variants("depot"),
        "budget": variants("budget"),
        "tuning": tuning(),
        "lambda": lambda_study(),
        "wait_bound": load_json("wait_bound.json"),
        "network": load_json("network.json"),
        "environment": environment(),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=ST._plain))

    def units(block, key="units"):
        return block.get(key, "-") if block else "not run"

    print("main:", units(summary["main"]), "units | wyndham:", units(summary["wyndham"]),
          "| ablation:", units(summary["ablation"]))
    for load, block in summary["rollout"].items():
        print(f"rollout {load}:", len(block["networks"]) if block else "not run",
              "networks,", len(block["policies"]) if block else 0, "policies")
    for name, block in summary["weather"].items():
        print(f"weather {name}:", len(block["rows"]) if block else "not run", "rows")
    print("distance:", "ok" if summary["distance"] else "not run",
          "| scale:", "ok" if summary["scale"] else "not run",
          "| weights:", len(summary["weights"]), "| depot:", len(summary["depot"]),
          "| budget:", len(summary["budget"]))
    print(f"wrote {RESULTS / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
