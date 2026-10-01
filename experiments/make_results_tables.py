"""
Tables and quoted numbers of the evaluation, from ``results/summary.json``.

Kept apart from `make_tables.py` only for length.  `build` receives the helpers
of that module, so the two write into one set of macros and one folder.  A study
that has not run yet leaves its table with a single row saying so, and its
macros as ``n/a``.
"""
from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Sequence

PLANNERS = {
    "insertion": "Insertion", "insertion_cheapest": "Insertion, cheapest",
    "insertion_construct": "Insertion, construction only", "ortools": "OR-Tools",
    "aco": "Ant colony", "genetic": "Genetic algorithm", "risk_graph": "Risk graph",
    "static_sweep": "Static sweep", "threshold": "Fill threshold",
}
RULES = {
    "urgency": "urgency", "ageing": "ageing", "tier_bounded": "tier, bounded",
    "tier_unbounded": "tier, unbounded", "reserve_bounded": "reservation, bounded",
    "full": "full, $r = 1$", "full_r2": "full, $r = 2$", "full_r4": "full, $r = 4$",
    "full_r8": "full, $r = 8$", "full_ageing": "full with ageing",
}
#: Planners whose rule is nominal, because they ignore prices and reservation.
RULELESS = ("static_sweep", "threshold")
REFERENCE = "insertion/full"


def policy_name(policy: str, with_rule: bool = True) -> str:
    planner, rule = policy.split("/")
    name = PLANNERS.get(planner, planner)
    if not with_rule or planner in RULELESS:
        return name
    return f"{name}, {RULES.get(rule, rule)}"


def tag_of(policy: str) -> str:
    """A letters-only macro fragment for a policy, such as ``InsFull``."""
    short = {"insertion": "Ins", "insertion_cheapest": "InsCheap",
             "insertion_construct": "InsCons", "ortools": "Ort", "aco": "Aco",
             "genetic": "Gen", "risk_graph": "Risk", "static_sweep": "Sweep",
             "threshold": "Thr"}
    rule = {"urgency": "Urg", "ageing": "Age", "tier_bounded": "TierB",
            "tier_unbounded": "TierU", "reserve_bounded": "ResB", "full": "Full",
            "full_r2": "Fullr", "full_r4": "Fullrr", "full_r8": "Fullrrr",
            "full_ageing": "FullAge"}
    planner, r = policy.split("/")
    return short.get(planner, "X") + rule.get(r, "X")


class Builder:
    def __init__(self, out: pathlib.Path, summary: Optional[Dict], macro, table, num,
                 whole, pvalue, dig, NA):
        self.out, self.s = out, summary or {}
        self.macro, self.table, self.num, self.whole = macro, table, num, whole
        self.pvalue, self.dig, self.NA = pvalue, dig, NA

    # -- formatting ------------------------------------------------------
    def mean_sd(self, block: Optional[Dict], digits: int = 1) -> str:
        if not block or not block.get("n"):
            return self.NA
        return f"{self.num(block['mean'], digits)} $\\pm$ {self.num(block['sd'], digits)}"

    def mean(self, block: Optional[Dict], digits: int = 1) -> str:
        if not block or not block.get("n"):
            return self.NA
        return self.num(block["mean"], digits)

    def p_cell(self, p) -> str:
        if p is None:
            return self.NA
        return "$<0.001$" if p < 0.001 else f"{p:.3f}"

    def rel_cell(self, comparison: Optional[Dict]) -> str:
        if not comparison or not comparison.get("n_pairs"):
            return self.NA
        ci = comparison.get("relative_ci") or [None, None]
        return (f"{self.num(comparison['relative_pct'], 1, signed=True)} "
                f"[{self.num(ci[0], 1, signed=True)}, {self.num(ci[1], 1, signed=True)}]")

    def empty(self, columns: int, what: str) -> List[List[str]]:
        return [[f"\\multicolumn{{{columns}}}{{l}}{{\\emph{{{what} has not run yet.}}}}"]]

    # -- single-cycle comparisons -----------------------------------------
    def single_cycle(self, key: str, label: str, caption: str, order: Sequence[str],
                     prefix: str, with_rule: bool = False) -> None:
        block = self.dig(self.s, key)
        columns = 10
        header = ["Planner & Objective & vs.\\ insertion (\\%) & $\\delta$ & $p$ & km & "
                  "Served & Overflow & Hazard (h) & Time (s) \\\\"]
        if not block:
            self.table(self.out / f"tab_{label}.tex", caption, f"tab:{label}",
                       "l" + "r" * (columns - 1), header, self.empty(columns, "This study"),
                       tabcolsep="3pt")
            return
        policies = block["policies"]
        against = self.dig(block, "against", REFERENCE, default={})
        rows = []
        for policy in [p for p in order if p in policies] + sorted(
                p for p in policies if p not in order):
            row = policies[policy]
            cmp = against.get(policy)
            overflow = [self.dig(row, "late_served", "mean"),
                        self.dig(row, "overflow_unserved", "mean")]
            rows.append([
                policy_name(policy, with_rule=with_rule),
                self.mean_sd(row.get("objective")),
                "reference" if policy == REFERENCE else self.rel_cell(cmp),
                "" if policy == REFERENCE or not cmp else self.num(cmp.get("cliffs_delta"), 2),
                "" if policy == REFERENCE or not cmp else self.p_cell(cmp.get("p_holm", cmp.get("p"))),
                self.mean(row.get("km")),
                self.mean(row.get("served")),
                self.NA if None in overflow else self.num(sum(overflow), 1),
                self.mean(row.get("hazard_response_h"), 2),
                self.mean(row.get("elapsed_s"), 0),
            ])
            t = tag_of(policy)
            self.macro(f"{prefix}{t}Obj", self.mean(row.get("objective")))
            self.macro(f"{prefix}{t}Km", self.mean(row.get("km")))
            self.macro(f"{prefix}{t}Served", self.mean(row.get("served")))
            if cmp and policy != REFERENCE:
                self.macro(f"{prefix}{t}Rel", self.num(cmp.get("relative_pct"), 1))
                ci = cmp.get("relative_ci") or [None, None]
                self.macro(f"{prefix}{t}RelLo", self.num(ci[0], 1))
                self.macro(f"{prefix}{t}RelHi", self.num(ci[1], 1))
                self.macro(f"{prefix}{t}Delta", self.num(cmp.get("cliffs_delta"), 2))
                self.macro(f"{prefix}{t}P", self.pvalue(cmp.get("p_holm", cmp.get("p"))))
                self.macro(f"{prefix}{t}Lower", self.whole(cmp.get("a_lower_in")))
        self.macro(f"{prefix}Units", self.whole(block.get("units")))
        self.table(self.out / f"tab_{label}.tex", caption, f"tab:{label}",
                   "l" + "r" * (columns - 1), header, rows,
                   notes=("Ten snapshots of the primary sample. " if with_rule else
                          "Matched computing time; every planner except the risk graph "
                          "under the full rule. ") +
                         "Means $\\pm$ standard deviation over snapshots. The relative "
                         "difference is the ratio of means with a 95 percent percentile "
                         "bootstrap interval; $\\delta$ is Cliff's delta and $p$ the "
                         "Holm-adjusted Wilcoxon signed-rank $p$-value against the "
                         "insertion planner. Overflow counts containers reached after "
                         "their predicted overflow plus unserved containers predicted to "
                         "overflow within 24~h.",
                   tabcolsep="3pt")

    # -- multi-cycle ------------------------------------------------------
    def rollout(self, load: str, order: Sequence[str]) -> None:
        block = self.dig(self.s, "rollout", load)
        label = f"rollout_{load}"
        columns = 9
        header = ["Policy & Worst & \\multicolumn{2}{c}{Wait at collection (h)} & Never & "
                  "Overflow & Hazards & Served & km per \\\\",
                  "\\cmidrule(lr){3-4}",
                  " & wait (h) & mean & p95 & collected & (\\%) & in cycle (\\%) & per cycle & "
                  "container \\\\"]
        caption = f"Multi-cycle results, {load} load."
        notes = ("Worst wait is the largest over all networks; other columns are means "
                 "over networks.")
        if not block:
            self.table(self.out / f"tab_{label}.tex", caption, f"tab:{label}",
                       "l" + "r" * (columns - 1), header, self.empty(columns, "This run"),
                       tabcolsep="3pt", notes=notes)
            return
        policies = block["policies"]
        rows = []
        for policy in [p for p in order if p in policies] + sorted(
                p for p in policies if p not in order):
            row = policies[policy]
            rows.append([
                policy_name(policy),
                self.num(row.get("worst_wait_over_networks_h"), 0),
                self.mean(row.get("wait_mean_h")), self.mean(row.get("wait_p95_h"), 0),
                self.whole(row.get("never_collected_total")),
                self.mean(row.get("overflow_pct"), 2),
                self.mean(row.get("hazard_served_pct"), 0),
                self.mean(row.get("served_per_cycle")),
                self.mean(row.get("km_per_served"), 2),
            ])
            t = tag_of(policy)
            pre = {"moderate": "rm", "tight": "rt", "hazard": "rh"}[load]
            self.macro(f"{pre}{t}Worst", self.num(row.get("worst_wait_over_networks_h"), 0))
            self.macro(f"{pre}{t}Pninetyfive", self.mean(row.get("wait_p95_h"), 0))
            self.macro(f"{pre}{t}Mean", self.mean(row.get("wait_mean_h")))
            self.macro(f"{pre}{t}Never", self.whole(row.get("never_collected_total")))
            self.macro(f"{pre}{t}Overflow", self.mean(row.get("overflow_pct"), 2))
            self.macro(f"{pre}{t}Served", self.mean(row.get("served_per_cycle")))
            self.macro(f"{pre}{t}Km", self.mean(row.get("km_per_cycle")))
            self.macro(f"{pre}{t}KmPer", self.mean(row.get("km_per_served"), 2))
            self.macro(f"{pre}{t}Hazard", self.mean(row.get("hazard_served_pct"), 0))
            self.macro(f"{pre}{t}Objective", self.mean(row.get("objective_per_cycle")))
            cmp = self.dig(block, "against_reference", policy, default={})
            for metric, name in (("worst_wait_h", "Worst"), ("wait_p95_h", "Pninetyfive"),
                                 ("overflow_pct", "Overflow"), ("km_per_cycle", "Km"),
                                 ("km_per_served", "KmPer"), ("hazard_served_pct", "Hazard"),
                                 ("served_per_cycle", "Served"),
                                 ("objective_per_cycle", "Objective")):
                c = cmp.get(metric)
                if c and c.get("n_pairs"):
                    self.macro(f"{pre}{t}{name}P", self.pvalue(c.get("p_holm", c.get("p"))))
                    self.macro(f"{pre}{t}{name}Rel", self.num(c.get("relative_pct"), 1))
        pre = {"moderate": "rm", "tight": "rt", "hazard": "rh"}[load]
        self.macro(f"{pre}Networks", self.whole(len(block.get("networks", []))))
        self.table(self.out / f"tab_{label}.tex", caption, f"tab:{label}",
                   "l" + "r" * (columns - 1), header, rows, tabcolsep="3pt", notes=notes)

    def bound(self) -> None:
        """Proposition 1 against every reserving policy and load."""
        columns = 8
        header = ["Load & Policy & $r$ reserved & Bound known & Largest & Bound at & "
                  "Worst & Breaches \\\\",
                  " & & (smallest) & in advance (h) & backlog $M$ & backlog (h) & wait (h) & \\\\"]
        rows, breaches, checked = [], 0, 0
        for load in ("moderate", "tight", "hazard"):
            block = self.dig(self.s, "rollout", load)
            if not block:
                continue
            for policy, b in sorted(block.get("bound", {}).items()):
                rows.append([load, policy_name(policy),
                             self.whole(b.get("min_heads_reserved")),
                             self.num(b.get("bound_known_in_advance_h"), 0),
                             self.whole(b.get("max_backlog")),
                             self.num(self.dig(b, "bound_at_observed_backlog_h", "max"), 0),
                             self.num(b.get("worst_wait_h"), 0),
                             f"{b.get('breaches', 0)} of {b.get('networks', 0)}"])
                breaches += int(b.get("breaches", 0))
                checked += int(b.get("networks", 0))
        if not rows:
            rows = self.empty(columns, "The multi-cycle run")
        self.macro("boundBreaches", self.whole(breaches) if checked else self.NA)
        self.macro("boundChains", self.whole(checked) if checked else self.NA)
        self.table(self.out / "tab_bound.tex",
                   "Waiting-time bound in the multi-cycle runs.",
                   "tab:bound", "ll" + "r" * (columns - 2), header, rows, tabcolsep="3pt",
                   notes="Every policy that reserves queue heads; values are the largest "
                         "over networks.")

    # -- weather ----------------------------------------------------------
    def weather(self) -> None:
        columns = 8
        header = ["Scenario & Mode & Policy & Served & Dropped & Heads & Worst wait & "
                  "Overflow \\\\",
                  " & & & per cycle & per cycle & dropped & in spell (h) & per cycle \\\\"]
        rows = []
        for scenario in ("rain", "storm", "heat", "onset"):
            block = self.dig(self.s, "weather", scenario)
            if not block:
                continue
            for key, row in block.get("rows", {}).items():
                mode, policy = key.split("|")
                rows.append([scenario, mode, policy_name(policy),
                             self.mean(row.get("served_in_spell")),
                             self.mean(row.get("dropped_per_cycle_in_spell")),
                             self.whole(row.get("dropped_heads_total")),
                             self.num(self.dig(row, "max_wait_in_spell_h", "max"), 0),
                             self.mean(row.get("overflow_events_in_spell"), 2)])
                t = tag_of(policy)
                m = {"blind": "Blind", "protected": "Prot", "current": "Curr",
                     "forecast": "Fore"}.get(mode, "X")
                s = {"rain": "Rain", "storm": "Storm", "heat": "Heat", "onset": "Onset"}[scenario]
                self.macro(f"wx{s}{m}{t}Served", self.mean(row.get("served_in_spell")))
                self.macro(f"wx{s}{m}{t}Dropped", self.mean(row.get("dropped_per_cycle_in_spell")))
                self.macro(f"wx{s}{m}{t}Heads", self.whole(row.get("dropped_heads_total")))
                self.macro(f"wx{s}{m}{t}Worst",
                           self.num(self.dig(row, "max_wait_in_spell_h", "max"), 0))
                self.macro(f"wx{s}{m}{t}Breaches", self.whole(row.get("bound_breaches")))
        if not rows:
            rows = self.empty(columns, "The weather study")
        self.table(self.out / "tab_weather_results.tex",
                   "Adverse weather spells at the moderate load.",
                   "tab:weatherresults", "lll" + "r" * (columns - 3), header, rows,
                   tabcolsep="3pt",
                   notes="Six-day spells. Means over networks within the spell; heads "
                         "dropped and worst wait over all networks.")

    # -- distance model ---------------------------------------------------
    def distance(self) -> None:
        block = self.dig(self.s, "distance")
        columns = 7
        header = ["Planner & Reported & Driven & Understated & Routes over & Worst & "
                  "Driven vs.\\ street \\\\",
                  " & km & km & (\\%) & shift & overrun (min) & plan (\\%) \\\\"]
        rows = []
        for planner in ("insertion", "ortools", "aco"):
            b = self.dig(block, "planners", planner)
            if not b:
                continue
            street = b.get("against_street_plan_km")
            rows.append([PLANNERS[planner], self.mean(b.get("reported_km")),
                         self.mean(b.get("driven_km")),
                         self.num(b.get("understatement_pct"), 1),
                         f"{b.get('routes_over_shift', 0)} of {b.get('routes', 0)}",
                         self.num(b.get("overrun_min_worst"), 0),
                         self.rel_cell(street) if street else self.NA])
            t = {"insertion": "Ins", "ortools": "Ort", "aco": "Aco"}[planner]
            self.macro(f"dist{t}Under", self.num(b.get("understatement_pct"), 1))
            self.macro(f"dist{t}Over", self.whole(b.get("routes_over_shift")))
            self.macro(f"dist{t}Routes", self.whole(b.get("routes")))
            if street:
                self.macro(f"dist{t}VsStreet", self.num(street.get("relative_pct"), 1))
                self.macro(f"dist{t}VsStreetP", self.pvalue(street.get("p")))
        if not rows:
            rows = self.empty(columns, "The distance study")
        self.table(self.out / "tab_distance.tex",
                   "Plans on a constant detour factor driven on the street graph.",
                   "tab:distance", "l" + "r" * (columns - 1), header, rows, tabcolsep="3pt",
                   notes="Factor 1.30; primary sample; means over snapshots.")


MAIN_ORDER = ["insertion/full", "ortools/full", "aco/full", "genetic/full",
              "risk_graph/tier_unbounded", "static_sweep/full", "threshold/full"]
ABLATION_ORDER = ["insertion/urgency", "insertion/ageing", "insertion/tier_bounded",
                  "insertion/tier_unbounded", "insertion/reserve_bounded",
                  "insertion/full", "insertion/full_ageing", "insertion_cheapest/full",
                  "insertion_construct/full"]
ROLLOUT_ORDER = ["insertion/urgency", "insertion/ageing", "insertion/tier_bounded",
                 "insertion/tier_unbounded", "insertion/reserve_bounded", "insertion/full",
                 "insertion/full_r2", "insertion/full_r4", "insertion/full_r8",
                 "insertion/full_ageing", "ortools/urgency", "ortools/tier_unbounded",
                 "ortools/full", "ortools/full_r4", "aco/tier_unbounded",
                 "genetic/tier_unbounded", "risk_graph/tier_unbounded",
                 "static_sweep/full", "threshold/full"]


def build(out: pathlib.Path, summary: Optional[Dict], macro, table, num, whole,
          pvalue, dig, NA) -> None:
    """Write the evaluation tables and their macros."""
    b = Builder(out, summary, macro, table, num, whole, pvalue, dig, NA)
    b.single_cycle("main", "main", "Single-cycle comparison on the Dhaka samples.",
                   MAIN_ORDER, "mn")
    b.single_cycle("wyndham", "wyndham", "Single-cycle comparison on the Wyndham days.",
                   MAIN_ORDER, "wy")
    b.single_cycle("ablation", "ablation", "Rule variants under the insertion planner.",
                   ABLATION_ORDER, "ab", with_rule=True)
    for load in ("moderate", "tight", "hazard"):
        b.rollout(load, ROLLOUT_ORDER)
    b.bound()
    b.weather()
    b.distance()
