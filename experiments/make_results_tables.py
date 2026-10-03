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
        # The rule variants differ in the overdue containers they leave, and
        # hardly in overflow or hazard response, so their table shows the former.
        columns = 9 if with_rule else 10
        header = [("Variant & Objective & vs.\\ full (\\%) & $\\delta$ & $p$ & km & "
                   "Served & Overdue left & Time (s) \\\\") if with_rule else
                  ("Planner & Objective & vs.\\ insertion (\\%) & $\\delta$ & $p$ & km & "
                   "Served & Overflow & Hazard (h) & Time (s) \\\\")]
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
                ABLATION_NAMES.get(policy, policy_name(policy)) if with_rule
                else policy_name(policy, with_rule=False),
                self.mean_sd(row.get("objective")),
                "reference" if policy == REFERENCE else self.rel_cell(cmp),
                "" if policy == REFERENCE or not cmp else self.num(cmp.get("cliffs_delta"), 2),
                "" if policy == REFERENCE or not cmp else self.p_cell(cmp.get("p_holm", cmp.get("p"))),
                self.mean(row.get("km")),
                self.mean(row.get("served")),
                *([self.mean(row.get("overdue_unserved"), 1)] if with_rule else
                  [self.NA if None in overflow else self.num(sum(overflow), 1),
                   self.mean(row.get("hazard_response_h"), 2)]),
                self.mean(row.get("elapsed_s"), 0),
            ])
            t = tag_of(policy)
            self.macro(f"{prefix}{t}Obj", self.mean(row.get("objective")))
            self.macro(f"{prefix}{t}Km", self.mean(row.get("km")))
            self.macro(f"{prefix}{t}Served", self.mean(row.get("served")))
            self.macro(f"{prefix}{t}CoPerKm", self.mean(row.get("co2_per_km"), 2))
            self.macro(f"{prefix}{t}OverdueLeft", self.mean(row.get("overdue_unserved"), 1))
            if cmp and policy != REFERENCE:
                self.macro(f"{prefix}{t}Rel", self.num(cmp.get("relative_pct"), 1))
                ci = cmp.get("relative_ci") or [None, None]
                self.macro(f"{prefix}{t}RelLo", self.num(ci[0], 1))
                self.macro(f"{prefix}{t}RelHi", self.num(ci[1], 1))
                self.macro(f"{prefix}{t}Delta", self.num(cmp.get("cliffs_delta"), 2))
                self.macro(f"{prefix}{t}P", self.pvalue(cmp.get("p_holm", cmp.get("p"))))
                self.macro(f"{prefix}{t}Lower", self.whole(cmp.get("a_lower_in")))
        self.macro(f"{prefix}Units", self.whole(block.get("units")))
        # The gap between OR-Tools and insertion within each sample.
        samples = block.get("by_sample") or {}
        gaps = [100.0 * (v["ortools/full"] - v[REFERENCE]) / v[REFERENCE]
                for v in samples.values() if "ortools/full" in v and REFERENCE in v]
        if gaps:
            self.macro(f"{prefix}OrtSampleMin", self.num(min(gaps), 1))
            self.macro(f"{prefix}OrtSampleMax", self.num(max(gaps), 1))
            self.macro(f"{prefix}Samples", self.whole(len(gaps)))
        self.table(self.out / f"tab_{label}.tex", caption, f"tab:{label}",
                   "l" + "r" * (columns - 1), header, rows,
                   notes=("Ten snapshots of the primary sample. " if with_rule else
                          "Matched computing time; every planner except the risk graph "
                          "under the full rule. ") +
                         "Means $\\pm$ standard deviation over snapshots. The relative "
                         "difference is the ratio of means with a 95 percent percentile "
                         "bootstrap interval; $\\delta$ is Cliff's delta and $p$ the "
                         "Holm-adjusted Wilcoxon signed-rank $p$-value against the " +
                         ("full rule" if with_rule else "insertion planner") +
                         (". Overdue left is the number of overdue containers a plan "
                          "leaves unserved." if with_rule else
                          ". Overflow counts containers reached after "
                          "their predicted overflow plus unserved containers predicted to "
                          "overflow within 24~h."),
                   tabcolsep="2pt" if with_rule else "3pt", fontsize="footnotesize")

    # -- multi-cycle ------------------------------------------------------
    #: Tested rollout metrics and the macro fragment of each.
    ROLLOUT_METRICS = (("worst_wait_h", "Worst"), ("wait_p95_h", "Pninetyfive"),
                       ("overflow_pct", "Overflow"), ("km_per_cycle", "Km"),
                       ("km_per_served", "KmPer"), ("hazard_served_pct", "Hazard"),
                       ("served_per_cycle", "Served"), ("objective_per_cycle", "Objective"))

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
            self.macro(f"{pre}{t}WorstMean", self.mean(row.get("worst_wait_h"), 0))
            self.macro(f"{pre}{t}PastTwo", self.num(row.get("share_served_past_two_deadlines_pct"), 2))
            self.macro(f"{pre}{t}ShiftUse", self.mean(row.get("shift_use_pct"), 0))
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
            for metric, name in self.ROLLOUT_METRICS:
                c = cmp.get(metric)
                if c and c.get("n_pairs"):
                    self.macro(f"{pre}{t}{name}P", self.pvalue(c.get("p_holm", c.get("p"))))
                    self.macro(f"{pre}{t}{name}Rel", self.num(c.get("relative_pct"), 1))
        pre = {"moderate": "rm", "tight": "rt", "hazard": "rh"}[load]
        self.macro(f"{pre}Networks", self.whole(len(block.get("networks", []))))
        # The full rule set against the timetable, in the direction the
        # literature reports it: the change from the timetable.
        full, sweep = policies.get(REFERENCE), policies.get("static_sweep/full")
        if full and sweep:
            for metric, short in (("km_per_served", "KmPer"), ("overflow_pct", "Overflow")):
                a, b = self.dig(full, metric, "mean"), self.dig(sweep, metric, "mean")
                if a is not None and b:
                    self.macro(f"{pre}FullVsSweep{short}", self.num(100.0 * (a - b) / b, 1))
            a, b = self.dig(full, "overflow_pct", "mean"), self.dig(sweep, "overflow_pct", "mean")
            if a:
                self.macro(f"{pre}SweepOverflowRatio", self.num(b / a, 1))
        # Each rule against the full rule on the same planner, as rmW<policy><metric>.
        for name, family in block.get("against", {}).items():
            policy, anchor = name.split(" vs ")
            if anchor.split("/")[0] != policy.split("/")[0]:
                continue
            t = tag_of(policy)
            for metric, short in self.ROLLOUT_METRICS:
                c = family.get(metric)
                if c and c.get("n_pairs"):
                    self.macro(f"{pre}W{t}{short}P", self.pvalue(c.get("p_holm", c.get("p"))))
                    self.macro(f"{pre}W{t}{short}Rel", self.num(c.get("relative_pct"), 1))
        self.table(self.out / f"tab_{label}.tex", caption, f"tab:{label}",
                   "l" + "r" * (columns - 1), header, rows, tabcolsep="3pt", notes=notes,
                   fontsize="footnotesize")

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
                stem = "bd" + {"moderate": "Mod", "tight": "Tight", "hazard": "Haz"}[load] \
                    + tag_of(policy)
                self.macro(f"{stem}Known", self.num(b.get("bound_known_in_advance_h"), 0))
                self.macro(f"{stem}Backlog", self.whole(b.get("max_backlog")))
                self.macro(f"{stem}AtBacklog",
                           self.num(self.dig(b, "bound_at_observed_backlog_h", "max"), 0))
                self.macro(f"{stem}Slack", self.mean(b.get("slack_h"), 0))
                self.macro(f"{stem}Reserved", self.whole(b.get("min_heads_reserved")))
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
        modes = {"blind": "Blind", "protected": "Prot", "current": "Curr", "forecast": "Fore"}
        order = ["insertion/urgency", "insertion/full", "ortools/full"]
        for scenario in ("rain", "storm", "heat", "onset"):
            block = self.dig(self.s, "weather", scenario)
            if not block:
                continue
            s = {"rain": "Rain", "storm": "Storm", "heat": "Heat", "onset": "Onset"}[scenario]
            keys = sorted(block.get("rows", {}),
                          key=lambda k: (list(modes).index(k.split("|")[0]),
                                         order.index(k.split("|")[1])
                                         if k.split("|")[1] in order else 9))
            if rows:
                rows.append("\\addlinespace")
            for index, key in enumerate(keys):
                row = block["rows"][key]
                mode, policy = key.split("|")
                self.macro(f"wx{s}{modes.get(mode, 'X')}{tag_of(policy)}WaitMean",
                           self.mean(row.get("max_wait_in_spell_h"), 0))
                rows.append([scenario.capitalize() if index == 0 else "",
                             mode.capitalize(), policy_name(policy),
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
            # One mode against another on the same networks, as
            # wx<scenario><better><worse><policy><metric>.
            for name, family in block.get("against", {}).items():
                pair, policy = name.split("|")
                better, worse = pair.split(" vs ")
                stem = f"wx{s}{modes[better]}{modes[worse]}{tag_of(policy)}"
                for metric, short in (("served_in_spell", "Served"),
                                      ("dropped_per_cycle_in_spell", "Dropped"),
                                      ("overflow_events_in_spell", "Overflow"),
                                      ("max_wait_in_spell_h", "Wait")):
                    c = family.get(metric)
                    if c and c.get("n_pairs"):
                        self.macro(f"{stem}{short}P", self.pvalue(c.get("p_holm", c.get("p"))))
                        self.macro(f"{stem}{short}Rel", self.num(c.get("relative_pct"), 1))
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
            self.macro(f"dist{t}Overrun", self.num(b.get("overrun_min_worst"), 0))
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


    # -- supporting single-cycle studies ---------------------------------
    #: Letters-only macro fragments for the variant names, which hold digits.
    LETTERS = {"co2x0": "CoZero", "co2x3": "CoThree", "timex0.5": "TimeHalf",
               "timex2": "TimeTwo", "overflowx0.5": "OverHalf", "overflowx2": "OverTwo",
               "lambdax0.5": "LamHalf", "lambdax2": "LamTwo", "4": "Four", "9": "Nine",
               "18": "Eighteen", "werribee": "Werribee", "point_cook": "PointCook",
               "0.1": "Tenth", "0.25": "Quarter", "1": "One", "2": "Two"}

    def support(self) -> None:
        """Objective weights, depots and budgets: each planner against insertion."""
        columns = 8
        header = ["Study & Variant & Units & OR-Tools (\\%) & $p$ & Ant colony (\\%) & $p$ & "
                  "Genetic (\\%) \\\\"]
        weights = {"co2x0": "CO$_2$ $\\times 0$", "co2x3": "CO$_2$ $\\times 3$",
                   "timex0.5": "Time $\\times 0.5$", "timex2": "Time $\\times 2$",
                   "overflowx0.5": "Overflow $\\times 0.5$",
                   "overflowx2": "Overflow $\\times 2$",
                   "lambdax0.5": "$\\lambda \\times 0.5$", "lambdax2": "$\\lambda \\times 2$"}
        depots = {"4": "Dhaka, rank 4", "9": "Dhaka, rank 9", "18": "Dhaka, rank 18",
                  "werribee": "Werribee", "point_cook": "Point Cook"}
        budgets = {"0.1": "$\\times 0.1$", "0.25": "$\\times 0.25$", "1": "$\\times 1$",
                   "2": "$\\times 2$"}
        rows: List = []
        ranges: Dict[str, List[float]] = {}
        for study, names, short in (("weights", weights, "Weights"),
                                    ("depot", depots, "Depot"),
                                    ("budget", budgets, "Budget")):
            block = self.dig(self.s, study, default={}) or {}
            present = [k for k in names if k in block]
            if not present:
                continue
            if rows:
                rows.append("\\addlinespace")
            for index, key in enumerate(present):
                b = block[key]
                against = self.dig(b, "against", REFERENCE, default={})
                cells = [{"weights": "Weights", "depot": "Depot",
                          "budget": "Budget"}[study] if index == 0 else "",
                         names[key], self.whole(b.get("units"))]
                for planner in ("ortools/full", "aco/full", "genetic/full"):
                    c = against.get(planner)
                    if c and c.get("n_pairs"):
                        cells.append(self.num(c.get("relative_pct"), 1, signed=True))
                        if planner != "genetic/full":
                            cells.append(self.p_cell(c.get("p_holm", c.get("p"))))
                        ranges.setdefault(f"{short}{tag_of(planner)}", []).append(
                            float(c["relative_pct"]))
                        tag = self.LETTERS[key]
                        self.macro(f"sup{short}{tag}{tag_of(planner)}Rel",
                                   self.num(c.get("relative_pct"), 1))
                        self.macro(f"sup{short}{tag}{tag_of(planner)}P",
                                   self.pvalue(c.get("p_holm", c.get("p"))))
                        self.macro(f"sup{short}{tag}{tag_of(planner)}Lower",
                                   self.whole(c.get("a_lower_in")))
                        self.macro(f"sup{short}{tag}Units", self.whole(b.get("units")))
                    else:
                        cells += ["--"] if planner == "genetic/full" else ["--", ""]
                rows.append(cells)
        for name, values in ranges.items():
            self.macro(f"sup{name}Min", self.num(min(values), 1))
            self.macro(f"sup{name}Max", self.num(max(values), 1))
        if not rows:
            rows = self.empty(columns, "The supporting studies")
        self.table(self.out / "tab_support.tex",
                   "Planners under other weights, depots and time budgets.",
                   "tab:support", "ll" + "r" * (columns - 2), header, rows, tabcolsep="3pt",
                   notes="Relative difference of the mean objective against the insertion "
                         "planner on the same snapshots; $p$ is the Holm-adjusted Wilcoxon "
                         "signed-rank $p$-value. Weights and budgets are multiples of the "
                         "main values and of the insertion time. Dhaka variants use the "
                         "primary sample; Werribee and Point Cook are Wyndham depots.")

    def scale(self) -> None:
        block = self.dig(self.s, "scale")
        columns = 8
        header = ["Containers & Vehicles & \\multicolumn{3}{c}{Insertion} & "
                  "\\multicolumn{2}{c}{OR-Tools} & OR-Tools vs. \\\\",
                  "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
                  " & & construction (s) & total (s) & served (\\%) & total (s) & "
                  "served (\\%) & insertion (\\%) \\\\"]
        rows, ends = [], []
        for n, row in sorted((self.dig(block, "sizes", default={}) or {}).items(),
                             key=lambda kv: int(kv[0])):
            ins, ort = row.get("insertion", {}), row.get("ortools", {})
            rel = None
            if ins and ort:
                rel = 100.0 * (ort["objective"]["mean"] - ins["objective"]["mean"]) \
                    / ins["objective"]["mean"]
            rows.append([n, self.whole(ins.get("vehicles")),
                         self.mean(ins.get("construct_s"), 1),
                         self.mean(ins.get("elapsed_s"), 1),
                         self.mean(ins.get("served_pct"), 1),
                         self.mean(ort.get("elapsed_s"), 1),
                         self.mean(ort.get("served_pct"), 1),
                         self.num(rel, 1, signed=True)])
            ends.append((n, self.mean(ins.get("construct_s"), 1), self.num(rel, 1)))
        for name, end in (("Small", ends[:1]), ("Large", ends[-1:])):
            for n, construct, rel in end:
                self.macro(f"scale{name}N", n)
                self.macro(f"scale{name}Construct", construct)
                self.macro(f"scale{name}OrtRel", rel)
        fit = self.dig(block, "fit", default={}) or {}
        self.macro("scaleExponent", self.num(fit.get("exponent"), 2))
        self.macro("scaleExponentLo", self.num(self.dig(fit, "exponent_ci", 0), 2))
        self.macro("scaleExponentHi", self.num(self.dig(fit, "exponent_ci", 1), 2))
        self.macro("scaleRsq", self.num(fit.get("r_squared"), 3))
        if rows:
            self.macro("scaleSizes", self.whole(len(rows)))
        else:
            rows = self.empty(columns, "The scale study")
        self.table(self.out / "tab_scale.tex", "Computing time by instance size.",
                   "tab:scale", "r" * columns, header, rows, tabcolsep="3pt",
                   notes="Three snapshots per size; OR-Tools receives the insertion time. "
                         "Means over snapshots.")

    def rollout_networks(self, load: str, order: Sequence[str]) -> None:
        """The longest wait of every policy on every network, for the supplement."""
        block = self.dig(self.s, "rollout", load)
        if not block:
            return
        networks = sorted(block["networks"], key=lambda n: int(n.lstrip("R")))
        header = ["Policy & " + " & ".join(networks) + " \\\\"]
        rows = []
        for policy in [p for p in order if p in block["policies"]]:
            per = block["policies"][policy].get("by_network", {})
            rows.append([policy_name(policy)] + [
                self.num(self.dig(per, n, "worst_wait_h"), 0) for n in networks])
        self.table(self.out / f"tab_networks_{load}.tex",
                   f"Longest wait (h) on each network, {load} load.",
                   f"tab:networks_{load}", "l" + "r" * len(networks), header, rows,
                   tabcolsep="1.8pt", fontsize="footnotesize")


MAIN_ORDER = ["insertion/full", "ortools/full", "aco/full", "genetic/full",
              "risk_graph/tier_unbounded", "static_sweep/full", "threshold/full"]
#: Row names of the ablation table, where every row uses the insertion planner.
ABLATION_NAMES = {
    "insertion/urgency": "Urgency", "insertion/ageing": "Ageing",
    "insertion/tier_bounded": "Tier, bounded", "insertion/tier_unbounded": "Tier, unbounded",
    "insertion/reserve_bounded": "Reservation, bounded", "insertion/full": "Full, $r = 1$",
    "insertion/full_ageing": "Full with ageing",
    "insertion_cheapest/full": "Full, cheapest insertion",
    "insertion_construct/full": "Full, construction only",
}
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
        b.rollout_networks(load, ROLLOUT_ORDER)
    b.bound()
    b.weather()
    b.distance()
    b.support()
    b.scale()
    # OR-Tools against insertion with every plan scored at a flat emission
    # factor, which removes the emission model from the comparison.
    flat = dig(summary or {}, "main_flat_emission", "ortools_against_insertion_flat_emission")
    macro("mnFlatRel", num(dig(flat, "relative_pct"), 1))
    macro("mnFlatRelLo", num(dig(flat, "relative_ci", 0), 1))
    macro("mnFlatRelHi", num(dig(flat, "relative_ci", 1), 1))
    macro("mnFlatP", pvalue(dig(flat, "p")))
    macro("mnFlatCoPerKm", num(dig(summary or {}, "main_flat_emission", "co2_per_km"), 2))
    # Search effort of the budgeted planners at matched time, main comparison.
    pol = dig(summary or {}, "main", "policies", default={}) or {}
    macro("mnAcoIterations", num(dig(pol, "aco/full", "iterations", "mean"), 0))
    macro("mnGenGenerations", num(dig(pol, "genetic/full", "generations", "mean"), 0))
    macro("mnInsConstruct", num(dig(pol, "insertion/full", "construct_s", "mean"), 0))
    macro("mnInsImprove", num(dig(pol, "insertion/full", "improve_s", "mean"), 0))
    macro("mnInsDescents", num(dig(pol, "insertion/full", "descents", "mean"), 0))
    macro("mnInsRuin", num(dig(pol, "insertion/full", "ruin_rounds", "mean"), 0))
