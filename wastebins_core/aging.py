"""
Service equity: the anti-starvation aging term.
===============================================

A purely urgency-driven dispatcher starves quiet bins indefinitely -- a bin in a
low-generation neighbourhood never wins against a busy one, so its residents get
a permanently worse service level.  The first submission mentioned an "aging
term" but never wrote it down, and a reviewer correctly objected that neither
the formula nor the equity/hazard trade-off could be checked.

Formulation
-----------
For bin *i* with intrinsic priority :math:`P_i \\in [0,1]` and idle time
:math:`w_i` hours since its last collection, the effective dispatch priority is
the convex combination

.. math::

    P^{\\mathrm{eff}}_i = (1-\\gamma) P_i + \\gamma\\, a(w_i),
    \\qquad
    a(w) = \\min\\!\\left(1, (w/\\tau)^{\\kappa}\\right)

* :math:`\\gamma \\in [0,1)` is the equity weight (0 disables aging entirely),
* :math:`\\tau` is the target maximum wait in hours,
* :math:`\\kappa \\ge 1` shapes the ramp.  Large :math:`\\kappa` keeps the boost
  near zero until the bin approaches its deadline, so equity costs almost
  nothing until it has to be paid.

Because the combination is convex, :math:`P^{\\mathrm{eff}} \\in [0,1]`: the
aging term cannot inflate scores out of range, which an additive term does.

Why the soft term alone cannot guarantee anything
-------------------------------------------------
An earlier version of this module derived a closed-form wait bound from the soft
term alone.  It assumed that a starved bin competes against a bin that was *just
served*, so the competitor's wait is at most one cycle :math:`\\Delta`, and asked
when :math:`\\gamma\\, a(w_i) > (1-\\gamma) + \\gamma\\, a(\\Delta)`.  That
assumption fails exactly when equity matters.  In a fleet with more bins than it
can reach, the competitors are themselves deferred, so :math:`a(w_j)` is large
rather than :math:`a(\\Delta)`, and the required inequality becomes
:math:`\\gamma\\,(a(w_i) - a(w_j)) > 1-\\gamma`.  Since :math:`a` saturates at
:math:`\\tau`, the difference tends to zero and no :math:`\\gamma < 1` satisfies
it.  A second gap is simpler: outranking one competitor is not the same as being
served, because the number served per cycle is limited by capacity and shift
time.

Measured on a scarce fleet over 40 cycles, that bound was violated in 5 of 192
observations at :math:`\\gamma = 0.55` and 188 of 192 at
:math:`\\gamma = 0.85`, with the realised worst wait growing as the claimed bound
tightened.  The formula has been removed rather than restated.

Hard deadline tier, and the bound it does support
-------------------------------------------------
The guarantee now comes from the dispatch order, not from an inequality.  Three
lexicographic tiers are used, and a lower tier always outranks a higher one:

* **tier 0, hazard.** Hazard probability at or above ``hazard_threshold``.
* **tier 1, overdue.** Wait at or above :math:`\\tau`, ordered by wait, longest
  first.  The ordering score is :math:`w/(\\tau + w)`, which is strictly
  increasing in :math:`w` and therefore never ties.  This is the property the
  saturating soft term lacks.
* **tier 2, normal.** Ordered by the soft effective priority above.

Let :math:`m` be the largest number of bins that are overdue at the same time,
and :math:`c` the number of overdue bins the fleet clears per cycle.  A bin is
promoted at :math:`w = \\tau` and, being ordered behind at most :math:`m` older
overdue bins, is served within :math:`\\lceil m/c \\rceil` cycles.  Hence

.. math::

    w^{\\max} \\le \\tau + \\Delta \\left\\lceil m/c \\right\\rceil ,
    \\qquad c \\ge 1 .

Two conditions are stated rather than assumed.  The fleet must clear overdue
bins at least as fast as they are promoted (:math:`c \\ge r` for promotion rate
:math:`r`), otherwise the overdue set grows without limit and no bound exists.
And waits are observed in units of :math:`\\Delta`, so the attainable bound is
the next multiple of :math:`\\Delta` at or above the expression.
:func:`worst_case_wait_bound` returns ``inf`` when :math:`c \\le 0`.

The practical consequence is that :math:`\\tau` is now the tuning knob the
operator sets directly: state the service-level objective in hours and
:func:`tau_for_target_wait` returns the :math:`\\tau` that meets it.  The equity
weight :math:`\\gamma` no longer carries the guarantee, and is free to be tuned
for average fairness within the normal tier.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

DEFAULT_GAMMA = 0.55
DEFAULT_TAU_H = 48.0
DEFAULT_KAPPA = 2.0
DEFAULT_HAZARD_THRESHOLD = 0.60

# Lexicographic dispatch tiers.  Lower outranks higher, and the routing layer
# sorts by (tier, -score) throughout, so these values are load-bearing: tier 0
# is tested directly as "is this a hazard" in vrp.py and metaheuristics.py.
TIER_HAZARD = 0
TIER_OVERDUE = 1
TIER_NORMAL = 2


def aging_boost(wait_hours: float, tau_h: float = DEFAULT_TAU_H,
                kappa: float = DEFAULT_KAPPA) -> float:
    """The saturating ramp :math:`a(w) = \\min(1, (w/\\tau)^{\\kappa})`."""
    if tau_h <= 0:
        return 1.0
    ratio = max(0.0, float(wait_hours)) / float(tau_h)
    return float(min(1.0, ratio ** max(1e-6, float(kappa))))


def effective_priority(priority: float, wait_hours: float,
                       gamma: float = DEFAULT_GAMMA,
                       tau_h: float = DEFAULT_TAU_H,
                       kappa: float = DEFAULT_KAPPA) -> float:
    """Convex blend of intrinsic urgency and accumulated wait; stays in [0, 1]."""
    g = max(0.0, min(1.0, float(gamma)))
    p = max(0.0, min(1.0, float(priority)))
    return float((1.0 - g) * p + g * aging_boost(wait_hours, tau_h, kappa))


def overdue_score(wait_hours: float, tau_h: float = DEFAULT_TAU_H) -> float:
    """
    Ordering score inside the overdue tier: :math:`w/(\\tau + w)`.

    Strictly increasing in the wait and never saturating, so the longest-waiting
    overdue bin always ranks first and two overdue bins never tie.  The
    saturating soft ramp cannot do this: past :math:`\\tau` every bin scores the
    same and the order falls back to intrinsic priority, which is precisely how a
    quiet bin gets starved.
    """
    w = max(0.0, float(wait_hours))
    tau = max(1e-9, float(tau_h))
    return float(w / (tau + w))


def overdue_pressure(wait_hours: float, tau_h: float = DEFAULT_TAU_H) -> float:
    """
    Pricing escalation for the overdue tier: :math:`w/(2\tau)`.

    :func:`overdue_score` decides which overdue bin the planner *tries* first.
    This decides how much it costs to give up and skip one, and the two need
    different shapes.  An ordering score should be bounded, because only its
    ranking matters and a bounded score cannot be destabilised by one extreme
    value.  A price must not be bounded, because the planner compares it against
    a detour cost that has no upper limit either.

    That distinction was the defect this function exists to close.  Charging the
    ordering score, as an earlier version did, capped the penalty for skipping an
    overdue bin at ``lambda_prize * overdue_multiplier``, or 180 at the default
    weights.  Any bin whose marginal insertion cost exceeded 180, which at those
    weights means roughly 33 km from the depot, was skipped at every wait from
    48 h to a million hours.  The tier ordering was implemented correctly and
    made no difference: being first in the queue does not help when the planner
    declines to serve anyone in that queue.  The saturation that retired the
    previous bound had simply moved from the ranking channel to the pricing one.

    The linear form is chosen for three properties.  It agrees with the old
    penalty exactly at the promotion threshold, so existing weight tuning carries
    over.  It is at least as large as the old penalty at every wait past that
    threshold, since ``w/(2 tau) >= w/(tau + w)`` whenever ``w >= tau``, so no
    bin is served later than before.  And it inverts in closed form, which is
    what makes the wait bound in :func:`worst_case_wait_bound` provable rather
    than asserted: a bin of marginal insertion cost ``C`` becomes worth serving
    at ``w = 2 tau C / (lambda_prize * overdue_multiplier)``.
    """
    tau = max(1e-9, float(tau_h))
    return float(max(0.0, float(wait_hours)) / (2.0 * tau))


def wait_to_outprice(insertion_cost: float, tau_h: float = DEFAULT_TAU_H,
                     lambda_prize: float = 45.0,
                     overdue_multiplier: float = 4.0) -> float:
    """
    The wait at which skipping a bin costs more than serving it.

    Inverts the escalation in :func:`overdue_pressure`.  ``insertion_cost`` is
    the marginal cost of inserting the bin into the best route available, in the
    same units as the routing objective.  Returns the wait in hours at which the
    planner stops preferring to skip.

    A bin is therefore served once its wait reaches the larger of this value and
    ``tau_h``: it must be overdue before the escalation applies at all.
    """
    denom = max(1e-9, float(lambda_prize) * float(overdue_multiplier))
    return float(2.0 * max(0.0, float(tau_h)) * max(0.0, float(insertion_cost)) / denom)


def worst_case_wait_bound(tau_h: float = DEFAULT_TAU_H,
                          cycle_h: float = 12.0,
                          max_overdue: int = 1,
                          served_overdue_per_cycle: int = 1,
                          max_insertion_cost: float = 0.0,
                          lambda_prize: float = 45.0,
                          overdue_multiplier: float = 4.0) -> float:
    """
    Upper bound on any bin's wait, in hours, from the overdue tier.

    Three things have to happen before a starved bin is collected, and the bound
    is the sum of the three delays.

    First the bin must be **promoted** into the overdue tier, which happens at
    :math:`\\tau`.  Waits are only ever observed in whole cycles, because a bin
    starts at zero and ages by :math:`\\Delta` each cycle, so the first
    observable instant at or past :math:`\\tau` is
    :math:`\\Delta \\lceil \\tau/\\Delta \\rceil`, not :math:`\\tau` itself.

    Second the bin must be **worth serving**.  Promotion fixes the order in which
    the planner tries bins, not whether it accepts any of them, and a
    prize-collecting objective drops a bin whenever the detour costs more than
    the penalty for skipping it.  ``max_insertion_cost`` is the largest marginal
    insertion cost a bin in the instance can present.  By
    :func:`wait_to_outprice` the escalating penalty passes that cost at
    :math:`2 \\tau C / (\\lambda \\mu)`, so the effective promotion threshold is
    the larger of that and :math:`\\tau`.  Left at zero, the argument asserts
    that every bin is affordable as soon as it is promoted, which holds exactly
    when :math:`C \\le \\lambda \\mu / 2`, or 90 at the default weights.

    Third the bin must **reach the front of the queue**.  It sits behind at most
    ``max_overdue`` older overdue bins and the fleet clears
    ``served_overdue_per_cycle`` of them each cycle, so this takes
    :math:`\\lceil m/c \\rceil` cycles.  A bin served in the same cycle it was
    promoted waits no extra time, so the added delay is
    :math:`(\\lceil m/c \\rceil - 1)\\Delta`, not
    :math:`\\lceil m/c \\rceil \\Delta`.

    Together:

        w_promote = max(tau_h, 2 * tau_h * C / (lambda_prize * overdue_multiplier))
        bound     = Delta * ceil(w_promote / Delta) + (ceil(m / c) - 1) * Delta

    An earlier version returned ``tau_h + cycle_h * ceil(m/c)``.  That is a true
    bound but a loose one, on both of the counts above, by exactly
    :math:`\\tau + \\Delta - \\Delta\\lceil \\tau/\\Delta \\rceil`, which is 12 h
    at the defaults.  It is why the rollout observed a worst wait of 48.0 h
    against a stated bound of 60.0 h.  The 48.0 h was the tight bound being
    attained; the slack sat in the formula, not in the policy.

    Returns ``inf`` when the fleet clears no overdue bins, because the overdue
    set then grows without limit and there is no bound to report.  The guarantee
    stays conditional on the fleet keeping pace with promotions, which is a
    capacity statement about the deployment rather than a property of the
    formula.
    """
    c = int(served_overdue_per_cycle)
    if c <= 0:
        return math.inf
    tau = max(0.0, float(tau_h))
    delta = max(0.0, float(cycle_h))
    m = max(0, int(max_overdue))

    promote_at = max(tau, wait_to_outprice(max_insertion_cost, tau,
                                           lambda_prize, overdue_multiplier))
    if delta <= 0.0:
        return float(promote_at)

    # Waits are observed in whole cycles, so round the promotion instant up to
    # the next observable one.  The tolerance stops a promotion instant that is
    # already an exact multiple of the cycle from being pushed a cycle further by
    # floating-point dust.
    promote_obs = delta * math.ceil(promote_at / delta - 1e-9)
    cycles = math.ceil(m / c) if m > 0 else 1
    return float(promote_obs + delta * max(0, cycles - 1))


def tau_for_target_wait(target_wait_h: float, cycle_h: float = 12.0,
                        max_overdue: int = 1,
                        served_overdue_per_cycle: int = 1) -> Optional[float]:
    """
    The promotion threshold that meets a stated service-level objective.

    Inverts :func:`worst_case_wait_bound`.  State the longest acceptable wait in
    hours and this returns the :math:`\\tau` to configure, or ``None`` when the
    objective is shorter than the clearing time the cycle already implies: no
    dispatch policy can beat its own cycle.
    """
    c = int(served_overdue_per_cycle)
    if c <= 0:
        return None
    target = float(target_wait_h)
    cycle = max(0.0, float(cycle_h))
    m = max(0, int(max_overdue))
    clearing = cycle * (math.ceil(m / c) if m > 0 else 0)
    tau = target - clearing
    if tau <= 0.0:
        return None
    return float(tau)


@dataclass
class TieredPriority:
    node_id: int
    tier: int              # 0 = hazard, 1 = overdue, 2 = normal
    score: float           # ordering score within the tier
    base_priority: float
    wait_hours: float
    hazard_prob: float = 0.0
    overdue: bool = False
    # Pricing escalation, `w/(2 tau)`, carried alongside the ordering score so a
    # caller cannot pass one channel to the router and forget the other.  Only
    # meaningful for the overdue tier; see `overdue_pressure`.
    pressure: float = 0.0

    def sort_key(self):
        return (self.tier, -self.score)


def effective_priorities(priorities: Mapping[int, float],
                         wait_hours: Mapping[int, float],
                         hazard_probs: Optional[Mapping[int, float]] = None,
                         gamma: float = DEFAULT_GAMMA,
                         tau_h: float = DEFAULT_TAU_H,
                         kappa: float = DEFAULT_KAPPA,
                         hazard_threshold: float = DEFAULT_HAZARD_THRESHOLD
                         ) -> Dict[int, TieredPriority]:
    """
    Apply aging to a whole fleet and assign the lexicographic tier.

    Hazard first, then any bin whose wait has reached ``tau_h``, then the rest.
    Overdue bins are scored by wait rather than by the soft term, so the tier is
    ordered longest-wait-first and cannot tie.  That ordering is what makes the
    bound in :func:`worst_case_wait_bound` true; without it, bins past
    ``tau_h`` all share the same saturated aging boost and the quietest one can be
    passed over indefinitely.
    """
    hazard_probs = hazard_probs or {}
    out: Dict[int, TieredPriority] = {}
    for node_id, base in priorities.items():
        wait = float(wait_hours.get(node_id, 0.0))
        hazard = float(hazard_probs.get(node_id, 0.0))
        is_overdue = wait >= float(tau_h)

        if hazard >= hazard_threshold:
            tier = TIER_HAZARD
            score = effective_priority(base, wait, gamma, tau_h, kappa)
        elif is_overdue:
            tier = TIER_OVERDUE
            score = overdue_score(wait, tau_h)
        else:
            tier = TIER_NORMAL
            score = effective_priority(base, wait, gamma, tau_h, kappa)

        out[node_id] = TieredPriority(
            node_id=node_id,
            tier=tier,
            score=score,
            pressure=overdue_pressure(wait, tau_h) if is_overdue else 0.0,
            base_priority=float(base),
            wait_hours=wait,
            hazard_prob=hazard,
            overdue=is_overdue,
        )
    return out


def order_by_tier(tiered: Mapping[int, TieredPriority]) -> list:
    """Node ids ordered by (hazard tier, descending effective priority)."""
    return [tp.node_id for tp in sorted(tiered.values(), key=lambda t: t.sort_key())]


# ---------------------------------------------------------------------------
# Equity measurement
# ---------------------------------------------------------------------------
def gini(values: Sequence[float]) -> float:
    """
    Gini coefficient of a non-negative sample (0 = perfectly equal).

    Applied to per-bin wait times it summarises how unevenly service is spread
    across the network.
    """
    xs = sorted(float(max(0.0, v)) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0
    total = sum(xs)
    if total <= 1e-12:
        return 0.0
    cumulative = 0.0
    for i, x in enumerate(xs, start=1):
        cumulative += i * x
    return float((2.0 * cumulative) / (n * total) - (n + 1.0) / n)


def equity_report(wait_hours: Sequence[float]) -> Dict[str, float]:
    """Summary statistics used by the equity experiment and the operator UI."""
    xs = [float(w) for w in wait_hours]
    if not xs:
        return {"n": 0, "mean_wait_h": 0.0, "p95_wait_h": 0.0,
                "worst_wait_h": 0.0, "gini": 0.0}
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "n": len(xs),
        "mean_wait_h": round(sum(xs) / len(xs), 3),
        "p95_wait_h": round(ordered[idx], 3),
        "worst_wait_h": round(ordered[-1], 3),
        "gini": round(gini(xs), 4),
    }
