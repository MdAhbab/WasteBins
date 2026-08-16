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

Worst-case wait bound
---------------------
Assume a dispatch cycle of length :math:`\\Delta` hours in which the planner
serves the highest-scoring feasible bins, and that any bin served in the
previous cycle has :math:`w \\le \\Delta`.  A never-served bin *i* (worst case
:math:`P_i = 0`) outranks the strongest competitor (worst case :math:`P_j = 1`,
just served) as soon as

.. math::

    \\gamma\\, a(w_i) > (1-\\gamma) + \\gamma\\, a(\\Delta),

which gives the finite guarantee

.. math::

    w_i^{\\max} = \\tau\\left[\\frac{1-\\gamma}{\\gamma} +
                  \\left(\\frac{\\Delta}{\\tau}\\right)^{\\kappa}\\right]^{1/\\kappa}
    \\qquad\\text{provided}\\qquad
    \\frac{1-\\gamma}{\\gamma} + (\\Delta/\\tau)^{\\kappa} < 1 .

The proviso requires :math:`\\gamma > 1/2` in the limit of a short cycle; below
that the aging term improves equity empirically but carries no hard guarantee,
and :func:`worst_case_wait_bound` reports ``inf`` rather than pretending
otherwise.

Zero-cost hazard response
-------------------------
Requiring a large :math:`\\gamma` would ordinarily delay hazardous bins.  That is
avoided with a **lexicographic tier**: bins whose hazard probability exceeds
``hazard_threshold`` are placed in tier 0 and always outrank tier 1, so the
aging weight is free to be tuned for equity *within* the non-hazard tier without
touching hazard response.  :func:`effective_priorities` returns the tier
alongside the score, and the routing layer sorts by ``(tier, -score)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

DEFAULT_GAMMA = 0.55
DEFAULT_TAU_H = 48.0
DEFAULT_KAPPA = 2.0
DEFAULT_HAZARD_THRESHOLD = 0.60


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


def worst_case_wait_bound(gamma: float = DEFAULT_GAMMA,
                          tau_h: float = DEFAULT_TAU_H,
                          kappa: float = DEFAULT_KAPPA,
                          cycle_h: float = 12.0) -> float:
    """
    Guaranteed upper bound on the wait of the least urgent bin, in hours.

    Returns ``inf`` when the parameters admit no hard guarantee, which is the
    honest answer for small ``gamma`` rather than a number that does not hold.
    """
    g = max(0.0, min(1.0, float(gamma)))
    if g <= 0.0:
        return math.inf
    k = max(1e-6, float(kappa))
    slack = (1.0 - g) / g + (max(0.0, float(cycle_h)) / float(tau_h)) ** k
    if slack >= 1.0:
        return math.inf
    return float(tau_h * (slack ** (1.0 / k)))


def gamma_for_target_wait(target_wait_h: float, tau_h: float = DEFAULT_TAU_H,
                          kappa: float = DEFAULT_KAPPA,
                          cycle_h: float = 12.0) -> Optional[float]:
    """
    Smallest ``gamma`` whose guaranteed bound meets ``target_wait_h``.

    This is the tuning knob the reviewer asked for: state the service-level
    objective in hours and read off the equity weight that certifies it.

    Because :math:`a(\\cdot)` saturates at :math:`w = \\tau`, no value of
    ``gamma`` can certify a wait at or beyond ``tau``.  Set ``tau_h`` to the
    service-level target itself (or higher) and the function returns a usable
    weight; otherwise it returns ``None`` rather than a value whose bound is
    actually infinite.
    """
    tau = float(tau_h)
    target = float(target_wait_h)
    k = max(1e-6, float(kappa))

    cycle = max(0.0, float(cycle_h))
    if not (0.0 < target < tau):
        return None
    if target <= cycle:
        # No dispatch policy can beat its own cycle length.
        return None
    ratio = (target / tau) ** k - (cycle / tau) ** k
    if ratio <= 0.0:
        return None
    # ratio = (1 - g) / g  =>  g = 1 / (1 + ratio)
    g = 1.0 / (1.0 + ratio)
    if not (0.0 < g < 1.0):
        return None
    # Guard against floating-point drift at the feasibility boundary.
    for candidate in (g, min(0.999, g + 1e-6), min(0.999, g + 1e-4)):
        if worst_case_wait_bound(candidate, tau, k, cycle_h) <= target + 1e-9:
            return float(candidate)
    return None


@dataclass
class TieredPriority:
    node_id: int
    tier: int              # 0 = hazard override, 1 = normal
    score: float           # effective priority after aging
    base_priority: float
    wait_hours: float
    hazard_prob: float = 0.0

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
    """Apply aging to a whole fleet and assign the lexicographic hazard tier."""
    hazard_probs = hazard_probs or {}
    out: Dict[int, TieredPriority] = {}
    for node_id, base in priorities.items():
        wait = float(wait_hours.get(node_id, 0.0))
        hazard = float(hazard_probs.get(node_id, 0.0))
        out[node_id] = TieredPriority(
            node_id=node_id,
            tier=0 if hazard >= hazard_threshold else 1,
            score=effective_priority(base, wait, gamma, tau_h, kappa),
            base_priority=float(base),
            wait_hours=wait,
            hazard_prob=hazard,
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
