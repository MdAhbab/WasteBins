"""
Priority algebra with trust-weighted dynamic renormalisation.
=============================================================

Each bin's urgency is a weighted sum of normalised sensor features.  The
interesting part is what happens when a feature is not trustworthy.

Three policies are implemented so the ablation can compare them directly on
identical inputs:

``zero_fill``
    The naive baseline: substitute 0 for the missing channel and keep the
    original weights.  A failed fill sensor therefore *lowers* a bin's priority
    exactly when the operator has least information about it -- the pathology
    the manuscript set out to fix.

``renormalise``
    Drop unavailable channels and rescale the surviving weights to sum to one.
    The score stays on a comparable scale, so a bin with two working sensors is
    still ranked on the evidence that exists.

``trust_weighted``  (default)
    The continuous generalisation.  Each channel's weight is scaled by the trust
    produced in :mod:`wastebins_core.health`, then renormalised.  A drifting or
    poisoned channel is faded out smoothly instead of being either fully
    believed or fully discarded, which matters because the realistic failure
    modes are partial.

All three reduce to the same score when every channel is healthy, so the policy
choice cannot flatter the proposed method on clean data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

POLICY_ZERO_FILL = "zero_fill"
POLICY_RENORMALISE = "renormalise"
POLICY_TRUST_WEIGHTED = "trust_weighted"
POLICIES = (POLICY_ZERO_FILL, POLICY_RENORMALISE, POLICY_TRUST_WEIGHTED)

# Default configuration -- mirrors Django's settings.DYNAMIC_FEATURES so the
# service and the experiments score bins identically.
DEFAULT_FEATURES: Dict[str, Dict] = {
    "distance_m": {"weight": 0.25, "min_val": 0.0, "max_val": 2000.0, "impact": "negative"},
    "waste_level": {"weight": 0.35, "min_val": 0.0, "max_val": 1.0, "impact": "positive"},
    "gas_level": {"weight": 0.25, "min_val": 0.0, "max_val": 1.0, "impact": "positive"},
    "temperature": {"weight": 0.10, "min_val": 10.0, "max_val": 40.0,
                    "optimal": 25.0, "impact": "deviation"},
    "humidity": {"weight": 0.05, "min_val": 50.0, "max_val": 100.0, "impact": "positive"},
}

# Minimum share of the original weight mass that must remain trustworthy before
# the score is considered meaningful at all.
MIN_EFFECTIVE_WEIGHT = 0.15


@dataclass
class PriorityResult:
    score: float
    policy: str
    used_channels: Dict[str, float] = field(default_factory=dict)   # channel -> effective weight
    dropped_channels: Dict[str, str] = field(default_factory=dict)  # channel -> reason
    effective_weight_mass: float = 0.0
    confidence: float = 1.0

    def as_dict(self) -> Dict:
        return {
            "score": round(self.score, 5),
            "policy": self.policy,
            "used_channels": {k: round(v, 4) for k, v in self.used_channels.items()},
            "dropped_channels": dict(self.dropped_channels),
            "effective_weight_mass": round(self.effective_weight_mass, 4),
            "confidence": round(self.confidence, 4),
        }


def normalise_feature(value: float, spec: Mapping) -> float:
    """Map a raw feature onto [0, 1] following its declared impact direction."""
    lo = float(spec.get("min_val", 0.0))
    hi = float(spec.get("max_val", 1.0))
    impact = spec.get("impact", "positive")

    if impact == "deviation":
        optimal = float(spec.get("optimal", (lo + hi) / 2.0))
        max_dev = max(abs(hi - optimal), abs(lo - optimal))
        norm = abs(float(value) - optimal) / max_dev if max_dev > 0 else 0.0
    else:
        norm = (float(value) - lo) / (hi - lo) if hi > lo else 0.0

    norm = max(0.0, min(1.0, norm))
    if impact == "negative":
        norm = 1.0 - norm
    return norm


def compute_priority(values: Mapping[str, Optional[float]],
                     trust: Optional[Mapping[str, float]] = None,
                     features: Optional[Mapping[str, Mapping]] = None,
                     policy: str = POLICY_TRUST_WEIGHTED) -> PriorityResult:
    """
    Score one bin.

    ``values`` maps channel -> measurement, using ``None`` (or a missing key) for
    an unavailable channel.  ``trust`` maps channel -> weight in [0, 1]; it is
    only consulted by the ``trust_weighted`` policy.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")

    features = features or DEFAULT_FEATURES
    trust = trust or {}

    numerator = 0.0
    weight_mass = 0.0
    declared_mass = 0.0
    used: Dict[str, float] = {}
    dropped: Dict[str, str] = {}

    for channel, spec in features.items():
        weight = float(spec.get("weight", 0.0))
        if weight <= 0.0:
            continue
        declared_mass += weight

        raw = values.get(channel)
        available = raw is not None
        try:
            available = available and float(raw) == float(raw)   # rejects NaN
        except (TypeError, ValueError):
            available = False

        if not available:
            if policy == POLICY_ZERO_FILL:
                # The pathology: the channel contributes a hard zero and still
                # consumes its full weight.
                numerator += 0.0 * weight
                weight_mass += weight
                used[channel] = weight
            else:
                dropped[channel] = "unavailable"
            continue

        channel_trust = float(trust.get(channel, 1.0))
        if policy == POLICY_TRUST_WEIGHTED:
            effective = weight * max(0.0, min(1.0, channel_trust))
            if effective <= 1e-9:
                dropped[channel] = "zero_trust"
                continue
        else:
            effective = weight

        numerator += normalise_feature(float(raw), spec) * effective
        weight_mass += effective
        used[channel] = effective

    if weight_mass <= 1e-9:
        return PriorityResult(score=0.0, policy=policy, used_channels={},
                              dropped_channels=dropped or {"*": "no_trusted_channels"},
                              effective_weight_mass=0.0, confidence=0.0)

    score = numerator / weight_mass
    confidence = weight_mass / declared_mass if declared_mass > 0 else 0.0

    return PriorityResult(
        score=max(0.0, min(1.0, score)),
        policy=policy,
        used_channels=used,
        dropped_channels=dropped,
        effective_weight_mass=weight_mass,
        confidence=max(0.0, min(1.0, confidence)),
    )


def is_actionable(result: PriorityResult) -> bool:
    """
    Whether a score rests on enough trustworthy evidence to dispatch on.

    Below the threshold the correct operational response is to send a technician
    to the node rather than to act on the priority, and the API reports it as
    such instead of silently emitting a confident-looking number.
    """
    return result.effective_weight_mass >= MIN_EFFECTIVE_WEIGHT


def blend_with_model(rule_score: float, model_risk: Optional[float],
                     confidence: float = 1.0, model_weight: float = 0.65) -> float:
    """
    Combine the interpretable rule score with the learned forward risk.

    The learned component is trusted in proportion to how much sensor evidence
    survived validation: when the inputs degrade, the system falls back towards
    the transparent rule rather than towards a model extrapolating on rubbish.
    """
    if model_risk is None:
        return float(rule_score)
    w = max(0.0, min(1.0, model_weight)) * max(0.0, min(1.0, confidence))
    return float((1.0 - w) * rule_score + w * float(model_risk))
