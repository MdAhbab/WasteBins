"""
Priority calculation.

This module is now a thin adapter over :mod:`wastebins_core.priority` and
:mod:`bins.services.dispatch`.  The scoring algebra, the trust-weighted
renormalisation and the model blending all live in the core package so that the
service and the published experiments cannot drift apart.

The class and module-level names are retained because the legacy JSON API and
the existing tests import them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from django.conf import settings

from wastebins_core.geo import haversine
from wastebins_core.priority import (DEFAULT_FEATURES, POLICY_RENORMALISE,
                                     POLICY_TRUST_WEIGHTED, POLICY_ZERO_FILL,
                                     blend_with_model, compute_priority,
                                     is_actionable, normalise_feature)

from bins.models import Node, SensorReading

# Kept for backward compatibility with older imports.
DEFAULT_DYNAMIC_FEATURES = {
    **{k: {"type": "priority", **v} for k, v in DEFAULT_FEATURES.items()},
    "traffic_density": {"type": "cost_multiplier", "weight": 0.10, "min_val": 0.0,
                        "max_val": 1.0, "impact": "negative"},
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (delegates to the core implementation)."""
    return haversine(lat1, lon1, lat2, lon2)


class PriorityCalculator:
    """
    Multi-criteria bin urgency with dynamic renormalisation.

    ``policy`` selects how unavailable channels are handled:
    ``zero_fill`` (the naive baseline), ``renormalise`` (drop and rescale), or
    ``trust_weighted`` (scale each weight by the sensor-health trust, then
    rescale) which is the default and the method reported in the manuscript.
    """

    def __init__(self, policy: str = POLICY_TRUST_WEIGHTED, **legacy_weights):
        self.policy = policy
        self.features = getattr(settings, "DYNAMIC_FEATURES", None) or DEFAULT_FEATURES
        # Older call sites passed individual weights; honour them if given.
        overrides = {
            "distance_weight": "distance_m",
            "waste_weight": "waste_level",
            "gas_weight": "gas_level",
            "temperature_weight": "temperature",
            "humidity_weight": "humidity",
        }
        if legacy_weights:
            self.features = {k: dict(v) for k, v in self.features.items()}
            for kwarg, channel in overrides.items():
                if kwarg in legacy_weights and channel in self.features:
                    self.features[channel]["weight"] = float(legacy_weights[kwarg])
            if "max_distance_m" in legacy_weights and "distance_m" in self.features:
                self.features["distance_m"]["max_val"] = float(legacy_weights["max_distance_m"])

    # -- single-bin scoring ------------------------------------------------
    def calculate_single_priority(self, **kwargs) -> float:
        """Score one bin from keyword measurements; absent keys are renormalised away."""
        trust = kwargs.pop("_trust", None)
        result = compute_priority(kwargs, trust=trust, features=self.features,
                                  policy=self.policy)
        return result.score

    def explain_single_priority(self, **kwargs) -> Dict:
        """Score plus the full reasoning trail (which channels counted, and why)."""
        trust = kwargs.pop("_trust", None)
        return compute_priority(kwargs, trust=trust, features=self.features,
                                policy=self.policy).as_dict()

    # -- fleet scoring -----------------------------------------------------
    def calculate_node_priorities(self, nodes: List[Node], user_lat: float,
                                  user_lng: float, use_ai_model: bool = True
                                  ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Priorities and traffic scores for a set of nodes.

        Returns ``(priorities, traffic_scores)`` keyed by node id, matching the
        signature the existing views rely on.
        """
        from bins.services import dispatch

        scores = dispatch.score_bins(nodes, user_lat=user_lat, user_lng=user_lng,
                                     policy=self.policy, use_model=use_ai_model)
        priorities = {nid: float(entry["priority"]) for nid, entry in scores.items()}

        latest = SensorReading.objects.filter(node_id__in=[n.id for n in nodes]) \
            .order_by("node_id", "-timestamp")
        traffic: Dict[int, float] = {}
        for reading in latest:
            traffic.setdefault(reading.node_id, float(reading.traffic_density or 0.0))

        for node in nodes:
            priorities.setdefault(node.id, 0.0)
            traffic.setdefault(node.id, 0.0)
        return priorities, traffic

    def select_top_priority_nodes(self, nodes: List[Node], user_lat: float,
                                  user_lng: float, max_nodes: int = 5
                                  ) -> List[Tuple[Node, float]]:
        priorities, _ = self.calculate_node_priorities(nodes, user_lat, user_lng)
        ranked = [(node, priorities.get(node.id, 0.0)) for node in nodes]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[:max_nodes]


# Module-level instance used by the views.
priority_calculator = PriorityCalculator()
