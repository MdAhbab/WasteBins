"""
Platform API: fleet dispatch, sensor health, explainability, emissions,
audit ledger and model governance.

Cost discipline
---------------
Read endpoints are cheap and safe to poll.  The only expensive operation --
solving the routing problem -- lives behind an explicit ``POST`` to
``/api/v1/fleet/plan/``, so no dashboard refresh can trigger it.  Endpoints that
mutate state (fault injection, service recording, corrector reset) are ``POST``
only and are written to the audit ledger.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
from django.conf import settings
from django.db.models import Avg, Count, Q
from django.utils import timezone
from rest_framework import status
from rest_framework.authentication import BasicAuthentication, SessionAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from wastebins_core import aging as CORE_AGING
from wastebins_core import emissions as CORE_EMISSIONS
from wastebins_core import faults as CORE_FAULTS
from wastebins_core import features as CORE_FEATURES
from wastebins_core import health as CORE_HEALTH
from wastebins_core import metaheuristics as CORE_META
from wastebins_core import priority as CORE_PRIORITY
from wastebins_core.traffic import bpr_speed_kmh, temporal_multiplier

from .drf_serializers import (AuditEntrySerializer, DepotSerializer,
                              ModelVersionSerializer, RoutePlanSerializer,
                              RoutePlanSummarySerializer, SensorHealthSerializer,
                              ServiceEventSerializer, VehicleSerializer)
from .models import (AuditEntry, Depot, ModelVersion, Node, RoutePlan,
                     SensorHealth, SensorReading, ServiceEvent, Vehicle)
from .services import audit, dispatch, models_registry, telemetry

logger = logging.getLogger(__name__)

_AUTH = [SessionAuthentication, BasicAuthentication]
_PERM = [IsAuthenticated]


def _float(request, key, default=None):
    raw = request.query_params.get(key, request.data.get(key) if hasattr(request, "data") else None)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _int(request, key, default=None):
    value = _float(request, key, None)
    return int(value) if value is not None else default


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------
class FleetConfigAPIView(APIView):
    """Depots, vehicles and the constraints the planner will enforce."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        depot, vehicles = dispatch.ensure_default_fleet()
        return Response({
            "depots": DepotSerializer(Depot.objects.all(), many=True).data,
            "vehicles": VehicleSerializer(Vehicle.objects.all().order_by("id"), many=True).data,
            "defaults": getattr(settings, "FLEET_DEFAULTS", {}),
            "solvers": CORE_META.available_solvers(),
            "traffic": dispatch.traffic_provider().describe(),
            "constraints": [
                "vehicle capacity with on-board compaction",
                "depot return, including mid-shift tipping trips",
                "hard shift duration limit",
                "per-bin service time windows",
                "waste-stream licensing per vehicle",
                "optional service (prize-collecting) with a hazard override",
            ],
        })


class VehicleDetailAPIView(APIView):
    """Update a vehicle's operating constraints."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def put(self, request, pk):
        try:
            vehicle = Vehicle.objects.get(pk=pk)
        except Vehicle.DoesNotExist:
            return Response({"error": "vehicle not found"}, status=status.HTTP_404_NOT_FOUND)
        serializer = VehicleSerializer(vehicle, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        audit.append("config", {"vehicle_id": pk, "changes": request.data},
                     actor=request.user.username)
        return Response(serializer.data)


class FleetPlanAPIView(APIView):
    """
    Generate a dispatch plan (``POST``) or fetch the most recent one (``GET``).

    Planning is the only expensive operation in the API and is deliberately not
    reachable by a ``GET``, so page renders and polling can never trigger it.
    """

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        plan_id = _int(request, "plan_id")
        record = (RoutePlan.objects.filter(pk=plan_id).first() if plan_id
                  else RoutePlan.objects.order_by("-timestamp").first())
        if record is None:
            return Response({"plan": None, "message": "No plan generated yet."})
        return Response({"plan": RoutePlanSerializer(record).data})

    def post(self, request):
        algorithm = str(request.data.get("algorithm", "proposed")).lower()
        available = CORE_META.available_solvers()
        if algorithm not in available:
            return Response(
                {"error": f"unknown solver '{algorithm}'", "available": available},
                status=status.HTTP_400_BAD_REQUEST)
        if not available[algorithm]:
            return Response(
                {"error": f"solver '{algorithm}' is not installed in this deployment"},
                status=status.HTTP_400_BAD_REQUEST)

        budget = _float(request, "time_budget_s",
                        settings.FLEET_DEFAULTS["PLAN_TIME_BUDGET_S"])
        budget = max(0.5, min(30.0, budget))     # bound so a client cannot hang a worker

        result = dispatch.generate_plan(
            user=request.user,
            algorithm=algorithm,
            user_lat=_float(request, "lat"),
            user_lng=_float(request, "lng"),
            time_budget_s=budget,
            use_traffic=str(request.data.get("use_traffic", "true")).lower() != "false",
            gamma=_float(request, "gamma"),
            tau_h=_float(request, "tau_h"),
        )
        if "error" in result:
            return Response(result, status=status.HTTP_400_BAD_REQUEST)
        return Response(result, status=status.HTTP_201_CREATED)


class FleetCompareAPIView(APIView):
    """
    Run several solvers on the identical live instance and compare them.

    Same bins, same fleet, same traffic surface, same objective -- which is what
    makes the comparison meaningful rather than a table of numbers produced
    under different assumptions.
    """

    authentication_classes = _AUTH
    permission_classes = _PERM

    def post(self, request):
        requested = request.data.get("algorithms") or ["proposed", "risk_graph", "genetic", "aco"]
        available = CORE_META.available_solvers()
        budget = max(0.5, min(15.0, _float(
            request, "time_budget_s",
            settings.FLEET_DEFAULTS["COMPARE_TIME_BUDGET_S"])))

        results: Dict[str, Dict] = {}
        for name in requested:
            name = str(name).lower()
            if not available.get(name):
                results[name] = {"error": "unavailable"}
                continue
            try:
                plan = dispatch.generate_plan(
                    user=request.user, algorithm=name, time_budget_s=budget,
                    persist=False,
                )
                if "error" in plan:
                    results[name] = plan
                else:
                    results[name] = {
                        "metrics": plan["metrics"],
                        "objective": plan["objective"],
                        "compute_ms": plan["compute_ms"],
                    }
            except Exception as exc:
                logger.exception("solver %s failed", name)
                results[name] = {"error": f"{type(exc).__name__}: {exc}"}

        ranked = sorted(
            [(k, v["objective"]) for k, v in results.items() if "objective" in v],
            key=lambda kv: kv[1])
        return Response({
            "results": results,
            "ranking": [{"algorithm": k, "objective": round(v, 3)} for k, v in ranked],
            "note": "All solvers were given the same bins, fleet, traffic surface and "
                    "objective weights, and all plans were re-scored with the same "
                    "feasibility simulator.",
        })


class PlanListAPIView(APIView):
    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        limit = max(1, min(100, _int(request, "limit", 20)))
        plans = RoutePlan.objects.order_by("-timestamp")[:limit]
        return Response({"plans": RoutePlanSummarySerializer(plans, many=True).data})


class ServiceEventAPIView(APIView):
    """Record that a bin was emptied, and list recent collections."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        limit = max(1, min(500, _int(request, "limit", 50)))
        events = (ServiceEvent.objects.select_related("node", "vehicle")
                  .order_by("-collected_at")[:limit])
        return Response({"events": ServiceEventSerializer(events, many=True).data,
                         "equity": dispatch.equity_report(days=_int(request, "days", 30))})

    def post(self, request):
        node_id = _int(request, "node_id")
        if node_id is None:
            return Response({"error": "node_id is required"},
                            status=status.HTTP_400_BAD_REQUEST)
        event = dispatch.record_service(
            node_id=node_id,
            vehicle_id=_int(request, "vehicle_id"),
            plan_id=_int(request, "plan_id"),
            fill=_float(request, "fill"),
            actor=request.user.username,
        )
        if event is None:
            return Response({"error": f"node {node_id} not found"},
                            status=status.HTTP_404_NOT_FOUND)
        return Response(ServiceEventSerializer(event).data, status=status.HTTP_201_CREATED)


class EquityAPIView(APIView):
    """Service-equity statistics and the certified worst-case wait bound."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        days = max(1, min(365, _int(request, "days", 30)))
        report = dispatch.equity_report(days=days)

        gamma = float(settings.ROUTING_AGING_GAMMA)
        tau = float(settings.ROUTING_AGING_TAU_H)
        kappa = CORE_AGING.DEFAULT_KAPPA
        cycle = _float(request, "cycle_h", 12.0)

        sweep = []
        for g in (0.0, 0.2, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9):
            bound = CORE_AGING.worst_case_wait_bound(g, tau, kappa, cycle)
            sweep.append({
                "gamma": g,
                "bound_h": None if bound == float("inf") else round(bound, 2),
                "guaranteed": bound != float("inf"),
            })

        target = _float(request, "target_wait_h")
        recommendation = None
        if target:
            suggested = CORE_AGING.gamma_for_target_wait(target, tau, kappa, cycle)
            recommendation = {
                "target_wait_h": target,
                "required_gamma": round(suggested, 4) if suggested is not None else None,
                "feasible": suggested is not None,
                "note": ("The aging ramp saturates at tau, so no weight can certify a wait "
                         "at or beyond tau. Raise tau to at least the target.")
                if suggested is None else None,
            }

        return Response(dispatch.json_safe({
            "history": report,
            "parameters": {"gamma": gamma, "tau_h": tau, "kappa": kappa,
                           "cycle_h": cycle},
            "formula": "P_eff = (1 - gamma) * P + gamma * min(1, (w / tau)^kappa)",
            "bound_formula": "w_max = tau * [ (1-gamma)/gamma + (cycle/tau)^kappa ]^(1/kappa)",
            "gamma_sweep": sweep,
            "recommendation": recommendation,
            "hazard_tier": {
                "threshold": CORE_AGING.DEFAULT_HAZARD_THRESHOLD,
                "note": "Bins above the hazard threshold occupy a strictly higher "
                        "lexicographic tier, so the equity weight can be tuned "
                        "without delaying hazard response.",
            },
        }))


# ---------------------------------------------------------------------------
# Sensor health
# ---------------------------------------------------------------------------
class SensorHealthAPIView(APIView):
    """Per-channel trust across the fleet, refreshed on demand."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        refresh = str(request.query_params.get("refresh", "false")).lower() == "true"
        nodes = list(Node.objects.filter(is_active=True).values_list("id", flat=True))

        if refresh:
            results = telemetry.assess_nodes(nodes, persist=True)
            summary = telemetry.health_summary(results)
        else:
            summary = None

        rows = (SensorHealth.objects.select_related("node")
                .order_by("node_id", "channel"))
        payload = SensorHealthSerializer(rows, many=True).data

        if summary is None:
            trusts = [float(r["trust"]) for r in payload] or [1.0]
            counts: Dict[str, int] = {}
            for r in payload:
                counts[r["status"]] = counts.get(r["status"], 0) + 1
            summary = {
                "channels_assessed": len(payload),
                "status_counts": counts,
                "mean_trust": round(float(np.mean(trusts)), 4),
                "min_trust": round(float(np.min(trusts)), 4),
                "nodes_flagged": len({r["node"] for r in payload if r["status"] != "ok"}),
            }

        return Response({
            "summary": summary,
            "channels": payload,
            "detectors": {
                "range": "hard physical-bound violation",
                "missing": "genuine absence (NaN), kept distinct from a measured zero",
                "stuck": "frozen register or zero-variance channel",
                "spike": "impulsive noise, via a Hampel filter on median/MAD",
                "drift": "slow bias, via a two-sided Page-Hinkley change test",
                "cross_channel": "physically impossible combinations between channels",
                "peer": "fleet inconsistency, the only detector that sees stealthy poisoning",
            },
            "trust_thresholds": {
                "ok": CORE_HEALTH.TRUST_SUSPECT,
                "suspect": CORE_HEALTH.TRUST_DEGRADED,
                "degraded": CORE_HEALTH.TRUST_FAILED,
            },
            "renormalisation_policies": list(CORE_PRIORITY.POLICIES),
            "active_policy": CORE_PRIORITY.POLICY_TRUST_WEIGHTED,
        })


class FaultTaxonomyAPIView(APIView):
    """The fault taxonomy the robustness evaluation covers."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        return Response({
            "modes": CORE_FAULTS.describe_taxonomy(),
            "channel_ranges": CORE_FAULTS.CHANNEL_RANGE,
        })


class FaultInjectionAPIView(APIView):
    """
    Inject a fault into a node's live telemetry, for demonstration and testing.

    Injected readings are flagged ``is_synthetic`` and carry a ``fault_label``,
    so they are always distinguishable from genuine measurements and can be
    removed again.  The action is written to the audit ledger.
    """

    authentication_classes = _AUTH
    permission_classes = _PERM

    def post(self, request):
        node_id = _int(request, "node_id")
        mode = str(request.data.get("mode", "")).strip()
        channel = str(request.data.get("channel", "waste_level"))
        n = max(1, min(200, _int(request, "n_samples", 20)))

        if mode not in CORE_FAULTS.FAULT_MODES:
            return Response({"error": f"unknown mode '{mode}'",
                             "modes": list(CORE_FAULTS.FAULT_MODES)},
                            status=status.HTTP_400_BAD_REQUEST)
        if channel not in telemetry.HEALTH_CHANNELS:
            return Response({"error": f"unknown channel '{channel}'",
                             "channels": list(telemetry.HEALTH_CHANNELS)},
                            status=status.HTTP_400_BAD_REQUEST)
        try:
            node = Node.objects.get(pk=node_id)
        except Node.DoesNotExist:
            return Response({"error": f"node {node_id} not found"},
                            status=status.HTTP_404_NOT_FOUND)

        recent = list(SensorReading.objects.filter(node=node)
                      .order_by("-timestamp")[:n])[::-1]
        if not recent:
            return Response({"error": "node has no readings to perturb"},
                            status=status.HTTP_400_BAD_REQUEST)

        values = np.array([float(getattr(r, channel) or 0.0) for r in recent], dtype=float)
        spec = CORE_FAULTS.FaultSpec(
            mode=mode,
            rate=_float(request, "rate", 0.5),
            magnitude=_float(request, "magnitude", 0.25),
            start_frac=_float(request, "start_frac", 0.0),
            duration_frac=_float(request, "duration_frac", 1.0),
            direction=str(request.data.get("direction", "amplify")),
            stealth=_float(request, "stealth", 0.25),
            seed=_int(request, "seed", 42),
            extra={"channel": channel},
        )
        try:
            corrupted, mask = CORE_FAULTS.inject(values, spec)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        now = timezone.now()
        created = []
        template = recent[-1]
        for i, value in enumerate(corrupted):
            payload = {
                "temperature": template.temperature,
                "humidity": template.humidity,
                "gas_level": template.gas_level,
                "waste_level": template.waste_level,
                "traffic_density": template.traffic_density,
            }
            payload[channel] = None if not np.isfinite(value) else float(value)
            # Every row written by the injector is labelled, including the
            # unaffected carrier samples.  The label -- not is_synthetic -- is
            # what the cleanup endpoint keys on, because the seeded demonstration
            # network is *also* synthetic and must survive a cleanup.
            created.append(SensorReading(
                node=node,
                timestamp=now + timedelta(seconds=i + 1),
                is_synthetic=True,
                fault_label=(mode if bool(mask[i]) else f"{mode}:carrier")[:32],
                **payload,
            ))
        SensorReading.objects.bulk_create(created)
        Node.objects.filter(pk=node.id).update(last_update=created[-1].timestamp)

        assessment = telemetry.assess_nodes([node.id], persist=True).get(node.id, {})
        audit.append("fault", {
            "node_id": node.id, "node": node.name, "mode": mode, "channel": channel,
            "n_samples": len(created), "affected": int(mask.sum()),
            "detected_status": assessment.get(channel).status if channel in assessment else None,
        }, actor=request.user.username)

        return Response({
            "node_id": node.id,
            "mode": mode,
            "channel": channel,
            "samples_written": len(created),
            "samples_affected": int(mask.sum()),
            "assessment": {c: a.as_dict() for c, a in assessment.items()},
            "note": "Injected readings are flagged is_synthetic and carry a fault_label; "
                    "use the clear endpoint to remove them.",
        }, status=status.HTTP_201_CREATED)

    def delete(self, request):
        """
        Remove only injected readings.

        Scoped to rows carrying a ``fault_label``.  Deleting on ``is_synthetic``
        alone would also destroy the seeded demonstration network, which is
        simulated but is the application's entire dataset.
        """
        query = SensorReading.objects.filter(is_synthetic=True).exclude(fault_label="")
        node_id = _int(request, "node_id")
        if node_id is not None:
            query = query.filter(node_id=node_id)

        touched = sorted(set(query.values_list("node_id", flat=True)))
        removed, _ = query.delete()
        if touched:
            telemetry.assess_nodes(touched, persist=True)
        audit.append("fault", {"action": "cleared_injected", "removed": removed,
                               "nodes": touched}, actor=request.user.username)
        return Response({"removed": removed, "nodes": touched})


# ---------------------------------------------------------------------------
# Explainability and models
# ---------------------------------------------------------------------------
class ExplainAPIView(APIView):
    """Per-bin attribution for the dispatch-relevant prediction."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request, pk):
        try:
            node = Node.objects.get(pk=pk)
        except Node.DoesNotExist:
            return Response({"error": f"node {pk} not found"},
                            status=status.HTTP_404_NOT_FOUND)

        readings = telemetry.recent_readings([node.id]).get(node.id) or []
        if not readings:
            return Response({"error": "node has no telemetry to explain"},
                            status=status.HTTP_400_BAD_REQUEST)

        assessment = telemetry.assess_nodes([node.id], persist=False).get(node.id, {})
        row = telemetry.build_feature_row(node, readings, assessment)
        if row is None:
            return Response({"error": "could not build a feature row"},
                            status=status.HTTP_400_BAD_REQUEST)

        head = str(request.query_params.get("head", "tto")).lower()
        prefer = str(request.query_params.get("method", "auto")).lower()
        explanation = models_registry.explain_prediction(row.values, head=head,
                                                         prefer=prefer)
        if "error" in explanation:
            return Response(explanation, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        prediction = models_registry.predict_one(row.values)
        return Response({
            "node": {"id": node.id, "name": node.name},
            "prediction": prediction,
            "explanation": explanation,
            "features": {k: round(float(v), 5) for k, v in row.values.items()},
            "feature_descriptions": CORE_FEATURES.FEATURE_DESCRIPTIONS,
            "health": {c: a.as_dict() for c, a in assessment.items()},
        })


class ModelStatusAPIView(APIView):
    """Model registry, metrics, HPO protocol and continual-learning state."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        payload = models_registry.status()
        payload["registry"] = ModelVersionSerializer(
            ModelVersion.objects.order_by("-trained_at")[:20], many=True).data
        payload["global_importance"] = models_registry.global_importance()[:20]
        payload["feature_contract"] = CORE_FEATURES.describe()
        payload["hpo_protocol"] = models_registry.load_forward_meta().get("hpo_protocol", {})
        return Response(payload)


class ContinualResetAPIView(APIView):
    """Discard the online corrector, reverting to the frozen base model."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def post(self, request):
        models_registry.reset_continual_learner()
        audit.append("model", {"action": "continual_reset"}, actor=request.user.username)
        return Response({"ok": True, "status": models_registry.status()["continual"]})


# ---------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------
class EmissionsAPIView(APIView):
    """Modal CO2 breakdown, and what a flat factor would have reported."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        plan_id = _int(request, "plan_id")
        payload = dispatch.emissions_breakdown(plan_id)

        # A worked comparison across duty cycles, so the difference between a
        # collection kilometre and a haul kilometre is visible rather than asserted.
        profile = CORE_EMISSIONS.DEFAULT_PROFILE
        scenarios = []
        for label, distance, speed, friction, payload_frac, idle, lifts in (
            ("Line-haul, empty", 8000, 45, 0.05, 0.0, 0, 0),
            ("Line-haul, full", 8000, 45, 0.05, 1.0, 0, 0),
            ("Collection, free-flowing", 1000, 30, 0.15, 0.5, 16, 4),
            ("Collection, congested", 1000, 12, 0.85, 0.5, 16, 4),
            ("Stationary at a bin", 0, 1, 0.0, 0.5, 10, 2),
        ):
            leg = CORE_EMISSIONS.leg_emissions(
                distance_m=distance, speed_kmh=speed,
                payload_kg=profile.capacity_kg * payload_frac,
                friction=friction, idle_minutes=idle, lifts=lifts,
                lifted_kg=lifts * 150.0, profile=profile)
            scenarios.append({"scenario": label, **leg.as_dict(profile)})

        payload["duty_cycle_comparison"] = scenarios
        return Response(payload)


class TrafficAPIView(APIView):
    """Current traffic surface, sampled at the bins, plus any active incidents."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        provider = dispatch.traffic_provider()
        when = timezone.now()
        nodes = list(Node.objects.filter(is_active=True)
                     .exclude(latitude__isnull=True).exclude(longitude__isnull=True)
                     .only("id", "name", "latitude", "longitude"))

        samples = []
        for node in nodes:
            friction = provider.friction(node.latitude, node.longitude, when)
            samples.append({
                "node_id": node.id, "name": node.name,
                "latitude": node.latitude, "longitude": node.longitude,
                "friction": round(float(friction), 4),
                "speed_kmh": round(float(bpr_speed_kmh(friction)), 2),
            })

        incidents = (provider.active_incidents(when)
                     if hasattr(provider, "active_incidents") else [])
        hourly = []
        for hour in range(24):
            multiplier = temporal_multiplier(hour, when.weekday())
            hourly.append({"hour": hour, "multiplier": round(multiplier, 4),
                           "speed_kmh": round(bpr_speed_kmh(multiplier * 0.9), 2)})

        return Response({
            "provider": provider.describe(),
            "sampled_at": when.isoformat(),
            "samples": samples,
            "incidents": incidents,
            "hourly_profile": hourly,
            "speed_model": "Bureau of Public Roads volume-delay function, "
                           "calibrated for saturated urban arterials",
        })


# ---------------------------------------------------------------------------
# Audit ledger
# ---------------------------------------------------------------------------
class AuditLedgerAPIView(APIView):
    """Browse the tamper-evident ledger."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        limit = max(1, min(200, _int(request, "limit", 50)))
        event_type = request.query_params.get("event_type")
        query = AuditEntry.objects.order_by("-sequence")
        if event_type:
            query = query.filter(event_type=event_type)

        counts = dict(AuditEntry.objects.values_list("event_type")
                      .annotate(n=Count("id")).values_list("event_type", "n"))
        return Response({
            "head": audit.head(),
            "total": AuditEntry.objects.count(),
            "counts_by_event": counts,
            "entries": AuditEntrySerializer(query[:limit], many=True).data,
            "design": audit.describe(),
        })


class AuditVerifyAPIView(APIView):
    """Verify the chain, or produce an inclusion proof for one entry."""

    authentication_classes = _AUTH
    permission_classes = _PERM

    def get(self, request):
        sequence = _int(request, "sequence")
        if sequence is not None:
            return Response(audit.inclusion_proof(sequence))
        return Response(audit.verify(limit=_int(request, "limit")))
