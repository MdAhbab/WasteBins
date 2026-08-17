"""
Dispatch: from stored telemetry to an executable, auditable fleet plan.

Pipeline
--------
1. Read the recent telemetry window for every active bin.
2. Assess sensor health, producing per-channel trust.
3. Score each bin: the interpretable rule under trust-weighted renormalisation,
   blended with the learned forward risk in proportion to surviving evidence.
4. Apply the equity aging term and the lexicographic hazard tier.
5. Build the routing problem with the real fleet, the traffic surface and the
   physical load of each bin.
6. Solve the prize-collecting CVRPTW, persist the plan, and write it to the
   audit ledger.

Every step is bounded in cost, and the expensive one (solving) runs only when a
plan is explicitly requested -- never on a dashboard render.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from wastebins_core import aging as CORE_AGING
from wastebins_core import emissions as CORE_EMISSIONS
from wastebins_core import features as CORE_FEATURES
from wastebins_core import metaheuristics as CORE_META
from wastebins_core import priority as CORE_PRIORITY
from wastebins_core import scenario as CORE_SCENARIO
from wastebins_core import traffic as CORE_TRAFFIC
from wastebins_core import vrp as CORE_VRP

from bins.models import (Depot, Node, RoutePlan, RouteStop, ServiceEvent,
                         Vehicle)
from bins.services import audit, models_registry, telemetry

logger = logging.getLogger(__name__)

_PROVIDER_CACHE: Dict[str, CORE_TRAFFIC.TrafficProvider] = {}

# Nominal hours between dispatch runs.  Used only to report the equity wait
# bound, which is stated per cycle: a bin promoted to the overdue tier waits at
# most tau plus the time needed to clear the bins already overdue ahead of it.
DISPATCH_CYCLE_H = 12.0


def json_safe(value):
    """
    Replace non-finite floats with ``None``.

    ``inf`` is a meaningful value internally -- it is how "no guaranteed bound"
    and "no overflow deadline" are represented -- but it is not valid JSON, and
    serialising it raises rather than degrading, taking the endpoint down.
    """
    if isinstance(value, float):
        return None if (math.isinf(value) or math.isnan(value)) else value
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Traffic
# ---------------------------------------------------------------------------
def traffic_provider() -> CORE_TRAFFIC.TrafficProvider:
    """Provider built from settings, cached so live-feed caches survive requests."""
    config = getattr(settings, "TRAFFIC", {}) or {}
    key = f"{config.get('PROVIDER')}|{config.get('API_URL')}"
    provider = _PROVIDER_CACHE.get(key)
    if provider is None:
        provider = CORE_TRAFFIC.make_provider(config)
        _PROVIDER_CACHE.clear()
        _PROVIDER_CACHE[key] = provider
    return provider


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------
def ensure_default_fleet() -> Tuple[Depot, List[Vehicle]]:
    """Create a depot and a starter fleet on first use, so the app is never empty."""
    defaults = getattr(settings, "FLEET_DEFAULTS", {})
    depot, _ = Depot.objects.get_or_create(
        name="Mirpur Central Depot",
        defaults={
            "latitude": float(defaults.get("DEPOT_LAT", 23.8069)),
            "longitude": float(defaults.get("DEPOT_LNG", 90.3687)),
        },
    )
    vehicles = list(Vehicle.objects.filter(is_active=True))
    if not vehicles:
        specs = [
            ("Truck-01", 6000.0, "euro4", ["general", "organic", "recyclable"]),
            ("Truck-02", 4500.0, "euro5", ["general", "recyclable"]),
            ("Truck-03", 3000.0, "euro6", ["general", "organic", "hazardous"]),
        ]
        for name, capacity, euro, streams in specs:
            vehicles.append(Vehicle.objects.create(
                name=name, depot=depot, capacity_kg=capacity,
                shift_minutes=float(defaults.get("SHIFT_MINUTES", 480.0)),
                avg_speed_kmh=float(defaults.get("AVG_SPEED_KMH", 20.0)),
                euro_class=euro, accepts_streams=streams,
            ))
    return depot, vehicles


def vehicle_specs(vehicles: Sequence[Vehicle], depot_index: int = 0
                  ) -> List[CORE_VRP.VehicleSpec]:
    specs = []
    for v in vehicles:
        specs.append(CORE_VRP.VehicleSpec(
            vehicle_id=v.id,
            name=v.name,
            depot_index=depot_index,
            capacity_kg=float(v.capacity_kg),
            body_volume_m3=float(getattr(v, "body_volume_m3", 0.0) or 0.0),
            shift_minutes=float(v.shift_minutes),
            shift_start_minute=int(v.shift_start_minute),
            avg_speed_kmh=float(v.avg_speed_kmh),
            compaction_ratio=float(v.compaction_ratio or 1.0),
            accepts_streams=tuple(v.accepts_streams or ()),
            profile=CORE_EMISSIONS.profile_from_vehicle(
                float(v.capacity_kg), float(v.kerb_mass_kg), v.euro_class, v.name),
        ))
    return specs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_bins(nodes: Sequence[Node], user_lat: Optional[float] = None,
               user_lng: Optional[float] = None, now=None,
               policy: str = CORE_PRIORITY.POLICY_TRUST_WEIGHTED,
               use_model: bool = True, persist_health: bool = False) -> Dict[int, Dict]:
    """
    Score every bin, returning the full reasoning trail rather than just a number.

    The trail is what the explainability and audit surfaces consume: which
    channels were trusted, what the rule said, what the model said, and how the
    two were blended.

    ``persist_health`` defaults to ``False`` so that read-only callers -- above
    all the polled dashboard -- do not write to the database on every ``GET``.
    Ingestion, an explicit health refresh and plan generation pass ``True``.
    """
    now = now or timezone.now()
    node_ids = [n.id for n in nodes]
    if not node_ids:
        return {}

    # Fetch the telemetry windows once and share them with the health pass, so a
    # single request never queries the same rows twice.
    windows = telemetry.recent_readings(node_ids)
    assessments = telemetry.assess_nodes(node_ids, persist=persist_health,
                                         readings=windows)

    node_by_id = {n.id: n for n in nodes}
    feature_rows: List[Dict[str, float]] = []
    feature_owners: List[int] = []

    results: Dict[int, Dict] = {}

    from wastebins_core.geo import haversine

    for node_id in node_ids:
        node = node_by_id[node_id]
        readings = windows.get(node_id) or []
        assessment = assessments.get(node_id, {})

        values: Dict[str, Optional[float]] = {}
        trust: Dict[str, float] = {}
        for field in telemetry.HEALTH_CHANNELS:
            channel = assessment.get(field)
            if channel is not None:
                values[field] = channel.value
                trust[field] = channel.trust
            elif readings:
                raw = getattr(readings[-1], field, None)
                values[field] = None if raw is None else float(raw)
                trust[field] = 1.0

        if user_lat is not None and user_lng is not None and \
                node.latitude is not None and node.longitude is not None:
            values["distance_m"] = haversine(user_lat, user_lng,
                                             node.latitude, node.longitude)
            trust["distance_m"] = 1.0

        rule = CORE_PRIORITY.compute_priority(
            values, trust=trust,
            features=getattr(settings, "DYNAMIC_FEATURES", None) or CORE_PRIORITY.DEFAULT_FEATURES,
            policy=policy,
        )

        results[node_id] = {
            "node_id": node_id,
            "name": node.name,
            "rule": rule.as_dict(),
            "actionable": CORE_PRIORITY.is_actionable(rule),
            "health": {f: a.as_dict() for f, a in assessment.items()},
            "fill": values.get("waste_level"),
            "hours_since_collection": round(node.hours_since_collection(now), 2),
            "model": None,
            "priority": rule.score,
            "hazard_prob": 0.0,
        }

        if use_model and readings:
            row = telemetry.build_feature_row(node, readings, assessment, now=now)
            if row is not None:
                feature_rows.append(row.values)
                feature_owners.append(node_id)

    if feature_rows:
        predictions = models_registry.predict_batch(feature_rows)
        for node_id, prediction in zip(feature_owners, predictions):
            if not prediction:
                continue
            entry = results[node_id]
            entry["model"] = prediction
            entry["hazard_prob"] = float(prediction.get("hazard_prob", 0.0))
            entry["priority"] = CORE_PRIORITY.blend_with_model(
                entry["rule"]["score"],
                prediction.get("risk_priority"),
                confidence=entry["rule"]["confidence"],
            )

    return results


def apply_equity(scores: Dict[int, Dict], nodes: Sequence[Node], now=None,
                 gamma: Optional[float] = None, tau_h: Optional[float] = None,
                 kappa: float = CORE_AGING.DEFAULT_KAPPA) -> Dict[int, CORE_AGING.TieredPriority]:
    now = now or timezone.now()
    gamma = float(settings.ROUTING_AGING_GAMMA if gamma is None else gamma)
    tau_h = float(settings.ROUTING_AGING_TAU_H if tau_h is None else tau_h)

    priorities = {nid: float(entry["priority"]) for nid, entry in scores.items()}
    hazards = {nid: float(entry.get("hazard_prob", 0.0)) for nid, entry in scores.items()}
    waits = {n.id: n.hours_since_collection(now) for n in nodes if n.id in priorities}

    return CORE_AGING.effective_priorities(priorities, waits, hazards,
                                           gamma=gamma, tau_h=tau_h, kappa=kappa)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def build_problem(nodes: Sequence[Node], vehicles: Sequence[Vehicle], depot: Depot,
                  scores: Dict[int, Dict], tiered: Dict[int, CORE_AGING.TieredPriority],
                  when: Optional[datetime] = None, use_traffic: bool = True):
    """Assemble the routing instance from live state."""
    when = when or timezone.now()
    usable = [n for n in nodes
              if n.latitude is not None and n.longitude is not None and n.id in scores]

    coords = [(float(depot.latitude), float(depot.longitude))]
    index_of: Dict[int, int] = {}
    for node in usable:
        index_of[node.id] = len(coords)
        coords.append((float(node.latitude), float(node.longitude)))

    travel = CORE_SCENARIO.build_travel(
        coords, when, provider=traffic_provider(),
        default_speed_kmh=float(getattr(settings, "FLEET_DEFAULTS", {}).get("AVG_SPEED_KMH", 20.0)),
        use_traffic=use_traffic,
    )

    fills, prizes, hazards, tiers, tto, streams = {}, {}, {}, {}, {}, {}
    capacities, densities, service, windows = {}, {}, {}, {}

    for node in usable:
        entry = scores[node.id]
        tp = tiered.get(node.id)
        fills[node.id] = float(entry.get("fill") or 0.0)
        prizes[node.id] = float(tp.score if tp else entry["priority"])
        hazards[node.id] = bool(tp.tier == 0 if tp else False)
        tiers[node.id] = int(tp.tier if tp else CORE_AGING.TIER_NORMAL)
        model = entry.get("model") or {}
        # Plan against the pessimistic P10 bound rather than the median: at the
        # median, half of all overflow deadlines would be a coin flip.  A bin is
        # only given a finite deadline when the model actually expects an
        # overflow inside the horizon -- otherwise the censored cap would be
        # mistaken for a real deadline and every bin would look urgent.
        p10 = model.get("tto_p10_h")
        cap = float(CORE_FEATURES.TTO_CAP_H)
        deadline = math.inf
        if p10 is not None and float(p10) < cap - 0.1:
            deadline = float(p10)
        elif float(model.get("hazard_prob", 0.0)) >= 0.5:
            deadline = float(CORE_FEATURES.HORIZON_H)
        tto[node.id] = deadline
        streams[node.id] = node.waste_stream
        capacities[node.id] = float(node.capacity_liters)
        densities[node.id] = float(node.waste_density_kg_per_m3)
        service[node.id] = float(node.service_minutes)
        windows[node.id] = (float(node.window_start_min), float(node.window_end_min))

    tasks = CORE_SCENARIO.make_tasks(
        [n.id for n in usable], fills, prizes, index_of=index_of, hazards=hazards,
        tiers=tiers, tto_hours=tto, streams=streams, capacities_l=capacities,
        densities=densities, service_minutes=service, windows=windows,
    )
    specs = vehicle_specs(vehicles, depot_index=0)
    return tasks, specs, travel, usable


@transaction.atomic
def persist_plan(plan: CORE_VRP.FleetPlan, params: Dict, user=None) -> RoutePlan:
    record = RoutePlan.objects.create(
        algorithm=plan.algorithm,
        params=params,
        metrics=plan.metrics,
        horizon_start=timezone.now(),
        generated_by=user if (user is not None and user.is_authenticated) else None,
        compute_ms=plan.compute_ms,
    )
    stops = []
    for route in plan.routes:
        for sequence, stop in enumerate(route.stops):
            stops.append(RouteStop(
                plan=record,
                vehicle_id=route.vehicle.vehicle_id,
                node_id=stop.task.node_id,
                sequence=sequence,
                arrival_min=stop.arrival_min,
                departure_min=stop.departure_min,
                leg_distance_m=stop.leg_distance_m,
                leg_co2_kg=stop.leg_co2_kg,
                load_after_kg=stop.load_after_kg,
                priority=stop.task.prize,
            ))
    if stops:
        RouteStop.objects.bulk_create(stops)
    return record


def generate_plan(user=None, algorithm: str = "proposed",
                  user_lat: Optional[float] = None, user_lng: Optional[float] = None,
                  time_budget_s: float = 5.0, use_traffic: bool = True,
                  gamma: Optional[float] = None, tau_h: Optional[float] = None,
                  persist: bool = True, when: Optional[datetime] = None) -> Dict:
    """
    Produce a dispatch plan end to end.

    This is the only entry point that runs the solver, and it is only reachable
    from an explicit action, so no page render can trigger it.
    """
    started = time.perf_counter()
    when = when or timezone.now()

    depot, vehicles = ensure_default_fleet()
    nodes = list(Node.objects.filter(is_active=True).select_related("group"))
    if not nodes:
        return {"error": "no active bins configured"}

    scores = score_bins(nodes, user_lat=user_lat, user_lng=user_lng, now=when,
                        persist_health=True)
    tiered = apply_equity(scores, nodes, now=when, gamma=gamma, tau_h=tau_h)
    tasks, specs, travel, usable = build_problem(nodes, vehicles, depot, scores,
                                                 tiered, when=when,
                                                 use_traffic=use_traffic)
    if not tasks:
        return {"error": "no bins have usable coordinates and telemetry"}

    weights = CORE_VRP.ObjectiveWeights()
    plan = CORE_META.solve_with(algorithm, tasks, specs, travel, weights,
                                time_budget_s=time_budget_s)
    if plan is None:
        return {"error": f"solver '{algorithm}' is not available in this deployment"}

    gamma_used = float(settings.ROUTING_AGING_GAMMA if gamma is None else gamma)
    tau_used = float(settings.ROUTING_AGING_TAU_H if tau_h is None else tau_h)
    params = {
        "algorithm": algorithm,
        "n_bins": len(tasks),
        "n_vehicles": len(specs),
        "traffic_provider": traffic_provider().describe(),
        "use_traffic": use_traffic,
        "aging": {
            "gamma": gamma_used,
            "tau_h": tau_used,
            "kappa": CORE_AGING.DEFAULT_KAPPA,
            # The bound comes from the overdue tier, not from gamma.  It assumes
            # the fleet clears overdue bins at least as fast as they are
            # promoted; `equity_report` reports the observed overdue count so
            # that assumption is visible rather than implied.
            "worst_case_wait_bound_h": json_safe(CORE_AGING.worst_case_wait_bound(
                tau_h=tau_used, cycle_h=DISPATCH_CYCLE_H)),
            "bound_assumes": (
                f"one dispatch cycle of {DISPATCH_CYCLE_H:.0f} h and that the fleet "
                f"clears overdue bins as fast as they are promoted"),
        },
        "objective_weights": asdict(weights),
        "model_version": models_registry.get_model_version(),
        "planned_for": when.isoformat(),
    }

    record = persist_plan(plan, params, user=user) if persist else None

    payload = CORE_VRP.plan_to_dict(plan)
    payload["plan_id"] = record.id if record else None
    payload["params"] = params
    payload["scores"] = {
        str(nid): {
            "priority": round(float(entry["priority"]), 4),
            "effective_priority": round(float(tiered[nid].score), 4) if nid in tiered else None,
            "tier": tiered[nid].tier if nid in tiered else 1,
            "hazard_prob": round(float(entry.get("hazard_prob", 0.0)), 4),
            "confidence": entry["rule"]["confidence"],
            "actionable": entry["actionable"],
            "hours_since_collection": entry["hours_since_collection"],
        }
        for nid, entry in scores.items()
    }
    payload["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    payload = json_safe(payload)

    if record is not None:
        audit.append("plan", {
            "plan_id": record.id,
            "algorithm": plan.algorithm,
            "metrics": plan.metrics,
            "params": params,
            "routes": [{"vehicle_id": r.vehicle.vehicle_id, "nodes": r.node_ids}
                       for r in plan.routes],
            "unserved": [t.node_id for t in plan.unserved],
        }, actor=getattr(user, "username", "system") or "system")

    return payload


@transaction.atomic
def record_service(node_id: int, vehicle_id: Optional[int] = None,
                   plan_id: Optional[int] = None, fill: Optional[float] = None,
                   collected_at=None, actor: str = "crew") -> Optional[ServiceEvent]:
    """
    Record that a bin was actually emptied.

    This closes the loop: it resets the aging clock, supplies the realised
    outcome the continual learner trains on, and creates the auditable evidence
    that the dispatch decision was carried out.
    """
    try:
        node = Node.objects.select_for_update().get(pk=node_id)
    except Node.DoesNotExist:
        return None

    collected_at = collected_at or timezone.now()
    previous = node.last_collected_at
    wait_hours = ((collected_at - previous).total_seconds() / 3600.0) if previous else 0.0

    if fill is None:
        latest = node.get_latest_reading()
        fill = float(latest.waste_level) if latest and latest.waste_level is not None else 0.0

    event = ServiceEvent.objects.create(
        node=node,
        vehicle_id=vehicle_id,
        plan_id=plan_id,
        collected_at=collected_at,
        fill_at_collection=float(fill),
        load_kg=node.load_kg(float(fill)),
        wait_hours=round(wait_hours, 3),
        was_overflowing=float(fill) >= 1.0,
    )
    node.last_collected_at = collected_at
    node.save(update_fields=["last_collected_at"])

    audit.append("service", {
        "node_id": node.id,
        "node": node.name,
        "vehicle_id": vehicle_id,
        "plan_id": plan_id,
        "collected_at": collected_at.isoformat(),
        "fill_at_collection": round(float(fill), 4),
        "wait_hours": round(wait_hours, 3),
        "was_overflowing": bool(float(fill) >= 1.0),
    }, actor=actor)
    return event


def equity_report(days: int = 30) -> Dict:
    """Service-equity statistics from the actual collection history."""
    since = timezone.now() - timedelta(days=int(days))
    waits = list(
        ServiceEvent.objects.filter(collected_at__gte=since, wait_hours__gt=0)
        .values_list("wait_hours", flat=True)
    )
    report = CORE_AGING.equity_report(waits)
    gamma = float(settings.ROUTING_AGING_GAMMA)
    tau = float(settings.ROUTING_AGING_TAU_H)
    overdue = [w for w in waits if w >= tau]
    bound = CORE_AGING.worst_case_wait_bound(
        tau_h=tau, cycle_h=DISPATCH_CYCLE_H,
        max_overdue=max(1, len(overdue)), served_overdue_per_cycle=1)
    report.update({
        "window_days": int(days),
        "gamma": gamma,
        "tau_h": tau,
        "kappa": CORE_AGING.DEFAULT_KAPPA,
        "overdue_now": len(overdue),
        "guaranteed_bound_h": json_safe(bound),
        "bound_basis": (
            "tau plus one cycle per outstanding overdue bin, assuming the fleet "
            "clears at least one overdue bin per cycle"),
        "bound_is_finite": math.isfinite(bound),
        "bound_respected": (report["worst_wait_h"] <= bound) if waits else None,
    })
    return json_safe(report)


def emissions_breakdown(plan_id: Optional[int] = None) -> Dict:
    """
    Modal CO2 breakdown for a plan, alongside what a flat factor would report.

    Showing both is the point: the flat model cannot distinguish a line-haul
    kilometre from a collection kilometre, and the gap between them is where the
    abatement opportunity actually lives.
    """
    record = (RoutePlan.objects.filter(pk=plan_id).first() if plan_id
              else RoutePlan.objects.order_by("-timestamp").first())
    if record is None:
        return {"error": "no route plan available"}

    metrics = record.metrics or {}
    distance_km = float(metrics.get("distance_km", 0.0))
    co2_kg = float(metrics.get("co2_kg", 0.0))
    flat_reference = 1.05        # the constant used in the original submission

    return {
        "plan_id": record.id,
        "algorithm": record.algorithm,
        "distance_km": round(distance_km, 3),
        "modal_co2_kg": round(co2_kg, 3),
        "modal_kg_per_km": round(co2_kg / distance_km, 4) if distance_km > 1e-9 else 0.0,
        "flat_factor_co2_kg": round(distance_km * flat_reference, 3),
        "flat_factor_kg_per_km": flat_reference,
        "difference_pct": round(100.0 * (co2_kg - distance_km * flat_reference)
                                / max(distance_km * flat_reference, 1e-9), 2),
        "model": CORE_EMISSIONS.describe_model(),
        "note": ("The flat factor cannot separate line-haul from collection duty. "
                 "The modal model reports cruise, stop-and-go, idle and compaction "
                 "separately, which is where the abatement levers are."),
    }
