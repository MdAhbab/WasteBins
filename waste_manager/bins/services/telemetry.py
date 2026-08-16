"""
Telemetry access and sensor-health assessment.

Turns stored ``SensorReading`` rows into the windowed structures the core
algorithms expect, runs the validation ensemble, and persists the resulting
per-channel trust so the dashboard and the planner see the same state.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from django.utils import timezone

from wastebins_core import features as CORE_FEATURES
from wastebins_core import health as CORE_HEALTH

from bins.models import Node, SensorHealth, SensorReading

# Model field -> core channel name.
CHANNEL_FIELDS = {
    "waste_level": "waste",
    "gas_level": "gas",
    "temperature": "temp",
    "humidity": "humidity",
}
HEALTH_CHANNELS = tuple(CHANNEL_FIELDS.keys())

DEFAULT_WINDOW = CORE_FEATURES.ROLL


_SELECT_FIELDS = ("id", "node_id", "timestamp", "waste_level", "gas_level",
                  "temperature", "humidity", "traffic_density", "is_synthetic",
                  "fault_label")


def recent_readings(node_ids: Sequence[int], window: int = DEFAULT_WINDOW
                    ) -> Dict[int, List[SensorReading]]:
    """
    Latest ``window`` readings per node, oldest first.

    Uses a ``ROW_NUMBER()`` window function so the database returns only the
    ``window`` rows per node that are actually needed.  The obvious alternative
    -- one query filtered by ``node_id__in`` and partitioned in Python -- reads
    the *entire* history of every requested bin; at a few weeks of telemetry
    that is tens of thousands of rows discarded per dashboard refresh, and it is
    what made the endpoint slow.

    Falls back to one indexed query per node if the backend lacks window
    functions (SQLite below 3.25, MySQL below 8).
    """
    node_ids = [int(n) for n in node_ids]
    if not node_ids:
        return {}
    window = max(1, int(window))

    try:
        return _recent_readings_windowed(node_ids, window)
    except Exception:                       # pragma: no cover - legacy backends
        return _recent_readings_per_node(node_ids, window)


def _recent_readings_windowed(node_ids: Sequence[int], window: int
                              ) -> Dict[int, List[SensorReading]]:
    from django.db import connection

    table = SensorReading._meta.db_table
    placeholders = ", ".join(["%s"] * len(node_ids))
    columns = ", ".join(_SELECT_FIELDS)
    sql = f"""
        SELECT {columns} FROM (
            SELECT {columns},
                   ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY timestamp DESC, id DESC) AS rn
            FROM {table}
            WHERE node_id IN ({placeholders})
        ) ranked
        WHERE rn <= %s
        ORDER BY node_id ASC, timestamp ASC, id ASC
    """
    grouped: Dict[int, List[SensorReading]] = {nid: [] for nid in node_ids}
    with connection.cursor() as cursor:
        cursor.execute(sql, [*node_ids, window])
        for row in cursor.fetchall():
            values = dict(zip(_SELECT_FIELDS, row))
            reading = SensorReading(**values)
            grouped.setdefault(reading.node_id, []).append(reading)
    return grouped


def _recent_readings_per_node(node_ids: Sequence[int], window: int
                              ) -> Dict[int, List[SensorReading]]:
    grouped: Dict[int, List[SensorReading]] = {}
    for node_id in node_ids:
        rows = list(SensorReading.objects
                    .filter(node_id=node_id)
                    .order_by("-timestamp", "-id")
                    .only(*_SELECT_FIELDS)[:window])
        grouped[node_id] = list(reversed(rows))
    return grouped


def readings_to_history(readings: Sequence[SensorReading]) -> Dict[str, List[float]]:
    """Per-channel value windows, with ``NaN`` for genuinely missing samples."""
    history: Dict[str, List[float]] = {ch: [] for ch in HEALTH_CHANNELS}
    for reading in readings:
        for field in HEALTH_CHANNELS:
            value = getattr(reading, field, None)
            history[field].append(float("nan") if value is None else float(value))
    return history


def readings_to_hours(readings: Sequence[SensorReading]) -> List[float]:
    """Elapsed hours of each sample relative to the oldest in the window."""
    if not readings:
        return []
    start = readings[0].timestamp
    return [(r.timestamp - start).total_seconds() / 3600.0 for r in readings]


def load_previous_trust(node_ids: Sequence[int]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for row in SensorHealth.objects.filter(node_id__in=list(node_ids)):
        out.setdefault(row.node_id, {})[row.channel] = float(row.trust)
    return out


def assess_nodes(node_ids: Sequence[int], window: int = DEFAULT_WINDOW,
                 persist: bool = True,
                 readings: Optional[Dict[int, List[SensorReading]]] = None
                 ) -> Dict[int, Dict[str, CORE_HEALTH.ChannelAssessment]]:
    """
    Run the validation ensemble over the fleet and optionally persist the state.

    Assessment is fleet-wide rather than per-node because the peer-consistency
    detector -- the only one that sees a stealthy poisoning attack -- needs the
    simultaneous distribution across other bins.

    ``readings`` may be supplied by a caller that has already fetched the
    windows, so a request does not pay for the same query twice.
    """
    node_ids = list(node_ids)
    if not node_ids:
        return {}

    if readings is None:
        readings = recent_readings(node_ids, window)
    histories = {nid: readings_to_history(items) for nid, items in readings.items()
                 if items}
    if not histories:
        return {}

    previous = load_previous_trust(histories.keys())
    # Core works in its own channel vocabulary; translate both ways.
    core_histories = {
        nid: {CHANNEL_FIELDS[f]: values for f, values in hist.items()}
        for nid, hist in histories.items()
    }
    core_previous = {
        nid: {CHANNEL_FIELDS[f]: t for f, t in trusts.items() if f in CHANNEL_FIELDS}
        for nid, trusts in previous.items()
    }
    core_results = CORE_HEALTH.assess_fleet(core_histories, previous_trust=core_previous)

    inverse = {v: k for k, v in CHANNEL_FIELDS.items()}
    results: Dict[int, Dict[str, CORE_HEALTH.ChannelAssessment]] = {}
    for nid, channels in core_results.items():
        results[nid] = {inverse[c]: assessment for c, assessment in channels.items()
                        if c in inverse}

    if persist:
        _persist_health(results)
    return results


def _persist_health(results: Dict[int, Dict[str, CORE_HEALTH.ChannelAssessment]]) -> None:
    now = timezone.now()
    existing = {
        (row.node_id, row.channel): row
        for row in SensorHealth.objects.filter(node_id__in=list(results.keys()))
    }
    to_create: List[SensorHealth] = []
    to_update: List[SensorHealth] = []

    for node_id, channels in results.items():
        for channel, assessment in channels.items():
            row = existing.get((node_id, channel))
            payload = dict(
                status=assessment.status,
                trust=assessment.trust,
                drift_estimate=assessment.drift_estimate,
                stuck_streak=assessment.stuck_streak,
                missing_streak=assessment.missing_streak,
                detail={"scores": assessment.scores, "flags": assessment.flags},
            )
            if assessment.status == CORE_HEALTH.STATUS_OK:
                payload["last_ok_at"] = now
            if row is None:
                to_create.append(SensorHealth(node_id=node_id, channel=channel, **payload))
            else:
                for key, value in payload.items():
                    setattr(row, key, value)
                to_update.append(row)

    if to_create:
        SensorHealth.objects.bulk_create(to_create, ignore_conflicts=True)
    if to_update:
        SensorHealth.objects.bulk_update(
            to_update,
            ["status", "trust", "drift_estimate", "stuck_streak",
             "missing_streak", "detail", "last_ok_at"],
        )


def build_feature_row(node: Node, readings: Sequence[SensorReading],
                      assessment: Optional[Dict[str, CORE_HEALTH.ChannelAssessment]] = None,
                      now=None) -> Optional[CORE_FEATURES.FeatureRow]:
    """
    Build the model input for one node from its reading window.

    Values are taken from the health assessment when one is available, so a
    drift-corrected or clamped reading -- not the raw one -- reaches the model,
    matching how the training data was prepared.
    """
    if not readings:
        return None

    history = readings_to_history(readings)
    hours = readings_to_hours(readings)

    if assessment:
        for field, channel_assessment in assessment.items():
            if field not in history or not history[field]:
                continue
            history[field] = list(history[field])
            history[field][-1] = (float("nan") if channel_assessment.value is None
                                  else float(channel_assessment.value))

    core_window = {CHANNEL_FIELDS[f]: values for f, values in history.items()}
    latest = readings[-1]
    now = now or timezone.now()

    waste = np.asarray(core_window.get("waste", []), dtype=float)
    finite = np.flatnonzero(np.isfinite(waste))
    staleness = 0.0 if finite.size == 0 else float(hours[-1] - hours[int(finite[-1])])

    return CORE_FEATURES.build_row(
        core_window,
        hours,
        timestamp_hour=latest.timestamp.hour + latest.timestamp.minute / 60.0,
        timestamp_dow=latest.timestamp.weekday(),
        dwell_h=node.hours_since_collection(now),
        staleness_h=staleness,
    )


def latest_reading_map(node_ids: Sequence[int]) -> Dict[int, SensorReading]:
    """Most recent reading per node, in a single query."""
    windows = recent_readings(node_ids, window=1)
    return {nid: items[-1] for nid, items in windows.items() if items}


def health_summary(results: Dict[int, Dict[str, CORE_HEALTH.ChannelAssessment]]) -> Dict:
    """Fleet-level rollup for the dashboard."""
    counts = {CORE_HEALTH.STATUS_OK: 0, CORE_HEALTH.STATUS_SUSPECT: 0,
              CORE_HEALTH.STATUS_DEGRADED: 0, CORE_HEALTH.STATUS_FAILED: 0}
    trusts: List[float] = []
    flagged_nodes = set()
    flag_counts: Dict[str, int] = {}

    for node_id, channels in results.items():
        for assessment in channels.values():
            counts[assessment.status] = counts.get(assessment.status, 0) + 1
            trusts.append(assessment.trust)
            if assessment.status != CORE_HEALTH.STATUS_OK:
                flagged_nodes.add(node_id)
            for flag in assessment.flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return {
        "channels_assessed": len(trusts),
        "status_counts": counts,
        "mean_trust": round(float(np.mean(trusts)), 4) if trusts else 1.0,
        "min_trust": round(float(np.min(trusts)), 4) if trusts else 1.0,
        "nodes_flagged": len(flagged_nodes),
        "flagged_node_ids": sorted(flagged_nodes),
        "top_flags": sorted(flag_counts.items(), key=lambda kv: -kv[1])[:6],
    }
