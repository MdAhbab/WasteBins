"""
Audit ledger service.

Appends tamper-evident records for the events a municipal auditor cares about:
telemetry ingestion, dispatch decisions, actual collections, model activations
and detected sensor faults.  Appends are serialised under a database
transaction so two concurrent writers cannot both claim the same sequence
number and silently fork the chain.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from wastebins_core import ledger as CORE_LEDGER

from bins.models import AuditEntry

logger = logging.getLogger(__name__)


def _config() -> Dict:
    return getattr(settings, "AUDIT_LEDGER", {}) or {}


def enabled() -> bool:
    return bool(_config().get("ENABLED", True))


def block_size() -> int:
    return int(_config().get("BLOCK_SIZE", CORE_LEDGER.DEFAULT_BLOCK_SIZE))


def hmac_key() -> str:
    return str(_config().get("HMAC_KEY", "") or "")


def _to_core(row: AuditEntry) -> CORE_LEDGER.LedgerEntry:
    return CORE_LEDGER.LedgerEntry(
        sequence=row.sequence,
        event_type=row.event_type,
        payload=row.payload,
        payload_sha256=row.payload_sha256,
        prev_hash=row.prev_hash,
        entry_hash=row.entry_hash,
        timestamp=row.created_at.isoformat(),
        merkle_root=row.merkle_root or "",
        signature=row.signature or "",
        actor=row.actor,
    )


@transaction.atomic
def append(event_type: str, payload: Dict, actor: str = "system") -> Optional[AuditEntry]:
    """
    Append one entry.

    Returns ``None`` when the ledger is disabled.  Failures are logged and
    swallowed: an audit-log outage must never take down waste collection, but it
    must also never pass silently.
    """
    if not enabled():
        return None

    try:
        head = (AuditEntry.objects
                .select_for_update()
                .order_by("-sequence")
                .first())
        sequence = 0 if head is None else head.sequence + 1
        prev_hash = CORE_LEDGER.GENESIS_HASH if head is None else head.entry_hash

        created_at = timezone.now()
        entry = CORE_LEDGER.build_entry(
            sequence=sequence,
            event_type=event_type,
            payload=payload,
            prev_hash=prev_hash,
            timestamp=created_at.isoformat(),
            actor=actor,
            hmac_key=hmac_key(),
        )

        merkle_root = ""
        size = block_size()
        if CORE_LEDGER.block_boundary(sequence, size):
            start = sequence - size + 1
            leaves = list(
                AuditEntry.objects
                .filter(sequence__gte=start, sequence__lt=sequence)
                .order_by("sequence")
                .values_list("entry_hash", flat=True)
            )
            leaves.append(entry.entry_hash)
            merkle_root = CORE_LEDGER.merkle_root(leaves)

        return AuditEntry.objects.create(
            sequence=sequence,
            event_type=event_type,
            payload=payload,
            payload_sha256=entry.payload_sha256,
            prev_hash=entry.prev_hash,
            entry_hash=entry.entry_hash,
            merkle_root=merkle_root,
            signature=entry.signature,
            actor=actor,
            created_at=created_at,
        )
    except Exception:
        logger.exception("audit ledger append failed for event %s", event_type)
        return None


def verify(limit: Optional[int] = None) -> Dict:
    """Verify the chain end to end and report every anomaly found."""
    query = AuditEntry.objects.order_by("sequence")
    if limit:
        # Verify the most recent `limit` entries, still in ascending order.
        total = query.count()
        query = query[max(0, total - int(limit)):]
    entries = [_to_core(row) for row in query]
    report = CORE_LEDGER.verify_chain(entries, hmac_key=hmac_key(),
                                      block_size=block_size())
    payload = report.as_dict()
    payload["total_entries"] = AuditEntry.objects.count()
    payload["signed"] = bool(hmac_key())
    payload["block_size"] = block_size()
    return payload


def inclusion_proof(sequence: int) -> Dict:
    """
    Merkle inclusion proof for one entry against its block root.

    Lets an auditor verify a single record against a published root without
    being handed the rest of the block, which matters when the block contains
    data about other residents.
    """
    size = block_size()
    block_index = int(sequence) // size
    start = block_index * size
    end = start + size

    rows = list(AuditEntry.objects
                .filter(sequence__gte=start, sequence__lt=end)
                .order_by("sequence"))
    if not rows:
        return {"error": f"no entries in block containing sequence {sequence}"}

    leaves = [r.entry_hash for r in rows]
    try:
        index = [r.sequence for r in rows].index(int(sequence))
    except ValueError:
        return {"error": f"sequence {sequence} not found"}

    root = CORE_LEDGER.merkle_root(leaves)
    proof = CORE_LEDGER.merkle_proof(leaves, index)
    anchored = rows[-1].merkle_root if len(rows) == size else ""

    return {
        "sequence": int(sequence),
        "block_index": block_index,
        "block_start": start,
        "block_entries": len(rows),
        "block_complete": len(rows) == size,
        "leaf": leaves[index],
        "merkle_root": root,
        "anchored_root": anchored,
        "root_matches_anchor": (anchored == root) if anchored else None,
        "proof": [{"side": side, "hash": h} for side, h in proof],
        "verified": CORE_LEDGER.verify_merkle_proof(leaves[index], proof, root),
    }


def head() -> Dict:
    row = AuditEntry.objects.order_by("-sequence").first()
    if row is None:
        return {"sequence": None, "entry_hash": CORE_LEDGER.GENESIS_HASH, "empty": True}
    return {
        "sequence": row.sequence,
        "entry_hash": row.entry_hash,
        "event_type": row.event_type,
        "created_at": row.created_at.isoformat(),
        "empty": False,
    }


def describe() -> Dict:
    payload = CORE_LEDGER.describe()
    payload["enabled"] = enabled()
    payload["block_size"] = block_size()
    payload["signed"] = bool(hmac_key())
    return payload


def log_readings(records: Sequence[Dict], actor: str = "ingest") -> Optional[AuditEntry]:
    """
    Record a batch of telemetry as one entry.

    Batching keeps the ledger proportionate: at one entry per reading a 30-bin
    network at one-minute resolution would write 43k rows a day, which makes
    verification expensive without adding evidence, since a batch digest commits
    to every reading in it just as strongly.
    """
    if not records:
        return None
    return append("reading", {
        "count": len(records),
        "readings": list(records)[:512],
        "digest": CORE_LEDGER.payload_digest(list(records)),
    }, actor=actor)
