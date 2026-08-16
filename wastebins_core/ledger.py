"""
Tamper-evident audit ledger.
============================

Municipal procurement asks a hard question of any automated dispatcher: if a
resident complains that their bin overflowed, can the operator prove what the
system knew and when it knew it -- and can they prove the record was not edited
afterwards?  The first submission acknowledged the requirement and shipped
nothing.

This module provides a permissioned **hash-chained append-only ledger**.  It is
deliberately not a blockchain: there is no distributed consensus, no proof of
work and no token, because a single municipality operating its own database has
no Byzantine-agreement problem to solve.  What it does have is a need for
*tamper evidence*, and that is exactly what a hash chain with periodic Merkle
anchoring provides, at a cost of one SHA-256 per record instead of a network.

Construction
------------
Each entry commits to its predecessor::

    payload_hash = SHA256( canonical_json(payload) )
    entry_hash   = SHA256( seq || prev_hash || event_type || payload_hash || timestamp )

Because ``entry_hash`` covers ``prev_hash``, altering any historical record
invalidates every entry after it.  An adversary with write access to the
database must therefore rewrite the entire suffix, which is detectable by any
party holding an earlier root.

Every ``block_size`` entries the ledger computes a **Merkle root** over that
block's entry hashes.  Publishing that single 32-byte root (in a council
newsletter, a public repository, a timestamping service) anchors the whole block
irreversibly: an auditor can later be handed one entry plus a logarithmic
inclusion proof and verify membership without access to the rest of the data,
which matters when the data is personal.

When ``hmac_key`` is configured each entry is additionally signed with
HMAC-SHA256, so an attacker who can write to the database still cannot forge
entries without the key.

Canonical serialisation
-----------------------
Hashes are only meaningful if two parties serialise the same payload the same
way.  :func:`canonical_json` sorts keys, forbids NaN, uses fixed separators and
normalises floats, so the digest is reproducible across processes and versions.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

GENESIS_HASH = "0" * 64
DEFAULT_BLOCK_SIZE = 64


# ---------------------------------------------------------------------------
# Canonical serialisation
# ---------------------------------------------------------------------------
def _normalise(value):
    """Recursively coerce a payload into a form with a stable JSON encoding."""
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            # NaN and Infinity are not valid JSON and would serialise
            # inconsistently across implementations.
            return None
        # Round-trip through repr to avoid platform-dependent float formatting.
        return float(f"{value:.12g}")
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def canonical_json(payload) -> str:
    """Deterministic JSON encoding used for every digest in the ledger."""
    return json.dumps(_normalise(payload), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def payload_digest(payload) -> str:
    return sha256_hex(canonical_json(payload))


def entry_digest(sequence: int, prev_hash: str, event_type: str,
                 payload_sha256: str, timestamp: str) -> str:
    """
    Digest binding an entry to its position and its predecessor.

    Field separators are explicit so that no combination of field values can be
    re-partitioned into a different but identically-hashing record.
    """
    material = "|".join([
        str(int(sequence)),
        str(prev_hash),
        str(event_type),
        str(payload_sha256),
        str(timestamp),
    ])
    return sha256_hex(material)


def sign(entry_hash: str, key: str) -> str:
    if not key:
        return ""
    return hmac.new(key.encode("utf-8"), entry_hash.encode("utf-8"),
                    hashlib.sha256).hexdigest()


def verify_signature(entry_hash: str, signature: str, key: str) -> bool:
    if not key:
        return True                      # unsigned ledgers are still chain-verified
    return hmac.compare_digest(sign(entry_hash, key), signature or "")


# ---------------------------------------------------------------------------
# Merkle tree
# ---------------------------------------------------------------------------
def merkle_root(leaves: Sequence[str]) -> str:
    """
    Merkle root over hex leaf digests.

    An odd node is promoted rather than duplicated.  Duplicating the last leaf
    (the well-known Bitcoin behaviour) admits a collision in which two different
    leaf sets yield the same root, which would defeat the point of the anchor.
    """
    if not leaves:
        return GENESIS_HASH
    level = list(leaves)
    while len(level) > 1:
        nxt: List[str] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(sha256_hex(level[i] + level[i + 1]))
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def merkle_proof(leaves: Sequence[str], index: int) -> List[Tuple[str, str]]:
    """
    Inclusion proof for ``leaves[index]`` as a list of ``(side, hash)`` steps.

    ``side`` is ``"left"`` or ``"right"``, describing where the sibling sits.
    """
    if not leaves or not (0 <= index < len(leaves)):
        return []
    proof: List[Tuple[str, str]] = []
    level = list(leaves)
    idx = int(index)

    while len(level) > 1:
        nxt: List[str] = []
        new_idx: Optional[int] = None
        i = 0
        # Pair adjacent nodes; the tracked index follows its parent.
        while i + 1 < len(level):
            left, right = level[i], level[i + 1]
            if idx == i:
                proof.append(("right", right))
                new_idx = len(nxt)
            elif idx == i + 1:
                proof.append(("left", left))
                new_idx = len(nxt)
            nxt.append(sha256_hex(left + right))
            i += 2
        # An odd final node is promoted unchanged, so it gains no proof step.
        if i == len(level) - 1:
            if idx == i:
                new_idx = len(nxt)
            nxt.append(level[i])
        level = nxt
        idx = new_idx if new_idx is not None else 0
    return proof


def verify_merkle_proof(leaf: str, proof: Iterable[Tuple[str, str]], root: str) -> bool:
    current = leaf
    for side, sibling in proof:
        current = sha256_hex(sibling + current) if side == "left" else \
            sha256_hex(current + sibling)
    return current == root


# ---------------------------------------------------------------------------
# Entry model
# ---------------------------------------------------------------------------
@dataclass
class LedgerEntry:
    sequence: int
    event_type: str
    payload: Dict
    payload_sha256: str
    prev_hash: str
    entry_hash: str
    timestamp: str
    merkle_root: str = ""
    signature: str = ""
    actor: str = "system"

    def as_dict(self) -> Dict:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "payload": self.payload,
            "payload_sha256": self.payload_sha256,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
            "timestamp": self.timestamp,
            "merkle_root": self.merkle_root,
            "signature": self.signature,
            "actor": self.actor,
        }


def build_entry(sequence: int, event_type: str, payload: Dict, prev_hash: str,
                timestamp: Optional[str] = None, actor: str = "system",
                hmac_key: str = "") -> LedgerEntry:
    """Construct the next entry in a chain; pure, so it is trivially testable."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    digest = payload_digest(payload)
    e_hash = entry_digest(sequence, prev_hash, event_type, digest, ts)
    return LedgerEntry(
        sequence=int(sequence),
        event_type=str(event_type),
        payload=payload,
        payload_sha256=digest,
        prev_hash=prev_hash,
        entry_hash=e_hash,
        timestamp=ts,
        signature=sign(e_hash, hmac_key),
        actor=actor,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
@dataclass
class VerificationReport:
    valid: bool
    checked: int = 0
    first_invalid_sequence: Optional[int] = None
    errors: List[Dict] = field(default_factory=list)
    head_hash: str = GENESIS_HASH
    blocks_verified: int = 0

    def as_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "entries_checked": self.checked,
            "first_invalid_sequence": self.first_invalid_sequence,
            "errors": self.errors[:50],
            "error_count": len(self.errors),
            "head_hash": self.head_hash,
            "blocks_verified": self.blocks_verified,
        }


def verify_chain(entries: Sequence[LedgerEntry], hmac_key: str = "",
                 block_size: int = DEFAULT_BLOCK_SIZE) -> VerificationReport:
    """
    Verify an entire chain: sequencing, payload digests, link hashes,
    signatures, and every stored Merkle root.

    Reports *all* failures rather than stopping at the first, because an auditor
    needs the extent of the tampering, not just its earliest point.
    """
    report = VerificationReport(valid=True)
    prev_hash = GENESIS_HASH
    expected_seq: Optional[int] = None
    block_leaves: List[str] = []

    def fail(seq: int, kind: str, detail: str) -> None:
        report.valid = False
        if report.first_invalid_sequence is None:
            report.first_invalid_sequence = seq
        report.errors.append({"sequence": seq, "error": kind, "detail": detail})

    for entry in entries:
        report.checked += 1

        if expected_seq is None:
            expected_seq = entry.sequence
        elif entry.sequence != expected_seq:
            fail(entry.sequence, "sequence_gap",
                 f"expected {expected_seq}, found {entry.sequence}")
            expected_seq = entry.sequence
        expected_seq += 1

        recomputed_payload = payload_digest(entry.payload)
        if recomputed_payload != entry.payload_sha256:
            fail(entry.sequence, "payload_modified",
                 "stored payload does not match its recorded digest")

        if entry.prev_hash != prev_hash:
            fail(entry.sequence, "broken_link",
                 f"prev_hash {entry.prev_hash[:12]}... does not match "
                 f"predecessor {prev_hash[:12]}...")

        recomputed_entry = entry_digest(entry.sequence, entry.prev_hash,
                                        entry.event_type, entry.payload_sha256,
                                        entry.timestamp)
        if recomputed_entry != entry.entry_hash:
            fail(entry.sequence, "entry_hash_mismatch",
                 "entry hash does not match its own fields")

        if hmac_key and not verify_signature(entry.entry_hash, entry.signature, hmac_key):
            fail(entry.sequence, "bad_signature", "HMAC signature does not verify")

        block_leaves.append(entry.entry_hash)
        if entry.merkle_root:
            expected_root = merkle_root(block_leaves)
            if expected_root != entry.merkle_root:
                fail(entry.sequence, "merkle_mismatch",
                     "stored block root does not match the block contents")
            else:
                report.blocks_verified += 1
            block_leaves = []

        prev_hash = entry.entry_hash

    report.head_hash = prev_hash
    return report


def block_boundary(sequence: int, block_size: int = DEFAULT_BLOCK_SIZE) -> bool:
    """Whether an entry at ``sequence`` closes a block and should carry a root."""
    if block_size <= 0:
        return False
    return (int(sequence) + 1) % int(block_size) == 0


def describe() -> Dict:
    """Design statement, surfaced by the API and quoted in the manuscript."""
    return {
        "mechanism": "permissioned SHA-256 hash chain with periodic Merkle anchoring",
        "not_a_blockchain": (
            "No distributed consensus, proof of work or token. A single operator "
            "running its own database has no Byzantine-agreement problem; it has a "
            "tamper-evidence problem, which a hash chain solves at one hash per "
            "record instead of a network."
        ),
        "guarantees": [
            "any edit to a historical record invalidates every later entry",
            "publishing one 32-byte block root anchors that block irreversibly",
            "an entry can be proven to belong to a block with a logarithmic proof, "
            "without disclosing the rest of the block",
            "with an HMAC key configured, database write access alone is not "
            "sufficient to forge entries",
        ],
        "limits": [
            "a hash chain proves integrity, not availability: an operator who "
            "deletes the whole ledger leaves no evidence beyond the missing "
            "published roots",
            "anchoring frequency bounds how far back an undetected rewrite could "
            "reach, so roots should be published at least once per block",
        ],
        "block_size_default": DEFAULT_BLOCK_SIZE,
        "hash": "SHA-256",
        "signature": "HMAC-SHA256 (optional)",
    }
