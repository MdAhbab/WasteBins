"""
Reassemble a wait-bound rollout that was run as separate jobs.
=============================================================

The rollout is six independent units: two deadlines by three equity weights,
sharing only the snapshots they are drawn from. Run as one job it is the longest
thing in the study by a factor of two. Run as six it finishes in the time the
slowest one takes, and this puts the pieces back together.

A merge is only valid when the pieces agree about how they were produced, so the
protocol of every piece is compared field by field and a disagreement stops the
merge rather than producing a file that silently mixes two configurations.

Run:  python -m experiments.merge_rollouts fleet_rollout_*.json --out fleet_dhaka_equity.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

RESULTS = pathlib.Path(__file__).parent / "results"

#: Fields that describe how a rollout was produced rather than what it found.
#: Every piece must agree on all of them.
PROTOCOL = ("cycle_h", "n_cycles", "burn_in_cycles", "n_networks",
            "shift_minutes", "tau_h", "fleet", "initial_waits")

ARMS = ("equity_bound_rollout", "equity_bound_rollout_multicycle")


def merge_arm(pieces: List[Dict], arm: str) -> Dict | None:
    """Combine one arm across pieces, or return None when no piece ran it."""
    present = [p[arm] for p in pieces if p.get(arm)]
    if not present:
        return None

    head = present[0]
    for other in present[1:]:
        for field in PROTOCOL:
            if head.get(field) != other.get(field):
                raise SystemExit(
                    f"{arm}: pieces disagree on '{field}' "
                    f"({head.get(field)!r} against {other.get(field)!r}). "
                    f"These are not runs of the same experiment and merging "
                    f"them would report a configuration that never ran.")

    sweep: List[Dict] = []
    seen = set()
    for piece in present:
        for row in piece["sweep"]:
            gamma = round(float(row["gamma"]), 6)
            if gamma in seen:
                raise SystemExit(
                    f"{arm}: gamma {gamma} appears in more than one piece. "
                    f"Two jobs ran the same weight, so one of them is being "
                    f"discarded silently; check the commands.")
            seen.add(gamma)
            sweep.append(row)

    merged = dict(head)
    merged["sweep"] = sorted(sweep, key=lambda r: r["gamma"])
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pieces", nargs="+",
                        help="result files to merge, relative to results/")
    parser.add_argument("--out", default="fleet_dhaka_equity.json")
    args = parser.parse_args()

    paths = [RESULTS / p if not pathlib.Path(p).is_absolute() else pathlib.Path(p)
             for p in args.pieces]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit("missing: " + ", ".join(str(p) for p in missing))

    pieces = [json.loads(p.read_text()) for p in paths]
    print(f"merging {len(pieces)} pieces")

    out_path = RESULTS / args.out
    merged = json.loads(out_path.read_text()) if out_path.exists() else {}

    for arm in ARMS:
        combined = merge_arm(pieces, arm)
        if combined is None:
            print(f"  {arm}: no piece ran it, leaving whatever is there")
            continue
        merged[arm] = combined
        weights = ", ".join(f"{r['gamma']:.2f}" for r in combined["sweep"])
        breached = [r["gamma"] for r in combined["sweep"] if not r["holds"]]
        verdict = "holds" if not breached else f"VIOLATED at {breached}"
        print(f"  {arm}: tau {combined['tau_h']:.0f} h, gammas {weights} -> {verdict}")

    # Anything the pieces carry that the destination lacks, such as the protocol
    # block from a fresh run, is kept rather than dropped.
    for piece in pieces:
        for key, value in piece.items():
            if key not in ARMS and key not in merged:
                merged[key] = value

    out_path.write_text(json.dumps(merged, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
