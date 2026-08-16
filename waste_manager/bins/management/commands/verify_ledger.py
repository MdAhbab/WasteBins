"""Verify the tamper-evident audit ledger from the command line."""
from __future__ import annotations

import json

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Verify the audit ledger chain, signatures and Merkle block roots."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=None,
                            help="Verify only the most recent N entries.")
        parser.add_argument("--proof", type=int, default=None,
                            help="Emit a Merkle inclusion proof for one sequence number.")
        parser.add_argument("--json", action="store_true", help="Machine-readable output.")

    def handle(self, *args, **options):
        from bins.services import audit

        if options["proof"] is not None:
            proof = audit.inclusion_proof(int(options["proof"]))
            if options["json"]:
                self.stdout.write(json.dumps(proof, indent=2))
                return
            if "error" in proof:
                self.stdout.write(self.style.ERROR(proof["error"]))
                return
            self.stdout.write(f"Sequence {proof['sequence']} in block {proof['block_index']}")
            self.stdout.write(f"  leaf         {proof['leaf']}")
            self.stdout.write(f"  merkle root  {proof['merkle_root']}")
            self.stdout.write(f"  proof steps  {len(proof['proof'])}")
            style = self.style.SUCCESS if proof["verified"] else self.style.ERROR
            self.stdout.write(style(f"  verified     {proof['verified']}"))
            return

        report = audit.verify(limit=options["limit"])
        if options["json"]:
            self.stdout.write(json.dumps(report, indent=2))
            return

        self.stdout.write(f"Entries in ledger : {report['total_entries']:,}")
        self.stdout.write(f"Entries checked   : {report['entries_checked']:,}")
        self.stdout.write(f"Blocks verified   : {report['blocks_verified']}")
        self.stdout.write(f"Signed            : {report['signed']}")
        self.stdout.write(f"Head hash         : {report['head_hash']}")

        if report["valid"]:
            self.stdout.write(self.style.SUCCESS("\nLedger is intact."))
        else:
            self.stdout.write(self.style.ERROR(
                f"\nTAMPERING DETECTED: {report['error_count']} problem(s), "
                f"first at sequence {report['first_invalid_sequence']}."))
            for err in report["errors"][:10]:
                self.stdout.write(self.style.ERROR(
                    f"  #{err['sequence']}  {err['error']}: {err['detail']}"))
