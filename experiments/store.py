"""
Append-only raw records, one line per solve.
============================================

Every revised experiment writes what it measured here before anything is
summarised, as JSON lines under ``results/raw/``.  A table or a figure is then a
function of these files and of nothing else, which is what makes "the numbers in
the paper come from the logs" checkable.

Two properties are needed and both are simple.

*Resumable.*  A run is a set of units, each with a key.  A unit whose key is
already in the store is skipped, so an interrupted run continues where it
stopped and a finished run does nothing when started again.

*Parallel.*  Each worker writes its own file, named for its shard, so no two
processes ever write to one file.  Reading a study reads every shard.
"""
from __future__ import annotations

import json
import os
import pathlib
import platform
import sys
import time
from typing import Dict, Iterable, Iterator, List, Optional

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
RAW = RESULTS / "raw"


def shard_path(study: str, shard: int) -> pathlib.Path:
    RAW.mkdir(parents=True, exist_ok=True)
    return RAW / f"{study}.s{int(shard):02d}.jsonl"


def read(study: str) -> List[Dict]:
    """Every record of a study, across all shards, in file order."""
    records: List[Dict] = []
    if not RAW.exists():
        return records
    for path in sorted(RAW.glob(f"{study}.s*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A worker stopped mid-write.  The partial line is the last
                    # one in its file, the unit has no complete record, and it
                    # is simply run again.
                    continue
    return records


def keys(study: str) -> set:
    return {r["key"] for r in read(study) if "key" in r}


def index(study: str) -> Dict[str, Dict]:
    """Records by key.  A key written twice keeps its last record."""
    return {r["key"]: r for r in read(study) if "key" in r}


def append(study: str, shard: int, record: Dict) -> None:
    line = json.dumps(record, separators=(",", ":"), default=_plain)
    with shard_path(study, shard).open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _plain(value):
    """JSON fallback for NumPy scalars and other simple non-native values."""
    for cast in (float, int, str):
        try:
            return cast(value)
        except (TypeError, ValueError):
            continue
    return repr(value)


def mine(units: Iterable, shard: int, of: int) -> Iterator:
    """The units this shard is responsible for: every ``of``-th, from ``shard``."""
    for position, unit in enumerate(units):
        if position % max(1, int(of)) == int(shard):
            yield unit


def environment() -> Dict:
    """Software and hardware a result was produced on, stored with the result."""
    versions = {"python": sys.version.split()[0]}
    for module in ("numpy", "scipy", "pandas", "ortools", "matplotlib"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = None
    return {
        "versions": versions,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def keep_awake() -> Optional[int]:
    """
    Ask Windows not to sleep while this process is running.

    A long run on a laptop stops when the machine idles to sleep, and the
    wall-clock budgets of the solves in flight are then meaningless.  This is a
    request scoped to the process: it changes no setting, and it lapses when the
    process ends.  It does nothing on other systems.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes
        continuous, system_required = 0x80000000, 0x00000001
        return int(ctypes.windll.kernel32.SetThreadExecutionState(
            continuous | system_required))
    except Exception:
        return None
