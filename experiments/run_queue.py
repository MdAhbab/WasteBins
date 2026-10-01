"""
Run the experiments as a queue of resumable jobs.
=================================================

Every experiment script is resumable and writes its own shard, so the whole
study is a list of independent jobs.  This runner keeps a fixed number of them
going at once, in the order they are listed, and records what happened to each.

The number of workers matters for the results and not only for the wait.  The
budgeted planners are given wall-clock time, so a job must have a processor core
to itself for the whole of that time.  Five workers on six physical cores leave
one core for the system.  More workers than cores would hand some planner less
computation than its budget says.

A job may name groups it has to wait for.  The ablation, for example, borrows
its reference solves from the main comparison and starts when that has finished.

Stopping and resuming.  Create a file called ``STOP`` in the log folder and no
new job is started; jobs already running finish.  Starting the runner again
skips nothing by itself, but every job skips the units it has already stored, so
a finished job returns at once.

One runner at a time.  A second runner would write to the same shard files and
share the processor cores the budgets assume, so the runner holds a lock file in
the log folder and refuses to start while the process that wrote it is alive.

Run:  python -m experiments.run_queue --plan full
      python -m experiments.run_queue --plan full --list
Out:  results/logs/<job>.log, results/logs/queue_status.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments import store as ST               # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS = ST.RESULTS / "logs"
LOCK = LOGS / "queue.lock"


def _alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    import ctypes
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return code.value == 259          # STILL_ACTIVE


@dataclass
class Job:
    name: str
    module: str
    args: List[str]
    group: str
    after: Tuple[str, ...] = ()
    state: str = "pending"            # pending, running, done, failed
    returncode: int = 0
    started: str = ""
    ended: str = ""
    minutes: float = 0.0
    extra: Dict = field(default_factory=dict)


def _shards(jobs: List[Job], group: str, module: str, base: List[str], count: int,
            after: Tuple[str, ...]) -> None:
    for k in range(count):
        jobs.append(Job(f"{group}-{k:02d}", module,
                        base + ["--shard", str(k), "--of", str(count)], group, after))


def plan(name: str) -> List[Job]:
    """
    The jobs of a plan, most important first.

    If time runs out, what has finished is the part the paper needs most: the
    planner comparison, then the rollouts that carry the waiting-time claim,
    then the weather study, then the supporting studies.
    """
    jobs: List[Job] = []
    compare, rollout = "experiments.exp_compare", "experiments.exp_rollout"
    tuned = ("select",)

    jobs.append(Job("tune-select", "experiments.exp_tune", ["--select"], "select"))
    if name == "select":
        return jobs

    # 65 units of about twelve minutes; thirteen jobs of five units each.
    _shards(jobs, "main", compare, ["--study", "main"], 13, tuned)
    _shards(jobs, "wyndham", compare, ["--study", "wyndham"], 1, tuned)
    # One network per job.  The moderate and tight loads run on sixteen
    # networks: with eight, the smallest two-sided p-value of a paired test is
    # 2/256, and a Holm adjustment over several metrics can never fall below
    # 0.05.  Shard k runs network Rk and writes its own file.
    sixteen = ["--networks", ",".join(f"R{k}" for k in range(16))]
    _shards(jobs, "rollout-moderate", rollout, ["--load", "moderate"] + sixteen, 16, tuned)
    for scenario in ("rain", "heat", "storm", "onset"):
        _shards(jobs, f"weather-{scenario}", "experiments.exp_weather",
                ["--scenario", scenario], 8, tuned)
    _shards(jobs, "rollout-tight", rollout, ["--load", "tight"] + sixteen, 16, tuned)
    _shards(jobs, "rollout-hazard", rollout, ["--load", "hazard"], 8, tuned)
    _shards(jobs, "ablation", compare, ["--study", "ablation"], 5, ("main",))
    _shards(jobs, "distance", "experiments.exp_distance_model",
            ["--snapshots", "15"], 5, tuned)
    _shards(jobs, "scale", "experiments.exp_scale", [], 5, tuned)
    for tag in ("co2x0", "co2x3", "timex0.5", "timex2", "overflowx0.5",
                "overflowx2", "lambdax0.5", "lambdax2"):
        _shards(jobs, f"weights-{tag}", compare, ["--study", f"weights-{tag}"], 2, tuned)
    for tag in ("4", "9", "18", "werribee", "point_cook"):
        _shards(jobs, f"depot-{tag}", compare, ["--study", f"depot-{tag}"], 2, tuned)
    for factor in ("0.1", "0.25", "2"):
        _shards(jobs, f"budget-{factor}", compare, ["--study", f"budget-{factor}"],
                2, ("main",))
    if name == "full":
        return jobs
    wanted = set(name.split(","))
    return [j for j in jobs if j.group in wanted or j.group == "select"]


class Queue:
    def __init__(self, jobs: List[Job], workers: int):
        self.jobs = jobs
        self.workers = workers
        self.lock = threading.Lock()
        self.started = time.time()

    def _write_status(self) -> None:
        counts: Dict[str, int] = {}
        for job in self.jobs:
            counts[job.state] = counts.get(job.state, 0) + 1
        payload = {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hours_running": round((time.time() - self.started) / 3600.0, 2),
            "workers": self.workers, "counts": counts,
            "jobs": [{"name": j.name, "state": j.state, "returncode": j.returncode,
                      "started": j.started, "ended": j.ended,
                      "minutes": round(j.minutes, 1)} for j in self.jobs],
        }
        tmp = LOGS / "queue_status.tmp"
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(LOGS / "queue_status.json")

    def _next(self):
        """The first pending job whose groups have all finished, or None."""
        open_groups = {j.group for j in self.jobs if j.state in ("pending", "running")}
        failed_groups = {j.group for j in self.jobs if j.state == "failed"}
        for job in self.jobs:
            if job.state != "pending":
                continue
            if any(g in failed_groups for g in job.after):
                job.state, job.ended = "failed", time.strftime("%H:%M:%S")
                job.extra["reason"] = "a group it waits for failed"
                continue
            if not any(g in open_groups for g in job.after):
                return job
        return None

    def _work(self) -> None:
        while True:
            with self.lock:
                if (LOGS / "STOP").exists():
                    return
                job = self._next()
                if job is None:
                    if not any(j.state == "running" for j in self.jobs):
                        self._write_status()
                        return
                else:
                    job.state, job.started = "running", time.strftime("%H:%M:%S")
                    self._write_status()
            if job is None:
                time.sleep(15.0)          # something is running that others wait for
                continue
            t0 = time.time()
            with (LOGS / f"{job.name}.log").open("a", encoding="utf-8") as log:
                log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"{job.module} {' '.join(job.args)}\n")
                log.flush()
                code = subprocess.call(
                    [sys.executable, "-B", "-m", job.module] + job.args,
                    cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            with self.lock:
                job.returncode = code
                job.minutes = (time.time() - t0) / 60.0
                job.state = "done" if code == 0 else "failed"
                job.ended = time.strftime("%H:%M:%S")
                self._write_status()

    def run(self) -> int:
        LOGS.mkdir(parents=True, exist_ok=True)
        self._write_status()
        threads = [threading.Thread(target=self._work, daemon=True)
                   for _ in range(self.workers)]
        for thread in threads:
            thread.start()
            time.sleep(2.0)               # stagger the starts
        for thread in threads:
            thread.join()
        self._write_status()
        failed = [j.name for j in self.jobs if j.state == "failed"]
        (LOGS / "queue_finished.txt").write_text(
            time.strftime("%Y-%m-%d %H:%M:%S") + "\n"
            + ("failed: " + ", ".join(failed) if failed else "all jobs finished") + "\n")
        return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="full",
                        help="'full', 'select', or a comma-separated list of groups")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    jobs = plan(args.plan)
    if args.list:
        for job in jobs:
            print(f"{job.name:<28}{job.module:<34}{' '.join(job.args)}"
                  + (f"   after {', '.join(job.after)}" if job.after else ""))
        print(f"{len(jobs)} jobs")
        return 0
    LOGS.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        holder = LOCK.read_text().split()
        if holder and holder[0].isdigit() and _alive(int(holder[0])):
            print(f"another runner is active (pid {holder[0]}, since "
                  f"{' '.join(holder[1:])}); not starting")
            return 2
    LOCK.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    try:
        ST.keep_awake()
        finished = LOGS / "queue_finished.txt"
        if finished.exists():
            finished.unlink()
        return Queue(jobs, args.workers).run()
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
