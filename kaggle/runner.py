"""
Kaggle CPU session runner for the second-round studies.
=======================================================

One session runs one job.  The job is the JSON object assigned to ``JOB``
below, which `make_kernels.py` fills in for each session.  The code and its
inputs come from the dataset ``wastebins-code`` and are copied to
``/kaggle/working/wb`` so that every record lands in the session output.

Job kinds
---------
``pilot``
    Records the processor, re-runs the deterministic controlled experiments
    and compares them with the stored workstation result, then times one short
    rollout chain alone and with ``workers`` chains at once.
``runs``
    Runs ``slots``, one list of units per worker process.  A unit is
    ``[module, load, [networks...]]``: ``exp_rollout`` runs the ablation
    rollouts and ``exp_replay`` the observed-demand windows.  Each process owns
    whole networks, so every policy of a network runs in one process on one
    processor and takes its budgets from its own reference chain.

Every finished chain is one line in a JSON-lines file and is never lost; a
heartbeat file and a report are rewritten every minute.  Workers are stopped
at ``deadline_h`` so that the session ends inside the time it was given.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

JOB = {"name": "unset", "kind": "pilot", "workers": 4, "deadline_h": 5.6,
       "slots": [], "file_shard_base": 0}

WORK = pathlib.Path("/kaggle/working")
WB = WORK / "wb"
LOGS = WORK / "logs"
STARTED = time.time()
ORTOOLS = "9.15.6755"


def log(message):
    stamp = time.strftime("%H:%M:%S", time.gmtime())
    line = f"[{stamp} +{(time.time() - STARTED) / 60:6.1f} min] {message}"
    print(line, flush=True)
    with (WORK / "runner.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def find_source():
    for root in pathlib.Path("/kaggle/input").rglob("wastebins_core"):
        if root.is_dir() and (root.parent / "experiments").is_dir():
            return root.parent
    raise SystemExit("wastebins_core not found under /kaggle/input")


def setup():
    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    source = find_source()
    if not WB.exists():
        shutil.copytree(source, WB)
    for needed in ("wastebins_core/vrp.py", "experiments/exp_rollout.py",
                   "experiments/results/tuning.json", "data/roadnet/dhaka.npz",
                   "data/dhaka/containers.csv", "data/wyndham/fill_daily.csv",
                   "data/corpus/processed/daily.csv", "data/roadnet/wyndham.npz"):
        if not (WB / needed).exists():
            raise SystemExit(f"missing input {needed}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    f"ortools=={ORTOOLS}"], check=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c",
                          "import ortools, numpy, scipy, pandas, sys;"
                          "print(sys.version.split()[0], ortools.__version__,"
                          " numpy.__version__, scipy.__version__, pandas.__version__)"],
                         capture_output=True, text=True, env=env)
    log(f"versions python/ortools/numpy/scipy/pandas: {out.stdout.strip()}")
    if ORTOOLS not in out.stdout:
        raise SystemExit("pinned OR-Tools version not loaded: " + out.stdout + out.stderr)
    return env


def report(extra=None):
    cpu = ""
    try:
        cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    except OSError:
        pass
    raw = WB / "experiments" / "results" / "raw"
    counts = {}
    if raw.exists():
        for path in sorted(raw.glob("*.jsonl")):
            counts[path.name] = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    body = {"job": JOB, "elapsed_min": round((time.time() - STARTED) / 60, 1),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "lscpu": cpu, "records": counts}
    body.update(extra or {})
    (WORK / "report.json").write_text(json.dumps(body, indent=2))
    (WORK / "heartbeat.txt").write_text(f"{JOB['name']} {body['utc']} "
                                        f"{sum(counts.values())} records\n")


def heartbeat(stop):
    while not stop.is_set():
        try:
            report()
        except Exception as error:      # a heartbeat must never stop the job
            print("heartbeat error", error, flush=True)
        stop.wait(60)


def run_workers(env, commands):
    deadline = STARTED + 3600.0 * float(JOB["deadline_h"])
    procs = []
    for k, cmd in enumerate(commands):
        handle = (LOGS / f"w{k}.log").open("w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, cwd=WB, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT), handle))
        log(f"worker {k}: {' '.join(cmd[2:])}")
    while any(p.poll() is None for p, _ in procs):
        if time.time() > deadline:
            log("deadline reached; stopping workers, finished chains are kept")
            for p, _ in procs:
                if p.poll() is None:
                    p.terminate()
            break
        time.sleep(20)
    for p, handle in procs:
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
        handle.close()
    return [p.returncode for p, _ in procs]


def pilot(env):
    stored = json.loads((WB / "experiments/results/wait_bound.json").read_text())
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "experiments.exp_wait_bound"], cwd=WB,
                   env=env, check=True, stdout=(LOGS / "wait_bound.log").open("w"))
    fresh = json.loads((WB / "experiments/results/wait_bound.json").read_text())
    keys = ("price_sweep", "bounded_penalty", "n_price_exact", "n_tight", "n_breaches")
    same = {k: stored.get(k) == fresh.get(k) for k in keys}
    common = ("containers", "reserved_per_cycle", "bound_h", "worst_wait_at_service_h",
              "attained", "holds", "max_backlog", "min_reserved")
    same["tightness"] = ([{c: r[c] for c in common} for r in stored["tightness"]]
                         == [{c: r[c] for c in common} for r in fresh["tightness"]])
    log(f"controlled experiments re-run in {time.time() - t0:.0f} s; identical: {same}")
    timing = {}
    probe = ("import sys, time, json; sys.path.insert(0, '.');"
             "from experiments import exp_rollout as R;"
             "w = R.World(sys.argv[1], 'tight', n_cycles=10);"
             "t = time.time(); o = R.run_chain(w, 'insertion', 'full');"
             "print(json.dumps({'chain_s': time.time() - t,"
             " 'per_solve_s': o['summary']['elapsed_per_solve_s']}))")
    for parallel in (1, int(JOB["workers"])):
        nets = [f"R{k}" for k in range(parallel)]
        t0 = time.time()
        procs = [subprocess.Popen([sys.executable, "-c", probe, net], cwd=WB, env=env,
                                  stdout=subprocess.PIPE, text=True) for net in nets]
        outs = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
        timing[f"parallel_{parallel}"] = {"wall_s": round(time.time() - t0, 1),
                                          "chains": outs}
        log(f"{parallel} chain(s) of 10 cycles at once: {timing[f'parallel_{parallel}']}")
    return {"identical_controlled": same, "timing": timing}


def main():
    env = setup()
    stop = threading.Event()
    beat = threading.Thread(target=heartbeat, args=(stop,), daemon=True)
    beat.start()
    extra = {}
    try:
        if JOB["kind"] == "pilot":
            extra = pilot(env)
        elif JOB["kind"] == "runs":
            commands = []
            for k, units in enumerate(JOB["slots"]):
                shard = int(JOB["file_shard_base"]) + k
                parts = []
                for module, load, nets in units:
                    if module == "exp_rollout":
                        parts.append(f"{sys.executable} -m experiments.exp_rollout --ablation "
                                     f"--load {load} --networks {','.join(nets)} "
                                     f"--shard 0 --of 1 --file-shard {shard}")
                    elif module == "exp_replay":
                        parts.append(f"{sys.executable} -m experiments.exp_replay "
                                     f"--load {load} --windows {','.join(nets)} "
                                     f"--file-shard {shard}")
                    else:
                        raise SystemExit(f"unknown module {module}")
                # A unit that fails does not stop the units after it.
                commands.append(["bash", "-c", " ; ".join(parts)])
            extra = {"return_codes": run_workers(env, commands)}
        else:
            raise SystemExit(f"unknown job kind {JOB['kind']}")
    finally:
        stop.set()
        report(extra)
        log("done")


if __name__ == "__main__":
    main()
