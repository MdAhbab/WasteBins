"""
Write one Kaggle kernel folder per session of the second-round studies.
=======================================================================

Each folder holds ``run.py`` (the runner with its job filled in) and
``kernel-metadata.json``.  Push one with
``kaggle kernels push -p <folder>`` from the parent of the folder.

Run:  python kaggle/make_kernels.py <out-dir> pilot
      python kaggle/make_kernels.py <out-dir> wave1
"""
import json
import pathlib
import sys

USER = "mdahbabhamidkhan"
DATASET = f"{USER}/wastebins-code"
HERE = pathlib.Path(__file__).resolve().parent
NETWORKS = [f"R{k}" for k in range(16)]


def sessions(wave):
    if wave == "pilot":
        return [{"name": "wb-pilot", "kind": "pilot", "workers": 4, "deadline_h": 1.0,
                 "slots": [], "file_shard_base": 0}]
    if wave == "wave1":
        # Units and their estimated minutes on one Kaggle core, from the pilot:
        # about 5.8 minutes per 40-cycle chain on a 60-container network and
        # about 2.3 on a Wyndham window.  The hazard load runs eleven chains,
        # the others ten, a window eight.  The moderate ablation runs on R0 to
        # R7, the first cut of the plan.
        units = ([("exp_rollout", "hazard", [n], 64) for n in NETWORKS]
                 + [("exp_rollout", "tight", [n], 58) for n in NETWORKS]
                 + [("exp_rollout", "moderate", [n], 58) for n in NETWORKS[:8]]
                 + [("exp_replay", load, [f"W{k:02d}"], 18)
                    for load in ("moderate", "tight") for k in range(16)])
        n_sessions, per_session = 5, 2
        slots = [[] for _ in range(n_sessions * per_session)]
        load = [0.0] * len(slots)
        # Longest units first, each to the least loaded slot.
        for unit in sorted(units, key=lambda u: -u[3]):
            k = min(range(len(slots)), key=lambda j: load[j])
            slots[k].append(list(unit[:3]))
            load[k] += unit[3]
        print("estimated minutes per slot:", [round(x) for x in load])
        return [{"name": f"wb-run-s{s}", "kind": "runs", "workers": per_session,
                 "deadline_h": 5.75, "file_shard_base": 10 * (s + 1),
                 "slots": slots[s * per_session:(s + 1) * per_session]}
                for s in range(n_sessions)]
    if wave == "wave2":
        # Wave 1 stopped one worker at its deadline three chains into window
        # W10 at the tight load.  The window is run again whole, so that every
        # policy of it takes its budgets from a reference chain of the same
        # session; the three partial chains of wave 1 are not merged.
        return [{"name": "wb-run-w2", "kind": "runs", "workers": 1, "deadline_h": 1.5,
                 "file_shard_base": 60, "slots": [[["exp_replay", "tight", ["W10"]]]]}]
    raise SystemExit(f"unknown wave {wave}")


def main():
    out_dir, wave = pathlib.Path(sys.argv[1]), sys.argv[2]
    template = (HERE / "runner.py").read_text(encoding="utf-8")
    marker = 'JOB = {"name": "unset", "kind": "pilot", "workers": 4, "deadline_h": 5.6,\n' \
             '       "slots": [], "file_shard_base": 0}'
    assert marker in template
    for job in sessions(wave):
        folder = out_dir / job["name"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run.py").write_text(
            template.replace(marker, "JOB = " + json.dumps(job)), encoding="utf-8")
        meta = {"id": f"{USER}/{job['name']}", "title": job["name"],
                "code_file": "run.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": False, "enable_tpu": False,
                "enable_internet": True, "dataset_sources": [DATASET],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(folder, [len(s) for s in job["slots"]], "units per slot")


if __name__ == "__main__":
    main()
