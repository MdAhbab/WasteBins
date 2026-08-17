"""
Sensor fault detection: precision, recall and specificity across the taxonomy.
==============================================================================

Answers reviewer 1.4 ("only zero-value hardware failure was tested") and
reviewer 2.6 ("real-world failure patterns are not modelled") with a measured
detection table over all nine fault modes, plus the false-positive rate on a
completely clean fleet -- which is the number that decides whether an operator
would actually leave the alarms switched on.

Evaluation protocol
-------------------
Detection is evaluated **sequentially**, exactly as the service runs it: a
sliding window advances over the stream and each assessment inherits the trust
state produced by the previous one.  Scoring a single isolated window instead
would flatter the detector, because the trust mechanism is explicitly designed
to accumulate evidence across cycles.

A channel counts as detected when its status is anything other than ``ok``.
Ground truth is the injection mask, so recall and precision are measured rather
than asserted.

Run:  python eval_sensor_health.py
Out:  results/sensor_health.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wastebins_core import faults as F  # noqa: E402
from wastebins_core import health as H  # noqa: E402
from wastebins_core import priority as P  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

N_NODES = 20
N_STEPS = 140
WINDOW = 20
STRIDE = 4
VICTIMS = (0, 3, 7)

# Seeds the detector thresholds were developed against.  Anything measured on
# these is a fitting score, not a generalisation score, and must not be the
# number the manuscript quotes.
SEEDS = (0, 1, 2, 3)

# Seeds held out from every tuning decision.  Kept in the harness rather than in
# a throwaway script precisely so the held-out figures are reproducible from the
# repository: a detector ensemble with this many thresholds is exactly the kind
# of thing that quietly overfits its development fixtures, and a claim that it
# has not done so is worth nothing unless a reader can re-run it.
HOLDOUT_CLEAN_SEEDS = tuple(range(100, 124))     # 24 unseen clean fleets
HOLDOUT_FAULT_SEEDS = tuple(range(200, 212))     # 12 unseen fault instances


def clean_fleet(n_nodes: int = N_NODES, n: int = N_STEPS, seed: int = 0) -> Dict:
    """
    A healthy fleet with the structure that makes detection hard.

    Shared ambient dynamics (a diurnal temperature cycle driving humidity) plus
    per-bin fill trajectories with their own arrival rates and collection resets.
    Both are legitimate sources of large signal changes, so a detector that
    simply reacts to change will fail this fixture.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    ambient = 28 + 5 * np.sin(2 * np.pi * (t - 9) / 48)
    humidity = 62 - 0.8 * (ambient - 28)

    fleet: Dict[int, Dict[str, np.ndarray]] = {}
    for i in range(n_nodes):
        rate = rng.uniform(0.004, 0.02)
        fill = np.clip(rng.uniform(0, 0.6) + rate * t, 0, 1.25)
        for k in sorted(rng.choice(np.arange(20, n - 10), size=2, replace=False)):
            fill[k:] = np.clip(rng.uniform(0, 0.05) + rate * (t[k:] - t[k]), 0, 1.25)
        gas = np.clip(0.35 * fill + rng.normal(0, 0.02, n), 0, 1)
        fleet[i] = {
            "waste": fill + rng.normal(0, 0.02, n),
            "gas": gas,
            "temp": ambient + 6 * gas + rng.normal(0, 0.4, n),
            "humidity": np.clip(humidity + rng.normal(0, 2, n), 0, 100),
        }
    return fleet


def sequential_assess(fleet: Dict, n: int = N_STEPS, window: int = WINDOW,
                      stride: int = STRIDE):
    """
    Slide a window over the stream, carrying detector state forward.

    The whole recursive state is carried, not just the trust scalar: the
    confirmation counters that separate an isolated statistical alarm from a
    sustained fault live in it, and harvesting only ``trust`` would silently
    reset them on every cycle.  ``health.carry_state`` is the same call the
    Django service makes between assessments.
    """
    state: Dict = {}
    final: Dict = {}
    for end in range(window, n + 1, stride):
        snapshot = {i: {c: v[end - window:end] for c, v in ch.items()}
                    for i, ch in fleet.items()}
        final = H.assess_fleet(snapshot, previous_trust=state)
        state = H.carry_state(final)
    return final


def inject_mode(fleet: Dict, mode: str, channel: str = "gas", seed: int = 11):
    """Apply one fault mode to the victim nodes; returns (fleet, truth)."""
    corrupted = {i: {c: np.array(v, dtype=float) for c, v in ch.items()}
                 for i, ch in fleet.items()}
    truth = {i: {c: False for c in corrupted[i]} for i in corrupted}
    for victim in VICTIMS:
        spec = F.FaultSpec(
            mode=mode, rate=0.9, magnitude=0.35,
            start_frac=0.45, duration_frac=0.55,
            stealth=0.45, direction="amplify",
            seed=seed + victim, extra={"channel": channel},
        )
        values, mask = F.inject(corrupted[victim][channel], spec)
        corrupted[victim][channel] = values
        truth[victim][channel] = bool(mask.any())
    return corrupted, truth


def _clean_rates(seeds) -> List[float]:
    rates = []
    for seed in seeds:
        fleet = clean_fleet(seed=seed)
        truth = {i: {c: False for c in fleet[i]} for i in fleet}
        metrics = H.detection_metrics(sequential_assess(fleet), truth)
        rates.append(metrics["false_positive_rate"])
    return rates


def specificity_study() -> Dict:
    """
    False-positive rate on clean fleets -- the operator's tolerance test.

    Reported on the tuning seeds and, separately, on seeds never used for tuning.
    The held-out figure is the one to quote; the tuned figure is included only so
    the gap between them is visible, since that gap is the overfitting measure.
    """
    tuned = _clean_rates(SEEDS)
    held = _clean_rates(HOLDOUT_CLEAN_SEEDS)
    return {
        "seeds": list(SEEDS),
        "false_positive_rate_per_seed": [round(r, 4) for r in tuned],
        "false_positive_rate_mean": round(float(np.mean(tuned)), 4),
        "false_positive_rate_max": round(float(np.max(tuned)), 4),
        "holdout": {
            "seeds": list(HOLDOUT_CLEAN_SEEDS),
            "n_seeds": len(HOLDOUT_CLEAN_SEEDS),
            "false_positive_rate_mean": round(float(np.mean(held)), 4),
            "false_positive_rate_max": round(float(np.max(held)), 4),
            "seeds_with_any_false_positive": int(sum(1 for r in held if r > 0)),
            "note": "never used for any tuning decision; this is the figure to cite",
        },
    }


def detection_study(seeds=SEEDS) -> Dict:
    """Per-mode detection, averaged over seeds."""
    per_mode: Dict[str, List[Dict]] = {mode: [] for mode in F.FAULT_MODES}
    for seed in seeds:
        fleet = clean_fleet(seed=seed)
        for mode in F.FAULT_MODES:
            corrupted, truth = inject_mode(fleet, mode, seed=11 + seed * 7)
            per_mode[mode].append(
                H.detection_metrics(sequential_assess(corrupted), truth))

    summary = {}
    for mode, runs in per_mode.items():
        summary[mode] = {
            "recall": round(float(np.mean([r["recall"] for r in runs])), 4),
            "precision": round(float(np.mean([r["precision"] for r in runs])), 4),
            "f1": round(float(np.mean([r["f1"] for r in runs])), 4),
            "false_positive_rate": round(
                float(np.mean([r["false_positive_rate"] for r in runs])), 4),
            "description": F.describe_taxonomy()[mode],
        }
    summary["_mean"] = {
        key: round(float(np.mean([summary[m][key] for m in F.FAULT_MODES])), 4)
        for key in ("recall", "precision", "f1", "false_positive_rate")
    }
    return summary


def renormalisation_study() -> Dict:
    """
    The downstream question: does trust-weighting actually protect the priority?

    Compares the three policies on identical corrupted inputs, measuring the
    error against the priority that would have been computed from the clean
    signal.  This is what the sensing work is *for* -- a detector that flags a
    fault but still lets it corrupt the dispatch decision has achieved nothing.
    """
    results = {}
    for mode in F.FAULT_MODES:
        errors = {policy: [] for policy in P.POLICIES}
        for seed in SEEDS:
            fleet = clean_fleet(seed=seed)
            corrupted, _ = inject_mode(fleet, mode, seed=11 + seed * 7)
            assessments = sequential_assess(corrupted)

            for node_id in fleet:
                clean_values = {
                    "waste_level": float(fleet[node_id]["waste"][-1]),
                    "gas_level": float(fleet[node_id]["gas"][-1]),
                    "temperature": float(fleet[node_id]["temp"][-1]),
                    "humidity": float(fleet[node_id]["humidity"][-1]),
                }
                reference = P.compute_priority(clean_values,
                                               policy=P.POLICY_RENORMALISE).score

                observed, trust = {}, {}
                for field, core in (("waste_level", "waste"), ("gas_level", "gas"),
                                    ("temperature", "temp"), ("humidity", "humidity")):
                    assessment = assessments[node_id].get(core)
                    if assessment is None:
                        continue
                    observed[field] = assessment.value
                    trust[field] = assessment.trust

                for policy in P.POLICIES:
                    if policy == P.POLICY_ZERO_FILL:
                        # The naive baseline sees the raw corrupted value and
                        # substitutes zero only for a genuinely absent sample.
                        raw = {}
                        for field, core in (("waste_level", "waste"), ("gas_level", "gas"),
                                            ("temperature", "temp"), ("humidity", "humidity")):
                            a = assessments[node_id].get(core)
                            raw[field] = None if a is None else a.raw_value
                        score = P.compute_priority(raw, policy=policy).score
                    else:
                        score = P.compute_priority(observed, trust=trust,
                                                   policy=policy).score
                    errors[policy].append(abs(score - reference))

        results[mode] = {
            policy: round(float(np.mean(values)), 5) if values else None
            for policy, values in errors.items()
        }
    results["_mean"] = {
        policy: round(float(np.mean([results[m][policy] for m in F.FAULT_MODES
                                     if results[m][policy] is not None])), 5)
        for policy in P.POLICIES
    }
    return results


def main() -> None:
    print("Sensor fault detection evaluation")
    print("=" * 72)

    specificity = specificity_study()
    print(f"\nClean fleet false-positive rate: "
          f"{specificity['false_positive_rate_mean']:.3f} "
          f"(max {specificity['false_positive_rate_max']:.3f} over "
          f"{len(SEEDS)} seeds)")

    print(f"Held-out false-positive rate: "
          f"{specificity['holdout']['false_positive_rate_mean']:.4f} "
          f"(max {specificity['holdout']['false_positive_rate_max']:.4f} over "
          f"{specificity['holdout']['n_seeds']} unseen seeds, "
          f"{specificity['holdout']['seeds_with_any_false_positive']} with any) "
          f"<- cite this one")

    detection = detection_study()
    print(f"\n{'fault mode':<22}{'recall':>9}{'precision':>11}{'F1':>8}{'FPR':>8}")
    print("-" * 58)
    for mode in F.FAULT_MODES:
        row = detection[mode]
        print(f"{mode:<22}{row['recall']:>9.2f}{row['precision']:>11.2f}"
              f"{row['f1']:>8.2f}{row['false_positive_rate']:>8.3f}")
    mean = detection["_mean"]
    print("-" * 58)
    print(f"{'MEAN':<22}{mean['recall']:>9.2f}{mean['precision']:>11.2f}"
          f"{mean['f1']:>8.2f}{mean['false_positive_rate']:>8.3f}")

    detection_holdout = detection_study(HOLDOUT_FAULT_SEEDS)
    hm = detection_holdout["_mean"]
    print(f"{'MEAN (held-out)':<22}{hm['recall']:>9.2f}{hm['precision']:>11.2f}"
          f"{hm['f1']:>8.2f}{hm['false_positive_rate']:>8.3f}   <- cite this row")
    weakest = min((m for m in F.FAULT_MODES),
                  key=lambda m: detection_holdout[m]["recall"])
    print(f"  weakest held-out mode: {weakest} "
          f"(recall {detection_holdout[weakest]['recall']:.2f})")

    renorm = renormalisation_study()
    print(f"\nPriority error under corruption (lower is better)")
    print(f"{'fault mode':<22}{'zero_fill':>12}{'renormalise':>14}{'trust_weighted':>16}")
    print("-" * 66)
    for mode in F.FAULT_MODES:
        row = renorm[mode]
        print(f"{mode:<22}{row['zero_fill']:>12.4f}{row['renormalise']:>14.4f}"
              f"{row['trust_weighted']:>16.4f}")
    print("-" * 66)
    m = renorm["_mean"]
    print(f"{'MEAN':<22}{m['zero_fill']:>12.4f}{m['renormalise']:>14.4f}"
          f"{m['trust_weighted']:>16.4f}")

    payload = {
        "protocol": {
            "n_nodes": N_NODES, "n_steps": N_STEPS, "window": WINDOW,
            "stride": STRIDE, "victims": list(VICTIMS), "seeds": list(SEEDS),
            "holdout_clean_seeds": list(HOLDOUT_CLEAN_SEEDS),
            "holdout_fault_seeds": list(HOLDOUT_FAULT_SEEDS),
            "which_figures_to_cite": "the holdout blocks; the tuning-seed figures "
                                     "are reported only to expose the gap between them",
            "evaluation": "sequential sliding window with trust carried forward, "
                          "matching how the service runs",
            "detected_when": "channel status is not 'ok'",
        },
        "specificity": specificity,
        "detection": detection,
        "detection_holdout": detection_holdout,
        "renormalisation_priority_error": renorm,
        "thresholds": {
            "drift_z": H.DRIFT_Z_THRESHOLD,
            "drift_self_z": H.DRIFT_SELF_Z_THRESHOLD,
            "peer_z": H.PEER_Z_THRESHOLD,
            "dispersion_z": H.DISPERSION_Z_THRESHOLD,
            "redundancy_z": H.REDUNDANCY_Z_THRESHOLD,
            "redundancy_min_r2": H.REDUNDANCY_MIN_R2,
            "spike_z": H.SPIKE_Z_THRESHOLD,
            "min_fleet_for_peers": H.MIN_FLEET_FOR_PEERS,
            "trust_suspect": H.TRUST_SUSPECT,
            "trust_degraded": H.TRUST_DEGRADED,
            "trust_failed": H.TRUST_FAILED,
            "anomaly_deadband": H.ANOMALY_DEADBAND,
            "unconfirmed_floor": H.UNCONFIRMED_FLOOR,
            "reliability_recovery": round(H.RELIABILITY_RECOVERY, 5),
            "reliability_suspect": H.RELIABILITY_SUSPECT,
        },
    }
    (RESULTS / "sensor_health.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {RESULTS / 'sensor_health.json'}")


if __name__ == "__main__":
    main()
