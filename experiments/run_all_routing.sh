#!/usr/bin/env bash
# Regenerate every routing result in the paper, in dependency order.
#
# The stages are separated because they have very different costs.  The
# benchmark comparison solves 25 snapshots once per policy; the equity rollout
# solves thousands of small problems and runs on a smaller network, which is
# stated in the paper rather than hidden.
set -u
cd "$(dirname "$0")/.."
LOG=experiments/results/run_all.log
: > "$LOG"

step () {
  echo "=== $* ===" | tee -a "$LOG"
  /usr/bin/time -f "  wall %E" python -u -m experiments.exp_fleet "$@" >> "$LOG" 2>&1 \
    || python -u -m experiments.exp_fleet "$@" >> "$LOG" 2>&1
  echo "  exit $?" | tee -a "$LOG"
}

# 1. Transfer study: 33 real containers on the Wyndham street graph.
step --study wyndham --only comparison --snapshots 25 --budget 3.0

# 2. Main study: 200 containers on the Dhaka street graph.
step --study dhaka --only comparison --snapshots 25 --budget 3.0

# 3. Distance-model ablation: the identical instances on great-circle distance.
step --study dhaka_gc --only comparison --snapshots 25 --budget 3.0

# 4. Sensitivity and the wait-bound rollout, on a 60-container network where
#    thousands of rollout solves are affordable.
step --study dhaka --bins 60 --vehicles 4 --only sensitivity --snapshots 25 \
     --budget 2.0 --out fleet_dhaka_equity.json
step --study dhaka --bins 60 --vehicles 4 --only equity --snapshots 25 \
     --budget 2.0 --rollout-budget 1.0 --out fleet_dhaka_equity.json

echo "ALL DONE" | tee -a "$LOG"
