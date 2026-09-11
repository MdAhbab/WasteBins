#!/usr/bin/env bash
# Regenerate every routing result in the paper, in dependency order.
#
# The stages are separated because their costs differ by an order of magnitude.
# The benchmark comparison solves 25 snapshots once per policy on the full
# network; the equity rollout solves several thousand small problems and runs on
# a smaller one, which the paper states rather than hides.
#
# Container positions are real in both cities and are read from the committed
# tables under data/, so nothing here touches the network.
set -u
cd "$(dirname "$0")/.."
LOG=experiments/results/run_all.log
: > "$LOG"

step () {
  echo "=== $* ===" | tee -a "$LOG"
  python -u -m experiments.exp_fleet "$@" >> "$LOG" 2>&1
  echo "  exit $?" | tee -a "$LOG"
}

# 1. Transfer study: the real Wyndham container positions. Cheapest, so it runs
#    first and surfaces any breakage before the long runs start.
step --study wyndham --only comparison --snapshots 25 --budget 3.0

# 2. Main study: 160 of the waste containers OpenStreetMap records for Dhaka.
step --study dhaka --only comparison --snapshots 25 --budget 3.0

# 3. Distance-model ablation: the identical instances on great-circle distance.
step --study dhaka_gc --only comparison --snapshots 25 --budget 3.0

# 4. Sensitivity and the wait-bound rollout, on a 60-container subset where
#    several thousand rollout solves are affordable.
step --study dhaka --bins 60 --vehicles 4 --only sensitivity --snapshots 25 \
     --budget 2.0 --out fleet_dhaka_equity.json
step --study dhaka --bins 60 --vehicles 4 --only equity --snapshots 25 \
     --budget 2.0 --rollout-budget 1.0 --out fleet_dhaka_equity.json

# 5. Street-graph measurements and the scalability curve.
echo "=== exp_network ===" | tee -a "$LOG"
python -u -m experiments.exp_network >> "$LOG" 2>&1
echo "  exit $?" | tee -a "$LOG"

echo "=== exp_scale ===" | tee -a "$LOG"
python -u -m experiments.exp_scale --sizes 40,80,160,320 --snapshots 3 >> "$LOG" 2>&1
echo "  exit $?" | tee -a "$LOG"

# 6. Figures, from the results just written.
echo "=== figures ===" | tee -a "$LOG"
python -u -m experiments.fig_network >> "$LOG" 2>&1
python -u -m experiments.fig_bound >> "$LOG" 2>&1
python -u -m experiments.fig_scale >> "$LOG" 2>&1
echo "  exit $?" | tee -a "$LOG"

echo "ALL DONE" | tee -a "$LOG"
