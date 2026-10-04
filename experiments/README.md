# Experiments

Every quantitative result of *Bounded Waiting Time in Sensor-Driven Waste
Collection* is produced here. Each solve is written once, as a JSON line under
`results/raw/`, and every table, figure and quoted number in the paper is a
function of those records. Nothing in the paper is typed by hand.

Container positions are real in both study areas and both street graphs are
compiled from OpenStreetMap. Demand is observed in Wyndham and simulated in
Dhaka, which publishes no fill data.

## Requirements

Python 3.11 or newer with `numpy pandas scipy scikit-learn matplotlib` and
`ortools`. No GPU is used. `exp_censoring.py` additionally needs `xgboost` and a
seeded database; nothing in the paper pipeline does.

## Running the paper's experiments

```bash
python -m experiments.run_queue --plan full --workers 5
```

The queue runs every study in order of importance, resumes after an
interruption, and refuses to start while another runner holds its lock.
Progress is in `results/logs/queue_status.json`; an empty file named `STOP` in
`results/logs/` stops it after the running jobs. The full plan takes about a day
on twelve logical cores.

Then, from the repository root:

```bash
python -m experiments.analyze
python -m experiments.make_tables
python -m experiments.fig_results
```

`analyze` writes `results/summary.json` (intervals, Wilcoxon tests with Holm
adjustment, Cliff's delta). `make_tables` writes every table and the macro file
`numbers.tex` into the manuscript's `tables/` folder, or into the folder given
with `--out`, such as `experiments/results/tables`. `fig_results` draws the
multi-cycle and weather figures.

## What each script does

| Script | Purpose |
|---|---|
| `build_roadnets.py` | Compiles both street graphs from Overpass into `data/roadnet/*.npz` |
| `dhaka_containers.py` | Extracts the mapped Dhaka waste facilities into `data/dhaka/` |
| `wyndham.py` | Parses the council feed into `data/wyndham/` |
| `weather_data.py` | Fetches the historical weather series into `data/weather/` |
| `instances.py` | The named container sets and snapshots |
| `policies.py` | The dispatch rules and planners under one interface, and canonical scoring |
| `store.py` | Append-only, resumable raw records, one file per shard |
| `exp_tune.py` | Planner settings on T160; the prize weight on T60 |
| `exp_compare.py` | Single-cycle comparisons at matched wall-clock time |
| `exp_rollout.py` | Forty dispatch cycles with evolving demand; the bound check |
| `exp_weather.py` | Adverse weather spells on the multi-cycle study |
| `exp_weather_demand.py` | Fill rate against weather in the Wyndham record |
| `exp_wait_bound.py` | The controlled price sweep and tightness instances |
| `exp_network.py` | Circuity, constant-factor error, asymmetry, round-trip costs |
| `exp_distance_model.py` | Plans on a constant detour factor, driven on the street graph |
| `exp_scale.py` | Construction and improvement time by instance size |
| `exp_emissions.py` | The emission model at stated settings |
| `exp_fleet.py` | Shared fleet, snapshot and travel-model code used by the studies above |
| `fig_bound.py`, `fig_results.py`, `figstyle.py` | Figures, from the result files |

`exp_model.py`, `exp_censoring.py`, `exp_continual.py`, `exp_realdata.py`,
`eval_sensor_health.py` and `eval_xai_agreement.py`, with `dataset.py` and
`sim.py`, evaluate the forecast, fault and explanation components of the
software. The paper does not use them.

## A note on reproducibility

The search budget is wall-clock, so a busier machine gives the metaheuristics
fewer iterations in the same time. The paper reports differences together with
their intervals across instances.

A study must run on one machine, together with any study it borrows reference
times from: `ablation` and `budget-*` take the insertion times of `main`. If a
run is interrupted on a new machine, move the shard files of its unfinished jobs
to `results/raw_partial/`, which the analysis does not read, and run them again
in full. The records here came from one machine; each
`results/raw/<study>.meta.json` names the machine of its study.

`build_roadnets.py` and `dhaka_containers.py` query Overpass live, so a rebuild
picks up map edits and will not reproduce the committed extracts byte for byte.
The fingerprint stored in each `.npz` identifies the extract the published
results were computed on.
