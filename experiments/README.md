# Experiments

Every quantitative result in *No Container Waits Forever* is produced here and
written to `results/` as JSON. Nothing in the paper is typed by hand: the tables
are generated from these files and a checker re-derives each quoted figure from
them.

Container positions are real in both study areas and both street graphs are
compiled from OpenStreetMap. Demand is observed in Wyndham and simulated in
Dhaka, which publishes no fill data. The forecast and fault layers are evaluated
on a simulated network, because no public dataset pairs container telemetry with
labelled sensor faults.

## Requirements

Python 3.11 or newer with `numpy pandas scikit-learn scipy matplotlib`.
No GPU is used anywhere. `exp_censoring.py` additionally needs `xgboost` and a
seeded database; nothing else does.

## Regenerating the routing results

```bash
bash run_all_routing.sh
```

That runs the stages below in dependency order and logs to
`results/run_all.log`. It takes several hours on twelve cores. Individual
stages:

| Command | Writes | Roughly |
|---|---|---|
| `python -m experiments.exp_fleet --study wyndham --only comparison` | `fleet_wyndham.json` | 5 min |
| `python -m experiments.exp_fleet --study dhaka --only comparison` | `fleet_dhaka.json` | 70 min |
| `python -m experiments.exp_fleet --study dhaka_gc --only comparison` | `fleet_dhaka_gc.json` | 65 min |
| `python -m experiments.exp_fleet --study dhaka --only sensitivity` | `fleet_dhaka_equity.json` | 65 min |
| `python -m experiments.exp_fleet --study dhaka --only equity --resample` | `fleet_dhaka_equity.json` | 140 min |
| `python -m experiments.exp_distance_model` | `distance_model.json` | 110 min |
| `python -m experiments.exp_network` | `network.json` | 40 min |
| `python -m experiments.exp_scale` | `scale.json` | 40 min |
| `python -m experiments.exp_wait_bound` | `wait_bound.json` | seconds |

The stages are independent apart from the two that share
`fleet_dhaka_equity.json`, so most of them can run at once. `exp_network`
measures properties of the street graph rather than of any plan, so it does not
need rerunning when the planner changes.

## What each script does

| Script | Purpose |
|---|---|
| `build_roadnets.py` | Compiles both street graphs from Overpass into `data/roadnet/*.npz` |
| `dhaka_containers.py` | Extracts the mapped Dhaka waste facilities into `data/dhaka/` |
| `wyndham.py` | Parses the council feed into `data/wyndham/` |
| `exp_fleet.py` | The routing comparison, the sensitivity sweeps and the wait-bound rollout |
| `exp_distance_model.py` | Plans on a constant detour factor, then measures that plan on the street graph |
| `exp_network.py` | Circuity, direction asymmetry, snap distance, the longest leg |
| `exp_scale.py` | Cost and quality at four instance sizes |
| `exp_wait_bound.py` | The controlled single-container sweep that tests the bound exactly |
| `exp_model.py` | Forecast bake-off under a temporal split and group-by-bin CV |
| `exp_censoring.py` | Whether a survival objective beats the censored squared loss. It does not |
| `eval_sensor_health.py` | The nine-mode fault taxonomy, on held-out seeds |
| `dataset.py`, `sim.py` | The simulator behind the forecast and fault evaluations |
| `fig_*.py`, `make_figures.py` | Figures, from the result files just written |

`exp_ablation.py`, `exp_continual.py`, `exp_equity.py`, `exp_realdata.py`,
`exp_routing.py`, `routing.py` and `eval_xai_agreement.py` belong to an earlier
version of this work. They still run, and their results are still in `results/`,
but nothing in the current paper is drawn from them.

## Checking the numbers

The paper's build directory carries the checker:

```bash
cd ../../TITS_Submission
python make_tables.py && python check_numbers.py && python build.py all
```

`check_numbers.py` re-derives every quoted figure from `results/` and reports
anything the prose no longer matches. It also refuses to pass when the wait
bound is violated in the rollout, because a paper that claims a guarantee must
not build while its own experiment breaks it.

## A note on reproducibility

The search budget is wall-clock, so a busier machine gives the metaheuristics
fewer iterations in the same three seconds and their distances move by a few
tenths of a percent between runs. Every conclusion in the paper rests on
differences an order of magnitude larger.

`build_roadnets.py` and `dhaka_containers.py` query Overpass live, so a rebuild
picks up map edits and will not reproduce the committed extracts byte for byte.
The fingerprint stored in each `.npz` identifies the extract the published
results were computed on.
