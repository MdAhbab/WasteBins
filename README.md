# Adaptive Priority-Weighted Dynamic Routing for IoT-Based Smart Waste Management

Research prototype and reproduction package for the accompanying manuscript. The
system ingests bin telemetry, judges whether that telemetry can be trusted,
forecasts which bins will overflow, and plans a multi-vehicle collection route
under real operating constraints — with every dispatch decision explainable and
recorded in a tamper-evident log.

**Status:** research prototype. All results below are reproducible from this
repository with the commands in [Reproducing the results](#reproducing-the-results).
Data is simulated plus public device telemetry; there has been no field
deployment. See [Limitations](#limitations), which is not a formality — read it
before quoting any number.

---

## Contents

- [What this is](#what-this-is)
- [Reproducing the results](#reproducing-the-results)
- [Architecture](#architecture)
- [The core package](#the-core-package)
- [Measured results](#measured-results)
- [API surface](#api-surface)
- [Management commands](#management-commands)
- [Tests](#tests)
- [Configuration](#configuration)
- [Limitations](#limitations)

---

## What this is

Five capabilities, each implemented in a framework-independent module that both
the web service and the experiment scripts import — so the code that produces
the published numbers is the code that ships:

**Sensing under failure.** Nine fault modes (dropout, stuck register, drift,
calibration error, noise burst, Gilbert–Elliott bursty loss, weather-correlated
failure, stealth-bounded poisoning) with a detector ensemble that separates a
genuine fault from legitimate signal change. Emptying a bin is a large, abrupt,
entirely normal level change; a detector that reacts to change alone fails
immediately. Detection feeds a per-channel trust weight that renormalises the
priority calculation rather than letting a bad sensor read as an empty bin.

**Forecasting, not self-consistency.** A gradient-boosting model predicts hours
until overflow and the probability of a hazard within six hours, from labels
derived strictly from each bin's *future* trajectory. Quantile heads give a
P10–P90 interval; the hazard head is isotonically calibrated on purged folds.

**Multi-vehicle routing under real constraints.** Prize-collecting CVRPTW:
capacity with on-board compaction, depot return including mid-shift tipping,
hard shift limits, per-bin service time windows, and waste-stream licensing per
vehicle. Compared against genetic (Prins route-first-cluster-second), Max–Min
Ant System, a risk-penalised graph heuristic, and OR-Tools guided local search —
all scored on one shared objective.

**Differentiated emissions.** A modal model rather than a flat factor: tractive
power against rolling and aerodynamic resistance, a Positive Kinetic Energy
term for stop-and-go, idle burn, and power-take-off compaction, with IPCC 2006
diesel factors and Euro-class multipliers. Traffic enters through a Bureau of
Public Roads volume-delay function.

**Governance.** KernelSHAP attributions with an exact efficiency constraint
(validated against TreeSHAP), a convex aging term with a *provable* worst-case
wait bound, and a SHA-256 hash-chained ledger with Merkle inclusion proofs.

---

## Reproducing the results

Python 3.11+ and Node 20+. Neither the database nor the trained model is in
version control, so both are built from scratch below — this is the whole
reproduction path, not a quickstart.

```bash
pip install -r waste_manager/requirements.txt
```

```bash
cd waste_manager && python manage.py migrate
```

Generate the simulated network — 20 bins, 20,160 readings, with Arrhenius
temperature-dependent decomposition, diurnal and weekly demand, Poisson dumping
bursts, and scheduled collections:

```bash
python manage.py seed_demo
```

Train the forward model (roughly one minute; writes to `model_store/`):

```bash
python manage.py train_forward
```

Run the service:

```bash
python manage.py runserver
```

Then the frontend, in a second terminal:

```bash
cd frontend && npm install && npm run dev
```

The dashboard is at `http://localhost:5173`. The map panel needs a Google Maps
browser key in `frontend/.env.local` as `VITE_GOOGLE_MAPS_API_KEY`; every other
page works without one, and the map panel says so rather than hanging.

### The experiment suite

Each script writes a JSON file to `experiments/results/`, which is the source of
truth for the manuscript's tables:

```bash
cd experiments && python eval_sensor_health.py
```

| Script | Produces | Answers |
|---|---|---|
| `eval_sensor_health.py` | `sensor_health.json` | Detection across all nine fault modes; false-positive rate on a clean fleet; priority error under each trust policy |
| `exp_fleet.py` | `routing.json` | All five solvers on one objective; constraint-violation audit |
| `exp_continual.py` | `continual.json` | Prequential error under seven drift scenarios; do-no-harm check; serving latency |
| `exp_realdata.py` | `realdata.json` | Validation against public device telemetry |

`exp_realdata.py` is the one script needing data that is not in the repository —
CSVs are gitignored, since a code repo is the wrong place to redistribute a
dataset. Fetch it first:

```bash
kaggle datasets download -d garystafford/environmental-sensor-data-132k -p experiments/realdata --unzip
```

That is "Environmental Sensor Telemetry Data" (Gary Stafford, CC0): three
physically separate ESP8266 nodes reporting CO, LPG, smoke, temperature,
humidity, light and motion at roughly one-second cadence over 8 days in July
2020 — 405,184 raw rows. The script also accepts `iot_telemetry_data.csv` in the
repository's parent directory, or a path as its first argument.

No public dataset carries bin *fill* level alongside gas, temperature and
humidity, so no real dataset can validate the whole pipeline end to end. What
this one contributes is the genuine noise, drift, dropout and device-to-device
heterogeneity of low-cost hardware on exactly the hazard modalities the system
depends on. Read the two findings in
[Limitations](#limitations) before citing it — they are not favourable.

---

## Architecture

```
wastebins_core/            framework-independent; imports no Django
  geo.py                   haversine, distance matrices, projections
  traffic.py               BPR volume-delay, corridors, incidents, live adapter
  emissions.py             modal fuel/CO2: tractive power, PKE, idle, PTO
  faults.py                nine-mode fault taxonomy and injectors
  health.py                detector ensemble, trust and reliability state
  priority.py              three trust policies for priority under failure
  aging.py                 convex aging with a provable wait bound; equity
  vrp.py                   prize-collecting CVRPTW: construct, evaluate, improve
  metaheuristics.py        GA, MMAS ant colony, risk graph, OR-Tools
  scenario.py              problem construction; static-sweep baseline
  features.py             31 features shared by training and serving
  hpo.py                   search spaces, purged splits, bake-off
  continual.py             bounded residual corrector, ADWIN2, Page-Hinkley
  xai.py                   KernelSHAP with exact efficiency, TreeSHAP, permutation
  ledger.py                hash chain, Merkle roots and inclusion proofs
  stats.py                 BCa bootstrap, Wilcoxon, Cliff's delta, Holm

waste_manager/             Django service
  bins/services/           telemetry, dispatch, models_registry, audit
  bins/api_platform.py     the v1 platform API
  bins/utils/ai/           training entry points

frontend/                  React 19 + Vite 8
experiments/               reproduction scripts and committed results
```

The rule that `wastebins_core` imports no Django is what closes the usual gap
where the paper measures one implementation and the prototype ships another.

---

## The core package

| Concern | Module | Notes |
|---|---|---|
| Travel time under congestion | `traffic.py` | BPR calibrated for saturated urban arterials; falls back to the synthetic provider when a live feed is unreachable |
| Fuel and CO₂ | `emissions.py` | Duty-cycle sensitive; `sanity_reference_factor` pins the model inside the published 1.5–3 mpg refuse-truck range |
| Fault injection | `faults.py` | `describe_taxonomy()` documents every mode; magnitudes are in each channel's own units |
| Detection and trust | `health.py` | Fleet-relative residuals; analytical redundancy predicts each channel from the others; reliability (data quality) is kept separate from trust (maintenance state) |
| Priority under failure | `priority.py` | `zero_fill` (naive), `renormalise`, `trust_weighted` |
| Fairness | `aging.py` | `worst_case_wait_bound` returns `inf` honestly when no guarantee exists rather than a misleading finite number |
| Routing | `vrp.py` | One `skip_cost` definition shared by construction and scoring, so the planner optimises the objective it is judged by |
| Continual learning | `continual.py` | Frozen base plus a bounded linear corrector, gated so it only affects predictions while it is measurably helping |

---

## Measured results

Regenerate everything with the commands above. The two blocks below were
verified in the current tree; the routing figures are the committed contents of
`experiments/results/routing.json`.

### Forecasting (20 bins, 19,200 rows: 14,420 train / 3,840 test)

Temporal hold-out with a 24 h purge gap matching the label horizon, and purged
expanding-window inner cross-validation.

| Metric | Pooled | Uncensored rows only |
|---|---|---|
| Time-to-overflow R² | 0.787 | **0.652** |
| Time-to-overflow MAE | 2.05 h | **3.28 h** |
| Mean-predictor baseline MAE | 6.37 h | — |
| Inner CV R² | 0.763 ± 0.027 | — |
| P10–P90 coverage | 0.831 (nominal 0.80) | mean width 5.14 h |
| Hazard ROC AUC | 0.987 | avg precision 0.942 |
| Hazard Brier | 0.024 (skill 0.787) | positive rate 0.131 |

**Read the second column.** `time_to_overflow` is right-censored at 24 h and
**66.3% of test labels are the cap** — a constant. The pooled R² is therefore
largely a score for recognising "not today", which is easy. The uncensored-row
figures are the honest measure of the forecasting task, and both are recorded in
the artefact metadata so the pooled number cannot be quoted alone.

### Sensor fault detection

| | Tuned seeds | Held-out (24 clean + 12 fault, unseen) |
|---|---|---|
| Clean-fleet false-positive rate | 0.000 | **0.004** (max 0.025) |
| Mean recall | 0.98 | **0.98** |
| Mean precision | 0.95 | **0.92** |

Quote the held-out column. Recall is 1.00 on eight of nine modes; bursty loss is
the weakest at 0.82, which is expected — a burst that lands mid-window leaves
less evidence than a persistent bias.

### Routing

Five solvers on one shared objective, lower is better: **proposed 279** < ant
colony 323 < genetic 324 < risk-penalised graph 346 < OR-Tools 421. Ant colony
and genetic are within one unit and should not be read as ordered. Zero
constraint violations (capacity, shift, time window, stream licensing,
double-service) on the live instance across all five solvers and on twelve
adversarial instances, checked by an independent re-simulation that does not
reuse the planner's own evaluator.

---

## API surface

Session-authenticated. `/api/v1/` is the current surface; the unversioned
`/api/` routes are retained for the original prototype's endpoints.

**Operations** — `nodes/`, `nodes/ensure/`, `nodes/<id>/history/`,
`readings/submit/`, `dashboard/`, `notifications/`

**Fleet** — `fleet/config/`, `fleet/vehicles/<id>/`, `fleet/plan/`,
`fleet/plans/`, `fleet/compare/`, `fleet/service/`, `fleet/equity/`

**Sensing** — `sensors/health/`, `sensors/faults/`, `sensors/inject/`

**Models** — `models/status/`, `models/continual/reset/`, `explain/<id>/`

**Sustainability** — `emissions/`, `traffic/`

**Assurance** — `audit/`, `audit/verify/`

`sensors/inject/` labels every row it writes, and its `DELETE` removes only
labelled synthetic rows — it cannot take the seeded network with it.

---

## Management commands

Run from `waste_manager/`.

| Command | Purpose |
|---|---|
| `seed_demo` | Build the simulated network and telemetry history |
| `train_forward` | Train the forward bundle; `--quick` for a small search budget |
| `verify_ledger` | Recompute the hash chain and report the first divergence |
| `check_system` | Database connectivity, model presence, data integrity |
| `load_sample_data` | Minimal fixture for the original prototype |

Training is deliberately an offline command. The request path performs only
bounded incremental updates — capped samples and milliseconds per update, set by
`ML_CONTINUAL_MAX_SAMPLES` and `ML_CONTINUAL_MAX_MS` — so using the application
never triggers a training workload.

---

## Tests

```bash
cd waste_manager && python manage.py test
```

Coverage is thin relative to the codebase and should be read as a smoke test,
not a safety net. `ForwardModelIntegrationTests` redirects the entire model
store to a temporary directory: it calls the real trainer, and Django isolates
the database but not the filesystem, so without that redirection running the
suite overwrites the deployed model with one trained on its own three-bin
fixture. `test_model_store_is_isolated_from_production` fails loudly if that
redirection ever breaks.

---

## Configuration

Copy `.env.example` to `.env` in the repository root. Every value is optional;
the defaults boot on SQLite with no configuration.

| Group | Keys |
|---|---|
| Django | `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`, `DJANGO_TIME_ZONE` |
| Database | `DB_ENGINE` (`sqlite`\|`mysql`\|`postgres`), `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` |
| ML serving | `ML_CONTINUAL_ENABLED`, `ML_CONTINUAL_MAX_SAMPLES`, `ML_CONTINUAL_MAX_MS`, `ML_ALLOW_INLINE_TRAINING` |
| Fleet | `FLEET_CAPACITY_KG`, `FLEET_SHIFT_MINUTES`, `FLEET_SERVICE_MINUTES`, `FLEET_AVG_SPEED_KMH`, `FLEET_DEPOT_LAT`, `FLEET_DEPOT_LNG` |
| Traffic | `TRAFFIC_PROVIDER` (`synthetic`\|`live`), `TRAFFIC_API_URL`, `TRAFFIC_API_KEY` |
| Audit | `AUDIT_LEDGER_ENABLED`, `AUDIT_BLOCK_SIZE`, `AUDIT_HMAC_KEY` |

The frontend reads `VITE_GOOGLE_MAPS_API_KEY` from `frontend/.env.local`. Vite
inlines any `VITE_`-prefixed variable into the client bundle, so that key is
public by construction: restrict it by HTTP referrer and to the Maps JavaScript
API in the Google Cloud console rather than treating it as a secret.

---

## Limitations

Stated plainly, because each one bounds a claim above.

**No field deployment.** Results come from simulation plus validation against
public device telemetry. The simulator encodes real mechanisms — temperature-
dependent decomposition, diurnal and weekly demand, congestion — but it is not
a municipality, and no claim here should be read as a field result.

**On real telemetry, the non-linear model does not earn its place.** In the
within-device hold-out, a plain logistic regression on the same features reaches
AUC 0.9862 against 0.9019 for the calibrated gradient boosting, and the latter's
Brier skill score over the base rate is only 0.049. The gradient boosting's
advantage in simulation comes from fill *dynamics* — accumulation rate,
acceleration, dwell — which no public dataset provides, so this comparison tests
the hazard modalities only. Stated as measured: on this data the simpler model
wins, and the reported baseline exists precisely so that is visible.

**Transfer to an unseen device is unreliable.** Leave-one-device-out is the
honest analogue of commissioning a newly installed bin, and it does not hold up:
mean AUC 0.7571 across the three devices, but the worst is **0.4550 — below
chance** — with a Brier score of 0.3943 on another. The three nodes sit in
physically different environments, so a model fitted on two of them does not
describe the third. The operational reading is that a new node needs its own
calibration period before its predictions carry weight, which is what the trust
and reliability layer is for; the wrong reading would be that the model
generalises across sites.

**Censored regression.** The regressor fits a squared loss against labels
right-censored at 24 h, which treats a censored label as observed and biases
long-horizon predictions downward. The honest metrics are reported on the
uncensored subset; a survival objective (AFT or Cox) with a concordance metric
is the correct fix and is not implemented.

**Compaction is applied to mass, not volume.** `effective_load = load_kg /
compaction_ratio` reduces mass by compacting, which is dimensionally wrong when
capacity is a mass limit — compaction changes volume. Reported utilisation was
corrected to match the constraint that actually binds, rather than redefining
the constraint silently, but the underlying model is still wrong and it spans
`vrp`, `scenario` and `dispatch`.

**Cross-sectional detection needs a fleet.** Peer and dispersion tests are
disabled below ten nodes, and the single-node drift test abstains entirely,
because one node cannot separate sensor drift from a change in the weather. On
small networks detection falls back to the within-node tests and recall drops.

**Route planning uses its full search budget.** Plan generation runs for its
configured budget (seconds, not milliseconds) rather than returning as fast as
possible. It is bounded and never exceeds the budget, but it is user-visible.

---

## Citation and contributors

Md Ahbab Hamid Khan, Ahnaf Atique, Khandokar Md. Rahat Hossain. Please cite the
accompanying manuscript; this repository is its reproduction package.
