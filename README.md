# No Container Waits Forever

Research prototype and reproduction package for *Bounded Waiting Time in
Sensor-Driven Waste Collection: A Reserved-Head Dispatch Rule for
Prize-Collecting Vehicle Routing*.

The system ingests container telemetry, judges whether that telemetry can be
trusted, forecasts which containers will overflow, and plans a multi-vehicle
collection round on a real street graph under real operating constraints. The
paper's claim concerns the dispatch rule: overdue containers form a queue
ordered by wait, the first `r` are reserved in every cycle together with a plan
that serves them, and every container is then collected within
`Δ·ceil(τ/Δ) + (ceil(M/r) - 1)·Δ` hours, with `M ≤ n` known before operation.
The bound holds for any planner placed under the rule. A fixed timetable also
bounds the wait, but only by ignoring fill; under this rule hazardous and nearly
full containers still come first, and the experiments measure overflow and
hazard response next to the wait to show what the bound costs. The forecasting, sensor
fault and explanation components below are part of the software and are not
used in the paper's experiments.

**Status:** research prototype. Every result below regenerates from this
repository with the commands in
[Reproducing the results](#reproducing-the-results).

Container positions are real in both study areas and the street graphs are real.
Demand is observed in Wyndham and simulated in Dhaka, which has no published fill
data. There has been no field deployment. See [Limitations](#limitations), which
is not a formality: read it before quoting any number.

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

Six capabilities, each implemented in a framework-independent module that both
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

**Real street networks, not a detour factor.** Distances are shortest paths on
the drivable OpenStreetMap graph, so one-way restrictions apply and the matrix is
asymmetric. Each leg carries its own uncongested speed, taken from the arcs its
shortest path uses. Two study areas are compiled and committed: Dhaka, at 134,437
nodes and 272,725 arcs, and Wyndham, at 44,180 and 77,543. Container positions
are real too: 431 waste facilities OpenStreetMap records for Dhaka, with a mapped
transfer station as the depot, and the 33 Wyndham City Council publishes with
three years of daily fill readings.

**Multi-vehicle routing under real constraints.** Prize-collecting CVRPTW:
capacity with on-board compaction, depot return including mid-shift tipping,
hard shift limits, per-container service time windows, and waste-stream licensing
per vehicle. Compared against a genetic algorithm (Prins
route-first-cluster-second), Max-Min Ant System, a priority-warped graph
heuristic, OR-Tools guided local search, a static sweep and a fill threshold, all
scored on one shared objective.

**Differentiated emissions.** A modal model rather than a flat factor: tractive
power against rolling and aerodynamic resistance, a Positive Kinetic Energy
term for stop-and-go, idle burn, and power-take-off compaction, with IPCC 2006
diesel factors and Euro-class multipliers. Traffic enters through a Bureau of
Public Roads volume-delay function.

**Weather as an operating condition.** Rain class, standing water, heat stress
and UV index map to travel speed, a speed limit, service time and gas readings,
with each rule taken from a published measurement. The response of fill rate to
weather is estimated on the Wyndham record. The live input is the Google Maps
Platform Weather API (key in `.env`, readings held in memory for at most one
hour, never written to disk); archived ERA5 weather comes from Open-Meteo.

**Governance.** KernelSHAP attributions with an exact efficiency constraint
(validated against TreeSHAP), a reserved-head dispatch rule with a worst-case
wait bound that holds for any planner, and a SHA-256 hash-chained ledger with
Merkle inclusion proofs.

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

### The paper's experiments

Every experiment writes one JSON record per solve or per policy chain to
`experiments/results/raw/<study>.sNN.jsonl`, and skips any unit already stored,
so an interrupted run resumes where it stopped. A queue runner starts them in
order of importance on five workers, so that each wall-clock budget has a
processor core to itself:

```bash
python -m experiments.run_queue --plan full --workers 5
```

The tables and every number quoted in the paper are then written from the raw
records, never typed by hand:

```bash
python -m experiments.analyze
```

```bash
python -m experiments.make_tables --out experiments/results/tables
```

Without `--out`, the tables go to the manuscript folder `../Paper/tables`, which
is not part of this repository.

**Machines.** Set `--workers` to the number of physical cores minus one. A
budgeted planner receives the wall-clock time that the insertion planner took on
the same instance, so a study and any study it borrows its reference times from
(`ablation` and `budget-*` borrow from `main`) must run on one machine. The
records in this repository came from two machines: a six-core Windows
workstation (tuning, `main`, `wyndham`, `ablation`, `budget-*`,
`rollout-moderate`, `weather-rain`) and a four-processor Linux machine (every
other study). Each `results/raw/<study>.meta.json` records the machine, the
package versions and the start time of its study.

| Module | Study | Answers |
|---|---|---|
| `exp_wait_bound.py` | controlled | Price sweep (Proposition 3) and tightness instances (Proposition 1) |
| `exp_tune.py` | `tune-*` | Planner settings on T160; the prize weight checked on T60 |
| `exp_compare.py` | `main`, `wyndham`, `ablation`, `weights-*`, `depot-*`, `budget-*` | Single-cycle planner and rule comparisons at matched wall-clock time |
| `exp_rollout.py` | `rollout-moderate`, `-tight`, `-hazard` | Forty dispatch cycles with evolving demand; per-container waits and the bound check |
| `exp_weather.py` | `weather-rain`, `-storm`, `-heat`, `-onset` | Six-day adverse spells; plans made blind, protected, on current conditions or on the forecast |
| `exp_weather_demand.py` | `weather_demand.json` | Fill rate against weather in the Wyndham record |
| `exp_distance_model.py` | `distance` | Plans made on a constant detour factor, driven on the street graph |
| `exp_network.py` | `network.json` | Circuity, constant-factor error, asymmetry and round-trip costs |
| `exp_scale.py` | `scale` | Construction and improvement time at seven instance sizes |
| `exp_emissions.py` | `emissions.json` | The emission model at stated settings, beside in-use measurements |

Named container sets (primary sample, tuning sets, rollout networks) are defined
in `experiments/instances.py`, so "tuned on" and "evaluated on" are statements
about named sets.

### System evaluation scripts

These evaluate the software components that the paper does not use:

```bash
cd experiments && python eval_sensor_health.py
```

| Script | Produces | Answers |
|---|---|---|
| `eval_sensor_health.py` | `sensor_health.json` | Detection across all nine fault modes; false-positive rate on a clean fleet; priority error under each trust policy |
| `eval_xai_agreement.py` | `xai_agreement.json` | Like-for-like agreement between the sampled and exact Shapley estimates, against the sampler's own seed-to-seed noise floor |
| `exp_continual.py` | `continual.json` | Prequential error under seven drift scenarios; do-no-harm check; serving latency |
| `exp_realdata.py` | `realdata.json` | Validation against public device telemetry, including leave-one-device-out transfer |
| `exp_censoring.py` | printed | Does a survival objective beat the censored squared loss? Measured: no |

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
| Travel time under congestion | `traffic.py` | BPR calibrated for saturated urban arterials; falls back to the synthetic provider when a live feed is unreachable, and also when a live feed that measures only the present is asked about a departure time further than `TRAFFIC_LIVE_VALIDITY_H` away (see below) |
| Fuel and CO₂ | `emissions.py` | Duty-cycle sensitive; `sanity_reference_factor` pins the model inside the published 1.5–3 mpg refuse-truck range |
| Fault injection | `faults.py` | `describe_taxonomy()` documents every mode; magnitudes are in each channel's own units |
| Detection and trust | `health.py` | Fleet-relative residuals; analytical redundancy predicts each channel from the others; reliability (data quality) is kept separate from trust (maintenance state) |
| Priority under failure | `priority.py` | `zero_fill` (naive), `renormalise`, `trust_weighted` |
| Fairness | `aging.py` | `worst_case_wait_bound` returns `inf` honestly when no guarantee exists rather than a misleading finite number |
| Routing | `vrp.py` | One `skip_cost` definition shared by construction and scoring, so the planner optimises the objective it is judged by |
| Continual learning | `continual.py` | Frozen base plus a bounded linear corrector, gated so it only affects predictions while it is measurably helping |

---

## Measured results

Regenerate everything with the commands above. The forecasting and fault
blocks below were verified in the current tree; the routing results are
generated from the raw records, as described under Routing.

### Forecasting (20 bins, 19,200 rows: 14,420 train / 3,840 test)

Temporal hold-out with a 24 h purge gap matching the label horizon, and purged
expanding-window inner cross-validation.

| Metric | Pooled | Uncensored rows only |
|---|---|---|
| Time-to-overflow R² | 0.801 | **0.680** |
| Time-to-overflow MAE | 2.44 h | **3.20 h** |
| Mean-predictor baseline MAE | 7.50 h | — |
| Inner CV R² | 0.757 ± 0.067 | — |
| P10–P90 coverage | 0.831 (nominal 0.80) | mean width 7.02 h |
| Hazard ROC AUC | 0.982 | avg precision 0.918 |
| Hazard Brier | 0.036 (skill 0.734) | — |

**Read the second column.** `time_to_overflow` is right-censored at 24 h and
**59.9% of test labels are the cap** — a constant. The pooled R² is therefore
largely a score for recognising "not today", which is easy. The uncensored-row
figures (1,540 of 3,840 test rows) are the honest measure of the forecasting
task, and both are recorded in the artefact metadata so the pooled number cannot
be quoted alone.

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

The routing results of the paper are written by `experiments/make_tables.py`
from the raw records into the manuscript's `tables/` folder; this file does not
repeat them. Every planner receives the same wall-clock time on every instance,
OR-Tools is given stream licences, overflow deadlines and the tipping time, and
the genetic algorithm and ant colony decode with the option to skip. Under these
conditions OR-Tools produces lower-cost single-cycle plans than the in-house
insertion planner on the tuning instances, which is why the paper evaluates the
dispatch rule under both. Every plan is checked against capacity, shift, time
window, stream licence and double service by a re-simulation that does not reuse
the planner's own evaluator.

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

**Breaking change to the ledger format.** Two defects were fixed in the entry
hash: the actor field sat outside the hash, and fields were concatenated without
length prefixes, so two different records could produce one hash. Both are now
inside a length-prefixed hash. This invalidates every entry written before the
change, and `verify_ledger` will report a divergence at the first such entry.
That is the correct outcome, because those old hashes did not commit to what
their entries claimed. Re-seed with `seed_demo` to rebuild a valid chain, or keep
the old entries and treat the divergence point as the format boundary. Do not
"fix" it by relaxing the verifier.
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
| Traffic | `TRAFFIC_PROVIDER` (`synthetic`\|`live`), `TRAFFIC_API_URL`, `TRAFFIC_API_KEY`, `TRAFFIC_LIVE_VALIDITY_H` |
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

**Censored regression, with the bias measured.** The served regressor fits a
squared loss against labels right-censored at 24 h, which treats a censored label
as an observed value and biases long-horizon predictions downward. The size of
that bias is now measured rather than described. Fitting the same estimator on
uncensored rows only, which is the second half of a two-part model where the
classifier decides whether an overflow happens at all:

| | pooled (served) | conditional |
|---|---|---|
| MAE on uncensored rows | 3.20 h | **2.23 h** |
| R² on uncensored rows | 0.680 | **0.834** |

So censoring costs about **0.97 h of mean absolute error** and 0.15 of R². A
concordance index is also reported, at **0.908**: it scores only whether the model
orders pairs of bins correctly and excludes pairs whose earlier time is censored,
so unlike R² it needs no uncensored label and is the ranking figure to quote.

**A survival objective was tried and is worse.** The textbook remedy for
censoring is an accelerated failure time model, which receives the interval
`[t, ∞)` for a censored row rather than the point `t`. Run through XGBoost's
`survival:aft` across three distributions and three scales (`experiments/exp_censoring.py`):

| model | C-index | MAE (uncensored) | R² (uncensored) |
|---|---|---|---|
| pooled squared loss (served) | **0.8995** | 3.271 | 0.6466 |
| conditional squared loss | 0.8681 | **2.393** | **0.7974** |
| best AFT (normal, scale 0.5) | 0.8518 | 3.451 | 0.5530 |

Every AFT variant ranked worse *and* timed worse. The likely reason, offered as a
hypothesis: censoring here is administrative and Type I, with every record cut at
the same known 24 h horizon because that is where the label stops being computed.
That uniformity is a fact of construction rather than an assumption. A row nearer
than 24 h to the end of its record was not watched for a full day, so
`forward_labels` censors it at its own shorter follow-up and flags it, and the
trainer drops those rows; every row that reaches the model therefore carries a
censoring time of exactly 24 h.
Accelerated failure time models assume a parametric survival distribution and
target censoring that varies between subjects, so the parametric assumption costs
more than the correct censoring treatment gains.

The deployed path therefore serves the pooled model, which ranks best, and
ranking is what the planner consumes. The conditional model is reported as the
unbiased timing estimator for rows where timing is defined. Switching the served
path to the two-part form remains open, since it would change the meaning of the
served number and with it the dispatch deadline logic and the operator display.

**A live traffic feed cannot cost a future departure, and is not asked to.** The
synthetic congestion surface is a function of the clock, so a route planned for
tomorrow's morning peak is costed at that peak. The commercial flow endpoints the
live adapter speaks to are not: they publish a measurement of the present and
expose no departure-time parameter. Extrapolating that reading into a forecast
would make the planner's departure-time sensitivity an artefact of the adapter
rather than a property of the network, so the adapter does not. A live reading is
used only for instants within `TRAFFIC_LIVE_VALIDITY_H` (default 0.5 h) of the
wall clock; a departure time beyond that is costed on the synthetic surface, the
substitution is counted in `out_of_window_fallbacks`, and `describe()` reports
both the count and which regime applied, so any run can be audited for how much
of its costing came from measurement and how much from the model. Where an
endpoint genuinely does accept a departure time, a `{time}` or `{epoch}`
placeholder in `TRAFFIC_API_URL` turns that on and the requested instant is sent
to the API. The operational reading is that live traffic improves *present*
costing, and that horizon planning remains on the modelled surface.

**Compaction is applied to mass, not volume.** `effective_load = load_kg /
compaction_ratio` reduces mass by compacting, which is dimensionally wrong when
capacity is a mass limit — compaction changes volume. Reported utilisation was
corrected to match the constraint that actually binds, rather than redefining
the constraint silently, but the underlying model is still wrong and it spans
`vrp`, `scenario` and `dispatch`.

**The wait bound rests on the reservation, and on one feasibility condition.**
Overdue containers form a queue ordered by wait, ties broken by container id. In
every cycle the dispatcher reserves the first `r` containers of that queue,
certifies them by inserting them into empty routes, gives them a skip penalty no
route can exceed, and rebuilds from the certificate any plan that drops one
(`vrp.certify_heads`, `vrp.restore_heads`). The bound

    W = Δ · ceil(τ / Δ) + (ceil(M / r) - 1) · Δ,    M ≤ n

follows from the queue order alone (`aging.queue_wait_bound`). It needs no price
and no insertion cost, holds under ties and for any planner, and is attained on
the star instances of `exp_wait_bound.py`. The condition is that every container
can be served alone by some vehicle within the shift; a container that cannot is
reported and skipped by the certificate rather than blocking the queue. The
certificate is sound and not complete, so fewer than `r` heads may be reserved in
a cycle, and the rollouts record the smallest number reserved.

The skip penalty plays a different part. A penalty priced on the bounded score
`w/(τ+w)` has a ceiling of `λ·μ = 180`, so a container whose round trip costs
more, about 33 km out at the defaults, is skipped at every wait. A penalty that
grows as `w/(2τ)` makes a dedicated trip worth taking after
`2·τ·C/(λ·μ)` hours when a vehicle is free, but gives no bound when every vehicle
is busy. The reservation gives the bound with either penalty; the unbounded
penalty keeps realised waits far below it.

Scarcity must be applied through demand or the fleet and **not** by a shift
shorter than a container's window start, which makes that container unservable
by any policy. The rollout raises an error in that case.

**Additive attribution is a lossy summary of this model.** The Shapley values
satisfy efficiency exactly — baseline plus contributions equals the prediction, and
the dashboard shows both numbers so it can be checked — but the *fidelity* of the
additive form, the R² of the additive approximation against the model's actual
local behaviour, is only about 0.20. That is not undersampling: it is flat across
500 to 8,000 coalitions (0.225, 0.204, 0.206, 0.196, 0.191), drifting slightly
down as the estimate stabilises. The gradient-boosted model is strongly
interacting locally, and no additive decomposition can represent it faithfully.
The honest claim is that the attributions are a valid Shapley decomposition and a
partial account of the model, not a full one; `fidelity_r2` is reported on every
explanation rather than hidden, and exact TreeSHAP is available for audit.

The 1,000-coalition default is chosen because more does not improve the *fidelity
of the additive form*. It does improve *estimator stability*, and the two must not
be confused. Measured against interventional TreeSHAP on a like-for-like
comparison, going from 1,000 to 8,000 coalitions halves the mean attribution error
(0.38 h to 0.14 h) and lifts top-3 driver agreement from 0.55 to 0.83. An operator
who needs a stable *ranking of drivers*, rather than a stable total, should raise
the budget. See `experiments/eval_xai_agreement.py`.

**Stream heterogeneity was investigated and stratification rejected.** Bins carry
different waste streams, and gas generation scales with organic content (0.85
organic, 0.55 general, 0.20 recyclable), so a recyclable bin sits systematically
below a fleet of general waste. The concern was that the cross-sectional tests,
which compare each bin against the fleet, would read that stable offset as a
fault.

Measured on the seeded 20-bin network with no fault injected anywhere, running
the detectors sequentially over 35 cycles as the service does, the false-positive
rate is a mean of 0.86 channels of 80 (0.011) with a worst cycle of 3 of 80
(0.038). That is inside the 0.05 tolerance, so the heterogeneity does not in fact
break the detector. An earlier figure of 7 of 80 was an artefact of repeatedly
calling the refresh endpoint, which accumulates trust state across calls; a
single assessment flags none.

Giving each well-represented stream its own intercept in the redundancy
regression was implemented and tested. It made detection **worse**, raising the
mean from 0.86 to 2.31 channels flagged, because on a 20-node fleet the extra
parameters cost more in degrees of freedom and leverage than the stream offset
costs in bias. The change was reverted. It may be worth revisiting on a network
large enough to support per-stream models.

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
