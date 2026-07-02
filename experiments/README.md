# Reproducible experiments (SCS revision)

All quantitative results in the revised manuscript are generated here from a
**physically-grounded, seeded simulation** — no proprietary or unavailable data.
Everything is honest about being simulation and is fully reproducible.

## Requirements
Python 3.11+ with `numpy pandas scikit-learn scipy matplotlib pyarrow`.

## Run everything
```bash
python dataset.py       # build + cache the 30-bin, 180-day dataset (seed=42) -> results/{latent.pkl,features.parquet}
python exp_model.py     # Q1/Q2 + model bake-off, quantile uncertainty, calibration, circular-pitfall -> results/model.json
python exp_ablation.py  # Q5 sensor-failure ablation (renorm vs naive, 0-90% missing)                 -> results/ablation.json
python exp_routing.py   # Q3/Q4/Q9 routing baselines + real km/CO2/time-to-serve + alpha sweep         -> results/routing.json
python exp_equity.py    # Q8 equity / anti-starvation aging sweep                                      -> results/equity.json
python make_figures.py  # all figures -> results/figures/ and ../../Micro/images/
```

## What each file is
| File | Purpose |
|---|---|
| `sim.py` | Latent bin-fill + decomposition + sensor-observation simulator with **forward-looking** hazard labels (non-circular). |
| `dataset.py` | Builds and caches the dataset once for reuse. |
| `exp_model.py` | Demonstrates the circular-target pitfall ($R^2=0.99$), then the honest forward task; Ridge vs Random Forest vs HistGradientBoosting under a **temporal split** and **group-by-bin CV**; quantile intervals; probability calibration; latency. |
| `exp_ablation.py` | Dynamic Weight Renormalisation vs naive zero-fill under sensor loss (priority error + rank correlation). |
| `routing.py` / `exp_routing.py` | Haversine matrix, greedy warp, **2-opt/Or-opt**, **prize-collecting orienteering**; distance, CO2, time-to-serve, missed-overflow. |
| `exp_equity.py` | Aging/anti-starvation term; worst-case wait, Gini, hazard-response cost. |

## Headline numbers (from `results/*.json`)
- Circular target (reproduced pitfall): RF $R^2=0.99$ (self-consistency).
- Forward time-to-overflow: HistGB $R^2=0.83$, MAE 2.17 h (RF 0.81; Ridge 0.70; persistence ≈0).
- Group-by-bin CV: $R^2=0.81\pm0.05$. Hazard: calibrated ROC-AUC 0.98, Brier 0.033.
- Renormalisation @50% sensor loss: Spearman 0.56 vs 0.42 (naive).
- Routing: orienteering 13.45 km / 14.12 kg CO2 vs static 22.05 km / 23.15 kg (−39%), fastest time-to-critical, fewest missed overflows.
- Equity: worst-case wait 231.7 h → 66.0 h with aging (−72%), hazard-response cost +0.013 h.

## In the Django app
The same methods are wired into the backend:
- `manage.py train_forward` trains the forward-looking bundle (HistGB regressor + P10/P50/P90 quantiles + calibrated hazard classifier) on stored telemetry.
- `bins/utils/dijkstra.py` adds `two_opt_order`, `or_opt_order`, `orienteering_route`, `apply_aging`, and a `refine` flag on `compute_optimal_route`.
- `bins/utils/priority_calculator.py` uses the forward bundle for risk-aware priorities when available.
- Unit tests: `bins/tests.py::RoutingRefinementTests`.
