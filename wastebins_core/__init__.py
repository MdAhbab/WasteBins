"""
wastebins_core
==============

Framework-independent implementation of every algorithm in the WasteBins
intelligent collection platform.  Nothing in this package imports Django, so the
*identical* code runs in three places:

  * the Django service (``waste_manager/bins/utils/*`` are thin adapters),
  * the reproducible experiment suite (``experiments/``),
  * the unit tests.

That removes the usual "the paper measured one implementation and the prototype
ships another" gap.

Modules
-------
geo             Haversine geometry and distance/travel-time matrices.
traffic         Contextual traffic friction (temporal, spatial, incident,
                weather) with a pluggable live-feed provider.
emissions       Physics-based, load- and speed-differentiated fuel/CO2 model
                covering cruise, congested stop-and-go, idle and compaction.
priority        Priority algebra with trust-weighted dynamic renormalisation.
faults          Sensor fault taxonomy and injectors (dropout, stuck-at, drift,
                calibration, bursty loss, weather-correlated, poisoning).
health          Runtime sensor validation producing per-channel trust weights.
features        Single source-of-truth feature engineering specification.
vrp             Capacitated VRP with time windows, shifts, depot return and
                prize-collecting skips, plus local search.
metaheuristics  Genetic algorithm, ant colony optimisation and a risk-penalised
                graph baseline sharing the VRP evaluator.
aging           Anti-starvation / service-equity term.
xai             Model-agnostic local attribution and global importance.
continual       Bounded online learner with drift detection and replay.
ledger          Hash-chained, Merkle-anchored tamper-evident audit log.
stats           Bootstrap confidence intervals, paired tests and effect sizes.
"""

__version__ = "2.0.0"

__all__ = [
    "geo", "traffic", "emissions", "priority", "faults", "health", "features",
    "vrp", "metaheuristics", "aging", "xai", "continual", "ledger", "stats",
]
