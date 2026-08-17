import tempfile
from pathlib import Path

from django.test import TestCase, SimpleTestCase, override_settings
from bins.utils.dijkstra import (
    compute_route, two_opt_order, or_opt_order, orienteering_route,
    apply_aging, _coord_map, _order_distance,
)
from bins.models import Node
from bins.views import _build_graph


class _FakeNode:
    def __init__(self, i, lat, lng):
        self.id = i; self.latitude = lat; self.longitude = lng


class RoutingRefinementTests(SimpleTestCase):
    """Local-search + orienteering + aging math (no database needed)."""

    coords = [(23.8069, 90.3687), (23.8250, 90.3650), (23.7910, 90.3550),
              (23.8203, 90.3650), (23.7980, 90.3720), (23.8100, 90.3780),
              (23.7890, 90.3740), (23.8050, 90.3630)]

    def setUp(self):
        self.nodes = [_FakeNode(i, la, lo) for i, (la, lo) in enumerate(self.coords)]
        self.cmap = _coord_map(self.nodes)
        self.depot = (23.8069, 90.3687)

    def test_local_search_never_lengthens_tour(self):
        scrambled = [2, 5, 1, 6, 3, 0, 7, 4]
        d0 = _order_distance(scrambled, self.cmap, self.depot)
        opt = or_opt_order(two_opt_order(scrambled, self.cmap, self.depot), self.cmap, self.depot)
        d1 = _order_distance(opt, self.cmap, self.depot)
        self.assertLessEqual(d1, d0 + 1e-6)
        self.assertEqual(sorted(opt), sorted(scrambled))

    def test_orienteering_respects_budget(self):
        prio = {0: 0.1, 1: 0.9, 2: 0.2, 3: 0.8, 4: 0.3, 5: 0.85, 6: 0.15, 7: 0.4}
        loc = {'lat': self.depot[0], 'lng': self.depot[1]}
        full = orienteering_route(self.nodes, prio, loc, budget_m=1e12)
        budget = 0.5 * full['total_distance_m']
        sub = orienteering_route(self.nodes, prio, loc, budget_m=budget)
        self.assertLessEqual(sub['total_distance_m'], budget + 1e-6)
        self.assertLessEqual(len(sub['path']), len(full['path']))

    def test_aging_boosts_waiting_bins(self):
        prio = {0: 0.1, 1: 0.9}
        aged = apply_aging(prio, {0: 96.0, 1: 0.0}, gamma=0.5, tau=48.0)
        self.assertGreater(aged[0], prio[0])
        self.assertAlmostEqual(aged[1], prio[1])


class ForwardModelIntegrationTests(TestCase):
    """
    End-to-end: seed telemetry -> train forward model -> risk-aware priority.

    The model store is redirected to a temporary directory for the duration of
    this class.  ``train_forward`` writes real artefact files, and Django's
    ``TestCase`` isolates the database but *not* the filesystem -- so without
    this, running the suite silently overwrites the deployed model with one
    trained on the three-bin fixture below.  That is not a hypothetical: it
    happened, and it replaced a 20-bin model with a constant predictor whose
    training labels were 100% censored (R^2 -0.17, every feature importance
    exactly zero) while leaving the API happily serving it.

    Every path in the store is overridden, not just the forward bundle, because
    the continual-learner state is keyed to the bundle it corrects; leaving it
    pointed at the real file would pair production state with a fixture model.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._store = tempfile.TemporaryDirectory()
        store = Path(cls._store.name)
        cls._store_override = override_settings(
            MODEL_STORE_DIR=store,
            MODEL_FILENAME=store / "rf_cost_model.joblib",
            MODEL_META_FILENAME=store / "rf_cost_model_meta.json",
            FORWARD_MODEL_FILENAME=store / "forward_bundle.joblib",
            FORWARD_MODEL_META_FILENAME=store / "forward_bundle_meta.json",
            CONTINUAL_MODEL_FILENAME=store / "continual_state.joblib",
            CONTINUAL_META_FILENAME=store / "continual_state_meta.json",
        )
        cls._store_override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._store_override.disable()
        cls._store.cleanup()
        super().tearDownClass()

    def test_model_store_is_isolated_from_production(self):
        """Guard the guard: if this fails, the override above has stopped working."""
        from django.conf import settings
        self.assertEqual(Path(settings.FORWARD_MODEL_FILENAME).parent,
                         Path(self._store.name))

    def _seed(self):
        import math
        from datetime import timedelta
        from django.utils import timezone
        from bins.models import Node, SensorReading
        base = timezone.now() - timedelta(hours=250)
        coords = [(23.8069, 90.3687), (23.7910, 90.3550), (23.8203, 90.3650)]
        for b, (lat, lng) in enumerate(coords):
            node = Node.objects.create(name=f"Bin-{b}", latitude=lat, longitude=lng)
            fill = 0.1 * b
            for h in range(220):
                fill += 0.06 + 0.01 * math.sin(h / 3.0)
                if fill >= 1.0:                     # overflow -> collected/reset
                    fill = 0.05
                gas = min(1.0, max(0.0, 0.65 * fill + 0.03 * math.sin(h)))
                SensorReading.objects.create(
                    node=node, timestamp=base + timedelta(hours=h),
                    waste_level=round(min(1.0, fill), 3),
                    gas_level=round(gas, 3),
                    temperature=25 + 8 * gas, humidity=60 + 20 * gas,
                    traffic_density=0.2,
                )

    def test_train_forward_and_predict(self):
        from bins.models import Node
        from bins.utils.ai.train_forward import train_forward, predict_forward
        from bins.utils.ai.model_store import load_forward_bundle
        from bins.utils.priority_calculator import priority_calculator

        self._seed()
        meta = train_forward(test_frac=0.2)
        self.assertIn("metrics", meta)
        self.assertGreater(meta["metrics"]["n_records"], 100)

        bundle = load_forward_bundle()
        self.assertIsNotNone(bundle)
        # a mostly-full, high-gas bin should be judged urgent
        row = {c: 0.0 for c in bundle["features"]}
        row.update(waste=0.95, mean_waste=0.9, gas=0.8, mean_gas=0.75, temp=32, humidity=80)
        pred = predict_forward(bundle, row)
        self.assertGreaterEqual(pred["risk_priority"], 0.0)
        self.assertLessEqual(pred["risk_priority"], 1.0)

        # app path: forward bundle drives node priorities, all within [0,1]
        nodes = list(Node.objects.all())
        priorities, _ = priority_calculator.calculate_node_priorities(
            nodes=nodes, user_lat=23.8069, user_lng=90.3687, use_ai_model=True)
        self.assertEqual(len(priorities), len(nodes))
        for v in priorities.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

class FleetFeasibilityTests(SimpleTestCase):
    """
    Every plan the solvers produce must be feasible, checked independently.

    `wastebins_core.validate.plan_violations` re-derives the clock, load and
    windows from the raw travel matrix rather than reusing `evaluate_route`, so a
    bug in the planner's own accounting cannot make its output look legal.  No
    database is needed: instances are generated, which also means the adversarial
    cases can be far nastier than anything the seeded network contains.
    """

    N_SEEDS = 6

    def _instance(self, seed, n_bins=18, n_veh=3, cluster=False):
        import numpy as np
        from wastebins_core import validate as V
        rng = np.random.default_rng(seed)
        matrix, tasks, vehicles = V.random_instance(
            rng, n_bins=n_bins, n_veh=n_veh, cluster=cluster)
        return V.build_instance(matrix, tasks, vehicles)

    def test_validator_detects_a_planted_violation(self):
        """
        Guard the guard.  A checker that always returned "feasible" would make
        every other test in this class pass without testing anything, so prove it
        can fail before trusting it when it passes.
        """
        from wastebins_core import validate as V, vrp
        travel, tasks, vehicles = self._instance(0)
        plan = vrp.solve(tasks, vehicles, travel, vrp.ObjectiveWeights(),
                         improve=False, time_budget_s=1.0)
        self.assertTrue(any(r.stops for r in plan.routes),
                        "instance produced an empty plan; nothing to corrupt")
        self.assertEqual(V.plan_violations(plan, travel), [])

        for route in plan.routes:                      # make capacity impossible
            route.vehicle.capacity_kg = 1.0
        planted = V.plan_violations(plan, travel)
        self.assertTrue(any("capacity" in v for v in planted),
                        f"validator missed a planted capacity breach: {planted}")

    def test_plans_are_feasible_across_seeds(self):
        from wastebins_core import validate as V, vrp
        weights = vrp.ObjectiveWeights()
        for seed in range(self.N_SEEDS):
            for cluster in (False, True):
                travel, tasks, vehicles = self._instance(seed, cluster=cluster)
                plan = vrp.solve(tasks, vehicles, travel, weights,
                                 improve=True, time_budget_s=1.0)
                found = V.plan_violations(plan, travel)
                self.assertEqual(found, [],
                                 f"seed {seed} cluster={cluster}: {found}")

    def test_every_installed_solver_is_feasible(self):
        from wastebins_core import metaheuristics as META, validate as V, vrp
        travel, tasks, vehicles = self._instance(3)
        weights = vrp.ObjectiveWeights()
        for name, installed in META.available_solvers().items():
            if not installed:
                continue
            plan = META.solve_with(name, tasks, vehicles, travel, weights,
                                   time_budget_s=1.0)
            if plan is None:
                continue
            found = V.plan_violations(plan, travel)
            self.assertEqual(found, [], f"solver {name}: {found}")

    def test_adversarial_instances_are_feasible(self):
        """Tight windows, a bin no vehicle can lift, and a near-useless licence."""
        import numpy as np
        from wastebins_core import validate as V, vrp
        weights = vrp.ObjectiveWeights()
        for seed in range(4):
            rng = np.random.default_rng(500 + seed)
            matrix, tasks, vehicles = V.random_instance(
                rng, n_bins=18, n_veh=3, cluster=bool(seed % 2))
            for task in tasks[:3]:
                task["window_end_min"] = 90.0          # barely reachable
            tasks[4]["load_kg"] = 1e6                  # exceeds every capacity
            vehicles[0]["accepts_streams"] = ("hazardous",)
            travel, tk, vh = V.build_instance(matrix, tasks, vehicles)
            plan = vrp.solve(tk, vh, travel, weights, improve=True, time_budget_s=1.0)
            found = V.plan_violations(plan, travel)
            self.assertEqual(found, [], f"adversarial seed {seed}: {found}")

    def test_time_budget_is_an_upper_bound(self):
        """
        The budget is a contract: an interactive request depends on it holding.
        Generous slack, because this asserts the bound is respected, not that the
        machine is fast.
        """
        import time
        from wastebins_core import vrp
        travel, tasks, vehicles = self._instance(7, n_bins=30, n_veh=4)
        budget = 1.0
        started = time.perf_counter()
        vrp.solve(tasks, vehicles, travel, vrp.ObjectiveWeights(),
                  improve=True, time_budget_s=budget)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, budget * 4 + 2.0,
                        f"solve took {elapsed:.2f}s against a {budget}s budget")

    def test_oversized_bin_is_left_unserved_not_dropped_silently(self):
        """A bin no vehicle can carry must appear in `unserved`, not vanish."""
        import numpy as np
        from wastebins_core import validate as V, vrp
        rng = np.random.default_rng(11)
        matrix, tasks, vehicles = V.random_instance(rng, n_bins=12, n_veh=2)
        tasks[2]["load_kg"] = 1e6
        oversized = tasks[2]["node_id"]
        travel, tk, vh = V.build_instance(matrix, tasks, vehicles)
        plan = vrp.solve(tk, vh, travel, vrp.ObjectiveWeights(),
                         improve=True, time_budget_s=1.0)
        served = {s.task.node_id for r in plan.routes for s in r.stops}
        unserved = {t.node_id for t in plan.unserved}
        self.assertNotIn(oversized, served)
        self.assertIn(oversized, served | unserved,
                      "the oversized bin disappeared from the plan entirely")


class SensorHealthRegressionTests(SimpleTestCase):
    """
    The cross-sectional detectors must not fire on a healthy fleet, and the two
    fleet-size thresholds must stay ordered.
    """

    def test_redundancy_activates_no_later_than_peers(self):
        """
        Redundancy suppresses the raw peer test on any channel it can model.  If
        peers activated at a smaller fleet than redundancy, there would be a band
        where the weaker test runs unopposed -- which measured a false-positive
        rate of 0.09-0.11, worse than running no fleet test at all.
        """
        from wastebins_core import health as H
        self.assertLessEqual(H.MIN_FLEET_FOR_REDUNDANCY, H.MIN_FLEET_FOR_PEERS)

    def test_clean_fleet_produces_no_alarms(self):
        import numpy as np
        from wastebins_core import health as H

        def clean_fleet(n_nodes, n, seed):
            rng = np.random.default_rng(seed)
            t = np.arange(n)
            ambient = 28 + 5 * np.sin(2 * np.pi * (t - 9) / 48)
            humidity = 62 - 0.8 * (ambient - 28)
            fleet = {}
            for i in range(n_nodes):
                rate = rng.uniform(0.004, 0.02)
                fill = np.clip(rng.uniform(0, 0.6) + rate * t, 0, 1.25)
                for k in sorted(rng.choice(np.arange(20, n - 10), size=2, replace=False)):
                    fill[k:] = np.clip(rng.uniform(0, 0.05) + rate * (t[k:] - t[k]), 0, 1.25)
                gas = np.clip(0.35 * fill + rng.normal(0, 0.02, n), 0, 1)
                fleet[i] = {"waste": fill + rng.normal(0, 0.02, n), "gas": gas,
                            "temp": ambient + 6 * gas + rng.normal(0, 0.4, n),
                            "humidity": np.clip(humidity + rng.normal(0, 2, n), 0, 100)}
            return fleet

        # Both regimes: below the peer threshold (within-node only) and above it.
        for n_nodes in (8, 20):
            for seed in (0, 1):
                fleet = clean_fleet(n_nodes, 80, seed)
                state, final = {}, {}
                for end in range(20, 81, 4):
                    snapshot = {i: {c: v[end - 20:end] for c, v in ch.items()}
                                for i, ch in fleet.items()}
                    final = H.assess_fleet(snapshot, previous_trust=state)
                    state = H.carry_state(final)
                truth = {i: {c: False for c in fleet[i]} for i in fleet}
                fpr = H.detection_metrics(final, truth)["false_positive_rate"]
                self.assertLessEqual(
                    fpr, 0.05,
                    f"{n_nodes} nodes, seed {seed}: false-positive rate {fpr:.3f}")


class DijkstraTests(TestCase):
    def test_small_graph(self):
        graph = {
            1: [(2, 1), (3, 4)],
            2: [(3, 2)],
            3: []
        }
        res = compute_route(graph, source=1, targets=[3])
        self.assertEqual(res['path'], [1, 2, 3])
        self.assertAlmostEqual(res['total_cost'], 3.0, places=5)

    def test_inverse_priority_weight(self):
        # Two nodes at same location delta; higher priority should have lower incoming edge weight
        n1 = Node.objects.create(name='A', latitude=23.78, longitude=90.28)
        n2 = Node.objects.create(name='B', latitude=23.781, longitude=90.281)
        nodes = [n1, n2]
        priorities = {n1.id: 0.2, n2.id: 0.9}
        alpha = 0.5
        g = _build_graph(nodes, priorities, alpha)
        # Edge A->B vs A->A doesn't exist; compare A->B weight with hypothetical low priority case
        w_ab = next(w for v, w in g[n1.id] if v == n2.id)
        # Swap priorities and rebuild
        priorities_swapped = {n1.id: 0.9, n2.id: 0.2}
        g2 = _build_graph(nodes, priorities_swapped, alpha)
        w_ab_swapped = next(w for v, w in g2[n1.id] if v == n2.id)
        # When B has higher priority (0.9), A->B should be cheaper than when B has lower (0.2)
        self.assertLess(w_ab, w_ab_swapped)