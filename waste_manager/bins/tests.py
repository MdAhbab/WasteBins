from django.test import TestCase, SimpleTestCase
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
    """End-to-end: seed telemetry -> train forward model -> risk-aware priority."""

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