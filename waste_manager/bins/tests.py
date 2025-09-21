from django.test import TestCase
from bins.utils.dijkstra import compute_route
from bins.models import Node
from bins.views import _build_graph

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