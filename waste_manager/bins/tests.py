from django.test import TestCase
from bins.utils.dijkstra import compute_route

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