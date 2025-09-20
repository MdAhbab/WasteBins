import heapq

def dijkstra(graph, source):
    # graph: {u: [(v, w), ...]}
    dist = {u: float('inf') for u in graph}
    prev = {u: None for u in graph}
    dist[source] = 0.0
    pq = [(0.0, source)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev

def reconstruct_path(prev, target):
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path

def compute_route(graph, source, targets=None):
    # If no targets, return distances tree; if targets, return combined shortest paths in visit order.
    dist, prev = dijkstra(graph, source)
    if not targets:
        return {'source': source, 'distances': dist, 'tree_prev': prev}
    route_path = []
    total_cost = 0.0
    current = source
    remaining = list(targets)
    while remaining:
        dist, prev = dijkstra(graph, current)
        # choose nearest target
        nearest = min(remaining, key=lambda t: dist.get(t, float('inf')))
        segment = reconstruct_path(prev, nearest)
        if route_path and segment and segment[0] == route_path[-1]:
            route_path.extend(segment[1:])
        else:
            route_path.extend(segment)
        total_cost += dist.get(nearest, 0.0)
        current = nearest
        remaining.remove(nearest)
    return {'path': route_path, 'total_cost': total_cost}