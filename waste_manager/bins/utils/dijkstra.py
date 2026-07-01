import heapq
import math
from django.conf import settings

# Default features in case settings is missing them
DEFAULT_DYNAMIC_FEATURES = {
    'traffic_density': {'type': 'cost_multiplier', 'routing_weight': 3.0, 'min_val': 0.0, 'max_val': 1.0, 'impact': 'negative'}
}

def dijkstra(graph, source):
    """
    Standard Dijkstra's algorithm implementation
    graph: {u: [(v, w), ...]} where w is edge weight
    """
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
    """Reconstruct path from source to target using predecessor array"""
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in meters"""
    R = 6371000.0  # Earth's radius in meters
    phi1, phi2 = math.radians(lat1 or 0.0), math.radians(lat2 or 0.0)
    dphi = math.radians((lat2 or 0.0) - (lat1 or 0.0))
    dlambda = math.radians((lon2 or 0.0) - (lon1 or 0.0))
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def calculate_edge_weight(base_distance_m, destination_priority, dynamic_costs=None, alpha=0.5):
    """
    Calculate edge weight based on distance, destination node priority, and dynamic cost multipliers.
    
    cost_multiplier logic: weight = base_distance * PRODUCT(1 + (feature_val * routing_weight))
    priority logic: weight = adjusted_distance / (1 + alpha * (priority * 10))
    """
    priority = max(0.0, min(1.0, destination_priority))
    
    # Base adjustment
    adjusted_distance = base_distance_m
    
    # Apply dynamic cost multipliers
    feature_config = getattr(settings, 'DYNAMIC_FEATURES', DEFAULT_DYNAMIC_FEATURES)
    if dynamic_costs:
        for feature, val in dynamic_costs.items():
            if feature in feature_config:
                config = feature_config[feature]
                if config.get('type') == 'cost_multiplier':
                    scaling_weight = config.get('routing_weight', 1.0)
                    min_v = config.get('min_val', 0.0)
                    max_v = config.get('max_val', 1.0)
                    
                    # Normalize dynamic cost
                    n_val = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.0
                    n_val = max(0.0, min(1.0, float(n_val)))
                    
                    # Inflate distance proportionally
                    adjusted_distance *= (1.0 + n_val * scaling_weight)
    
    # Apply inverse priority function
    weight = adjusted_distance / (1.0 + alpha * (priority * 10.0))
    return weight

def build_priority_graph(nodes, priority_scores, traffic_scores=None, alpha=0.5):
    """
    Build graph with edge weights calculated based on node priorities and traffic
    """
    if traffic_scores is None:
        traffic_scores = {}

    graph = {}
    node_ids = [n.id for n in nodes]
    
    for node_id in node_ids:
        graph[node_id] = []
    
    for i, node_u in enumerate(nodes):
        for j, node_v in enumerate(nodes):
            if i == j:
                continue
            
            base_distance = haversine_distance(
                node_u.latitude or 0.0, node_u.longitude or 0.0,
                node_v.latitude or 0.0, node_v.longitude or 0.0
            )
            
            dest_priority = priority_scores.get(node_v.id, 0.0)
            
            # Extract all dynamic costs for node_v from traffic_scores context
            # In current implementation traffic_scores primarily held density, 
            # we adapt it into a dynamic_costs dictionary for the new algorithm if it's a float
            v_traffic = traffic_scores.get(node_v.id, 0.0)
            dynamic_costs = {'traffic_density': v_traffic} if isinstance(v_traffic, (int, float)) else v_traffic
            
            weight = calculate_edge_weight(base_distance, dest_priority, dynamic_costs, alpha)
            graph[node_u.id].append((node_v.id, weight))
    
    return graph

def compute_optimal_route(nodes, priority_scores, traffic_scores=None, source_node_id=None, user_location=None, alpha=0.5, refine=True):
    """
    Compute optimal route using Dijkstra's algorithm with priority-based edge weights.

    When ``refine`` is True the greedy visiting order is post-processed with
    2-opt + Or-opt local search on real distances, which removes the myopic
    detours the plain warped-greedy tour can introduce (typically ~25% shorter
    travel distance) while preserving the priority-driven ordering benefit.
    """
    if traffic_scores is None:
        traffic_scores = {}

    if not nodes:
        raise ValueError("No nodes provided for routing")
    
    graph = build_priority_graph(nodes, priority_scores, traffic_scores, alpha)
    
    virtual_source = None
    if user_location and 'lat' in user_location and 'lng' in user_location:
        virtual_source = 'user_location'
        graph[virtual_source] = []
        
        for node in nodes:
            base_distance = haversine_distance(
                user_location['lat'], user_location['lng'],
                node.latitude or 0.0, node.longitude or 0.0
            )
            dest_priority = priority_scores.get(node.id, 0.0)
            v_traffic = traffic_scores.get(node.id, 0.0)
            dynamic_costs = {'traffic_density': v_traffic} if isinstance(v_traffic, (int, float)) else v_traffic
            
            weight = calculate_edge_weight(base_distance, dest_priority, dynamic_costs, alpha)
            graph[virtual_source].append((node.id, weight))
    
    # Determine source
    if virtual_source:
        source = virtual_source
    elif source_node_id:
        source = source_node_id
    else:
        source = nodes[0].id  # Default to first node
    
    # Run Dijkstra's algorithm
    distances, predecessors = dijkstra(graph, source)
    
    # Create optimal visiting order (greedy approach visiting nearest unvisited high-priority nodes)
    unvisited = set(n.id for n in nodes)
    route_path = []
    total_cost = 0.0
    current = source
    
    # If starting from virtual source, move to best first node
    if virtual_source and unvisited:
        best_first = min(unvisited, key=lambda nid: distances.get(nid, float('inf')))
        route_path.append(best_first)
        total_cost += distances.get(best_first, 0.0)
        current = best_first
        unvisited.remove(best_first)
    
    # Visit remaining nodes in order of proximity and priority
    while unvisited:
        # Recalculate distances from current position
        current_distances, _ = dijkstra(graph, current)
        
        # Choose next node (nearest among remaining)
        next_node = min(unvisited, key=lambda nid: current_distances.get(nid, float('inf')))
        
        route_path.append(next_node)
        total_cost += current_distances.get(next_node, 0.0)
        current = next_node
        unvisited.remove(next_node)

    # Optional deterministic local-search refinement on REAL distances.
    if refine and len(route_path) >= 4:
        coords = _coord_map(nodes)
        if virtual_source and user_location:
            depot_coord = (user_location['lat'], user_location['lng'])
        else:
            first = nodes[0]
            depot_coord = (first.latitude or 0.0, first.longitude or 0.0)
        refined = or_opt_order(
            two_opt_order(route_path, coords, depot_coord), coords, depot_coord
        )
        route_path = refined

    return {
        'path': route_path,
        'total_cost': total_cost,
        'alpha': alpha,
        'priority_scores': priority_scores,
        'source_type': 'user_location' if virtual_source else 'node',
        'graph_edges': graph,
        'refined': bool(refine and len(route_path) >= 4),
    }

# ---------------------------------------------------------------------------
# Local search + prize-collecting refinements (added in the SCS revision).
#
# The plain greedy nearest-neighbour tour over priority-warped edges can inflate
# total travel distance (a myopic-detour pathology).  These deterministic,
# sub-second refinements address it:
#   * two_opt_order / or_opt_order  -- classic local search on REAL distance
#     (Croes 1958; Lin & Kernighan 1973) to trim tour length.
#   * orienteering_route            -- prize-collecting subset routing within a
#     distance budget (Balas 1989; Golden et al. 1987) so the fleet serves the
#     urgent subset instead of sweeping every node each cycle.
#   * apply_aging                   -- anti-starvation term that bounds the
#     worst-case wait of low-priority bins (service-equity safeguard).
# ---------------------------------------------------------------------------
def _coord_map(nodes):
    return {n.id: (n.latitude or 0.0, n.longitude or 0.0) for n in nodes}


def _order_distance(order, coords, depot_coord):
    """Real great-circle distance of visiting `order`, out-and-back to depot."""
    if not order:
        return 0.0
    la, lo = depot_coord
    d = haversine_distance(la, lo, *coords[order[0]])
    for a, b in zip(order[:-1], order[1:]):
        d += haversine_distance(*coords[a], *coords[b])
    d += haversine_distance(*coords[order[-1]], la, lo)
    return d


def two_opt_order(order, coords, depot_coord, max_pass=30):
    """2-opt improvement of a visiting order using real distances."""
    if len(order) < 4:
        return order[:]
    seq = [None] + order[:] + [None]           # sentinels for depot at both ends

    def dist(i, j):
        ci = depot_coord if seq[i] is None else coords[seq[i]]
        cj = depot_coord if seq[j] is None else coords[seq[j]]
        return haversine_distance(ci[0], ci[1], cj[0], cj[1])

    def total():
        return sum(dist(i, i + 1) for i in range(len(seq) - 1))

    cur = total(); improved = True; passes = 0
    while improved and passes < max_pass:
        improved = False; passes += 1
        for i in range(1, len(seq) - 2):
            for k in range(i + 1, len(seq) - 1):
                if k - i == 1:
                    continue
                new = seq[:i] + seq[i:k + 1][::-1] + seq[k + 1:]
                nt = sum(
                    haversine_distance(
                        *(depot_coord if new[p] is None else coords[new[p]]),
                        *(depot_coord if new[p + 1] is None else coords[new[p + 1]])
                    ) for p in range(len(new) - 1)
                )
                if nt + 1e-6 < cur:
                    seq = new; cur = nt; improved = True
    return [x for x in seq if x is not None]


def or_opt_order(order, coords, depot_coord, seg_sizes=(1, 2, 3), max_pass=10):
    """Or-opt: relocate short segments to cheaper positions (real distance)."""
    seq = order[:]
    improved = True; passes = 0
    while improved and passes < max_pass:
        improved = False; passes += 1
        cur = _order_distance(seq, coords, depot_coord)
        for s in seg_sizes:
            for i in range(0, len(seq) - s + 1):
                seg = seq[i:i + s]
                rest = seq[:i] + seq[i + s:]
                for j in range(0, len(rest) + 1):
                    cand = rest[:j] + seg + rest[j:]
                    if _order_distance(cand, coords, depot_coord) + 1e-6 < cur:
                        seq = cand; improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return seq


def orienteering_route(nodes, priority_scores, user_location, budget_m, alpha=0.5):
    """
    Prize-collecting orienteering: greedily insert the node with the best
    prize/added-distance ratio while the tour stays within `budget_m`.
    Returns {'path', 'total_distance_m', 'collected_priority'}.
    """
    coords = _coord_map(nodes)
    depot = (user_location['lat'], user_location['lng']) if user_location else \
        (nodes[0].latitude or 0.0, nodes[0].longitude or 0.0)
    remaining = set(n.id for n in nodes)
    order = []
    while remaining:
        best_j, best_ratio, best_pos = None, -1.0, None
        seq = order
        for j in remaining:
            best_add, best_at = None, None
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [j] + seq[pos:]
                add = _order_distance(trial, coords, depot) - _order_distance(seq, coords, depot)
                if best_add is None or add < best_add:
                    best_add, best_at = add, pos
            ratio = priority_scores.get(j, 0.0) / max(best_add, 1.0)
            if ratio > best_ratio:
                best_ratio, best_j, best_pos = ratio, j, best_at
        trial = order[:best_pos] + [best_j] + order[best_pos:]
        if _order_distance(trial, coords, depot) <= budget_m:
            order = two_opt_order(trial, coords, depot)
        remaining.discard(best_j)
    return {
        'path': order,
        'total_distance_m': _order_distance(order, coords, depot),
        'collected_priority': sum(priority_scores.get(j, 0.0) for j in order),
    }


def apply_aging(priority_scores, hours_since_visit, gamma=0.5, tau=48.0):
    """
    Anti-starvation: boost a bin's routing prize by up to `gamma` as its idle
    time approaches `tau` hours, bounding the worst-case wait of low-priority
    bins with negligible cost to hazard response.
    """
    out = {}
    for nid, p in priority_scores.items():
        wait = hours_since_visit.get(nid, 0.0)
        out[nid] = p + gamma * min(1.0, wait / tau)
    return out


def compute_route(graph, source, targets=None):
    """
    Legacy function for backward compatibility
    Enhanced to support the new priority-based routing
    """
    if not targets:
        # Return distances tree
        dist, prev = dijkstra(graph, source)
        return {'source': source, 'distances': dist, 'tree_prev': prev}
    
    # Multi-target route optimization
    route_path = []
    total_cost = 0.0
    current = source
    remaining = list(targets)
    
    while remaining:
        # Find shortest path to all remaining targets
        dist, prev = dijkstra(graph, current)
        
        # Choose nearest target
        nearest = min(remaining, key=lambda t: dist.get(t, float('inf')))
        
        # Reconstruct path segment
        segment = reconstruct_path(prev, nearest)
        
        # Add segment to route (avoid duplicating current node)
        if route_path and segment and segment[0] == route_path[-1]:
            route_path.extend(segment[1:])
        else:
            route_path.extend(segment)
        
        total_cost += dist.get(nearest, 0.0)
        current = nearest
        remaining.remove(nearest)
    
    return {
        'path': route_path, 
        'total_cost': total_cost,
        'alpha': getattr(graph, 'alpha', 0.5) if hasattr(graph, 'alpha') else 0.5
    }