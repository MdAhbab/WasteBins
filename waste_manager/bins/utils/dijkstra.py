import heapq
import math

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

def calculate_edge_weight(base_distance_m, destination_priority, destination_traffic=0.0, alpha=0.5):
    """
    Calculate edge weight based on distance, destination node priority, and traffic
    
    weight = (base_distance_m * (1 + destination_traffic * 3.0)) / (1 + alpha * (priority * 10))
    """
    priority = max(0.0, min(1.0, destination_priority))
    traffic = max(0.0, min(1.0, destination_traffic))
    
    # Increase base distance drastically when destination has high traffic
    adjusted_distance = base_distance_m * (1.0 + traffic * 3.0)
    
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
            dest_traffic = traffic_scores.get(node_v.id, 0.0)
            
            weight = calculate_edge_weight(base_distance, dest_priority, dest_traffic, alpha)
            graph[node_u.id].append((node_v.id, weight))
    
    return graph

def compute_optimal_route(nodes, priority_scores, traffic_scores=None, source_node_id=None, user_location=None, alpha=0.5):
    """
    Compute optimal route using Dijkstra's algorithm with priority-based edge weights
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
            dest_traffic = traffic_scores.get(node.id, 0.0)
            weight = calculate_edge_weight(base_distance, dest_priority, dest_traffic, alpha)
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
    
    return {
        'path': route_path,
        'total_cost': total_cost,
        'alpha': alpha,
        'priority_scores': priority_scores,
        'source_type': 'user_location' if virtual_source else 'node',
        'graph_edges': graph
    }

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