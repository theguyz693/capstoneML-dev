"""
ml_evacuation_router.py
Emergency routing engine that computes optimal evacuation paths using
ML-predicted edge weights with Dijkstra and A* algorithms.
"""

import math
import networkx as nx

GROUND_EXITS = [
    'north entrance/exit',
    'east entrance/exit',
    'north main entrance',
    'east main entrance'
]

def euclidean_heuristic_3d(u, v, G):
    """
    Admissible 3D Euclidean distance heuristic for A* pathfinding.
    """
    n1 = G.nodes[u]
    n2 = G.nodes[v]
    dx = n1.get('x', 0) - n2.get('x', 0)
    dy = n1.get('y', 0) - n2.get('y', 0)
    dz = n1.get('z', 0) - n2.get('z', 0)
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def get_exits_for_graph(G):
    """
    Finds all valid exit/entrance nodes present in graph G.
    """
    exits = [n for n in GROUND_EXITS if n in G.nodes]
    if not exits:
        exits = [n for n in G.nodes if ('exit' in n.lower() or 'entrance' in n.lower()) and G.nodes[n].get('floor', 0) == 0]
    return exits

def analyze_path_metrics(G, path):
    """
    Calculates detailed metrics for a given path sequence.
    """
    total_physical_dist = 0.0
    total_ml_cost = 0.0
    max_hazard = 0.0
    crowd_levels = []
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = G[u][v]
        dist = edge_data.get('base_distance', edge_data.get('weight', 0.0))
        cost = edge_data.get('weight', dist)
        haz = edge_data.get('hazard_level', 0.0)
        crd = edge_data.get('crowd_density', 0.0)
        
        total_physical_dist += dist
        total_ml_cost += cost
        max_hazard = max(max_hazard, haz)
        crowd_levels.append(crd)
        
    avg_crowd = sum(crowd_levels) / len(crowd_levels) if crowd_levels else 0.0
    
    return {
        'path': path,
        'step_count': len(path),
        'physical_distance': round(total_physical_dist, 2),
        'predicted_cost': round(total_ml_cost, 2),
        'max_hazard': round(max_hazard, 4),
        'avg_crowd': round(avg_crowd, 4),
        'is_safe': max_hazard < 0.70
    }

def find_optimal_evacuation_routes(G, start_node, exit_nodes=None, algorithm='dijkstra'):
    """
    Computes optimal evacuation routes to all reachable exits using Dijkstra or A*.
    Returns a list of route analyses sorted by lowest predicted ML cost.
    """
    if start_node not in G.nodes:
        raise ValueError(f"Start node '{start_node}' not found in building graph.")
        
    if exit_nodes is None:
        exit_nodes = get_exits_for_graph(G)
        
    routes = []
    for exit_node in exit_nodes:
        if exit_node not in G.nodes or exit_node == start_node:
            continue
            
        try:
            if algorithm.lower() == 'astar':
                path = nx.astar_path(
                    G, source=start_node, target=exit_node, 
                    heuristic=lambda u, v: euclidean_heuristic_3d(u, v, G), 
                    weight='weight'
                )
            else: # dijkstra
                path = nx.shortest_path(G, source=start_node, target=exit_node, weight='weight')
                
            metrics = analyze_path_metrics(G, path)
            metrics['destination_exit'] = exit_node
            routes.append(metrics)
            
        except nx.NetworkXNoPath:
            continue
            
    if not routes:
        return []
        
    # Sort by lowest ML cost
    routes.sort(key=lambda r: r['predicted_cost'])
    return routes

def compare_static_vs_dynamic_routing(G_static, G_dynamic, start_node):
    """
    Directly compares the baseline static distance route vs the ML-informed dynamic route.
    Demonstrates hazard and crowd avoidance.
    """
    static_routes = find_optimal_evacuation_routes(G_static, start_node)
    dynamic_routes = find_optimal_evacuation_routes(G_dynamic, start_node)
    
    if not static_routes or not dynamic_routes:
        return None
        
    best_static = static_routes[0]
    best_dynamic = dynamic_routes[0]
    
    # Evaluate the static path on the dynamic graph to see what hazard the static path would encounter
    static_on_dynamic_metrics = analyze_path_metrics(G_dynamic, best_static['path'])
    static_on_dynamic_metrics['destination_exit'] = best_static['destination_exit']
    
    return {
        'start_node': start_node,
        'static_route': best_static,
        'static_route_with_current_hazards': static_on_dynamic_metrics,
        'dynamic_ml_route': best_dynamic,
        'path_changed': best_static['path'] != best_dynamic['path'],
        'exit_changed': best_static['destination_exit'] != best_dynamic['destination_exit']
    }
