"""
ml_edge_updater.py
Adapts existing building graph structures (single-floor & multi-floor)
and applies environmental hazard/crowd scenarios to update edge weights via ML predictions.
"""

import math
import networkx as nx
from ml_routing_model import get_predictor, MLEdgeCostPredictor

def build_ground_floor_graph():
    """
    Builds baseline NetworkX graph for Ground Floor from existing nodes and edges.
    """
    from campus_data import nodes_g_raw, edges_g
    
    G = nx.Graph()
    for name, pos in nodes_g_raw.items():
        G.add_node(name, pos=pos, floor=0, x=pos[0], y=pos[1], z=0)
        
    for u, v in edges_g:
        if u in nodes_g_raw and v in nodes_g_raw:
            p1, p2 = nodes_g_raw[u], nodes_g_raw[v]
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            is_stairs = ('stairs' in u.lower()) or ('stairs' in v.lower())
            accessibility = 0.0 if is_stairs else 1.0
            
            G.add_edge(
                u, v,
                base_distance=round(dist, 2),
                weight=round(dist, 2),
                hazard_level=0.0,
                crowd_density=0.0,
                accessibility=accessibility,
                is_stairs=is_stairs
            )
    return G

def build_full_campus_graph():
    """
    Builds the complete 7-story multi-floor NetworkX graph using existing campus_data.
    """
    from campus_data import (
        nodes_g_raw, edges_g,
        nodes_f1_raw, edges_f1,
        nodes_f2, edges_f2,
        nodes_f3, edges_f3,
        nodes_f4, edges_f4,
        nodes_f5, edges_f5,
        nodes_f6, edges_f6,
        bridges
    )
    
    G = nx.Graph()
    
    # Floor height and alignment
    FLOOR_HEIGHT = 350
    ground_anchor = nodes_g_raw.get('stairs 1', (1179, 1174))
    
    # 1. Ground Floor Nodes
    for name, (x, y) in nodes_g_raw.items():
        G.add_node(name, pos=(x, y), floor=0, x=x, y=y, z=0)
        
    # 2. Upper Floors Nodes
    upper_floors = [
        {"nodes": nodes_f1_raw, "edges": edges_f1, "anchor": "f1_stairs 1", "level": 1, "scale": 1.5},
        {"nodes": nodes_f2, "edges": edges_f2, "anchor": "f2_stairs 1", "level": 2, "scale": 1.8},
        {"nodes": nodes_f3, "edges": edges_f3, "anchor": "f3_stairs 1", "level": 3, "scale": 1.8},
        {"nodes": nodes_f4, "edges": edges_f4, "anchor": "f4_stairs 1", "level": 4, "scale": 0.4},
        {"nodes": nodes_f5, "edges": edges_f5, "anchor": "f5_stairs 1", "level": 5, "scale": 0.4},
        {"nodes": nodes_f6, "edges": edges_f6, "anchor": "f6_stairs 1", "level": 6, "scale": 0.4},
    ]
    
    for fl in upper_floors:
        anchor_coords = fl["nodes"].get(fl["anchor"], (0, 0))
        z = fl["level"] * FLOOR_HEIGHT
        for name, (x, y) in fl["nodes"].items():
            dx = (x - anchor_coords[0]) / fl["scale"]
            dy = (y - anchor_coords[1]) / fl["scale"]
            final_x = ground_anchor[0] + dx
            final_y = ground_anchor[1] + dy
            G.add_node(name, pos=(final_x, final_y), floor=fl["level"], x=final_x, y=final_y, z=z)
            
    # Combine all floor edges
    all_edge_tuples = list(edges_g)
    for fl in upper_floors:
        all_edge_tuples.extend(fl["edges"])
    all_edge_tuples.extend(bridges)
    
    for u, v in all_edge_tuples:
        if u in G.nodes and v in G.nodes:
            n1 = G.nodes[u]
            n2 = G.nodes[v]
            # 3D Euclidean distance
            dist = math.sqrt((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2 + (n1['z'] - n2['z'])**2)
            is_stairs = ('stairs' in u.lower()) or ('stairs' in v.lower())
            is_lift = ('lift' in u.lower()) or ('lift' in v.lower())
            
            accessibility = 0.0 if is_stairs else 1.0
            
            G.add_edge(
                u, v,
                base_distance=round(dist, 2),
                weight=round(dist, 2),
                hazard_level=0.0,
                crowd_density=0.0,
                accessibility=accessibility,
                is_stairs=is_stairs,
                is_lift=is_lift
            )
            
    return G

def apply_environmental_scenario(G, hazard_sources=None, crowd_sources=None, 
                                 mobility_impaired=False, predictor=None):
    """
    Applies spatial hazards and crowd hotspots to the graph, and updates edge weights via ML.
    
    hazard_sources: list of dicts [{'node': 'j5', 'radius': 300, 'intensity': 1.0}, ...]
                    or [{'edge': ('j5', 'j6'), 'hazard_level': 0.95}, ...]
    crowd_sources: list of dicts [{'node': 'j10', 'radius': 250, 'intensity': 0.8}, ...]
    """
    if predictor is None:
        predictor = get_predictor()
        
    hazard_sources = hazard_sources or []
    crowd_sources = crowd_sources or []
    
    # 1. Reset dynamic fields to clean defaults
    for u, v, d in G.edges(data=True):
        d['hazard_level'] = 0.0
        d['crowd_density'] = 0.0
        # If user is mobility impaired, stairs become non-accessible
        if mobility_impaired and d.get('is_stairs', False):
            d['accessibility'] = 0.0
        elif not mobility_impaired:
            d['accessibility'] = 1.0
            
    # 2. Compute spatial node hazard & crowd distributions
    node_hazards = {n: 0.0 for n in G.nodes}
    node_crowds = {n: 0.0 for n in G.nodes}
    
    # Process hazard sources
    for haz in hazard_sources:
        if 'node' in haz and haz['node'] in G.nodes:
            center_node = haz['node']
            radius = haz.get('radius', 350.0)
            intensity = haz.get('intensity', 1.0)
            center_pos = G.nodes[center_node]
            
            for n, data in G.nodes.items():
                if data.get('floor') == center_pos.get('floor'):
                    dist = math.sqrt((data['x'] - center_pos['x'])**2 + (data['y'] - center_pos['y'])**2)
                    if dist <= radius:
                        haz_val = intensity * (1.0 - (dist / radius))
                        node_hazards[n] = max(node_hazards[n], haz_val)
                        
        elif 'edge' in haz:
            u, v = haz['edge']
            haz_val = haz.get('hazard_level', 0.95)
            if G.has_edge(u, v):
                G[u][v]['hazard_level'] = max(G[u][v]['hazard_level'], haz_val)
                
    # Process crowd sources
    for crd in crowd_sources:
        if 'node' in crd and crd['node'] in G.nodes:
            center_node = crd['node']
            radius = crd.get('radius', 300.0)
            intensity = crd.get('intensity', 0.8)
            center_pos = G.nodes[center_node]
            
            for n, data in G.nodes.items():
                if data.get('floor') == center_pos.get('floor'):
                    dist = math.sqrt((data['x'] - center_pos['x'])**2 + (data['y'] - center_pos['y'])**2)
                    if dist <= radius:
                        crd_val = intensity * (1.0 - (dist / radius))
                        node_crowds[n] = max(node_crowds[n], crd_val)
                        
    # 3. Propagate node conditions to edges
    edge_records = []
    edges_list = list(G.edges(data=True))
    
    for u, v, d in edges_list:
        # Edge hazard is max of its endpoints or explicitly assigned value
        computed_haz = max(d.get('hazard_level', 0.0), node_hazards.get(u, 0.0), node_hazards.get(v, 0.0))
        computed_crd = max(d.get('crowd_density', 0.0), node_crowds.get(u, 0.0), node_crowds.get(v, 0.0))
        
        d['hazard_level'] = round(float(computed_haz), 4)
        d['crowd_density'] = round(float(computed_crd), 4)
        
        edge_records.append({
            'distance': d['base_distance'],
            'hazard_level': d['hazard_level'],
            'crowd_density': d['crowd_density'],
            'accessibility': d.get('accessibility', 1.0)
        })
        
    # 4. Predict weights using ML model in batch
    predicted_weights = predictor.batch_predict(edge_records)
    
    for idx, (u, v, d) in enumerate(edges_list):
        d['weight'] = round(float(predicted_weights[idx]), 2)
        
    return G
