"""
evac/hazard.py
Turns a fire description into a hazard level (0..1) for every node and edge.

Model (deliberately simple and explainable - NOT fire physics):
  1. Same-floor spread: hazard falls linearly from the fire node to zero at `radius`
     (same rule as ml_edge_updater.apply_environmental_scenario).
  2. Smoke up stairwells (new): if the hazard zone reaches a stair node, that stair
     SHAFT carries smoke to every floor ABOVE, losing SMOKE_STAIR_DECAY per floor.
     Smoke rises, so floors below are not affected through the shaft.
"""

import math
from . import config as C
from .building import stair_shaft


def node_hazards(G, fires):
    """fires: list of {'node': str, 'radius': float (graph units), 'intensity': 0..1}"""
    h = {n: 0.0 for n in G}
    for f in fires or []:
        src = f.get("node")
        if src not in G:
            continue
        radius = float(f.get("radius", 350.0))
        inten = float(f.get("intensity", 1.0))
        c = G.nodes[src]
        fl = c["floor"]

        # 1. same-floor radial spread
        for n, d in G.nodes(data=True):
            if d["floor"] != fl:
                continue
            dist = math.hypot(d["x"] - c["x"], d["y"] - c["y"])
            if dist <= radius:
                h[n] = max(h[n], inten * (1.0 - dist / radius))

        # 2. smoke rises through any stair shaft touched by the zone
        touched = {}
        for n, d in G.nodes(data=True):
            if d["floor"] == fl and h[n] > 0 and stair_shaft(n):
                touched[stair_shaft(n)] = max(touched.get(stair_shaft(n), 0.0), h[n])
        for n, d in G.nodes(data=True):
            s = stair_shaft(n)
            if s in touched and d["floor"] > fl:
                up = d["floor"] - fl
                if up <= C.SMOKE_STAIR_MAX_FLOORS:
                    h[n] = max(h[n], touched[s] * (C.SMOKE_STAIR_DECAY ** up))
    return h


def apply_hazard(G, fires):
    """Writes `hazard` onto nodes and edges (edge = max of its two ends)."""
    nh = node_hazards(G, fires)
    for n in G:
        G.nodes[n]["hazard"] = round(nh[n], 4)
    for u, v, d in G.edges(data=True):
        d["hazard"] = round(max(nh[u], nh[v]), 4)
    return G
