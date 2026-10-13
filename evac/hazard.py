"""
evac/hazard.py
Turns a fire description into a hazard level (0..1) for every node and edge.

Model (deliberately simple and explainable - NOT fire physics):
  1. Same-floor spread: hazard falls linearly from the fire node to zero at `radius`
     (same rule as ml_edge_updater.apply_environmental_scenario).
  2. Smoke up stairwells: if the hazard zone reaches a stair node, that stair
     SHAFT carries smoke to every floor ABOVE, losing SMOKE_STAIR_DECAY per floor.
     Smoke rises, so floors below are not affected through the shaft.
  3. Spreading fire: with 'spread_m_per_s' the radius grows over time and smoke
     reaches each higher floor of a touched stair shaft STAIR_SMOKE_DELAY_S later.
"""

import math
from . import config as C
from .building import stair_shaft


def fire_radius(f, t):
    """Radius at time t (graph units). A fire with 'spread_m_per_s' grows up to 'max_radius'."""
    r0 = float(f.get("radius", 350.0))
    if t is None or not f.get("spread_m_per_s"):
        return r0
    grow = float(f["spread_m_per_s"]) / C.METERS_PER_UNIT * t
    return min(float(f.get("max_radius", 4 * r0)), r0 + grow)


def is_dynamic(fires):
    return any(f.get("spread_m_per_s") for f in fires or [])


def node_hazards(G, fires, t=None):
    """fires: list of {'node': str, 'radius': float (graph units), 'intensity': 0..1,
                       optional 'spread_m_per_s', 'max_radius'}
    t: seconds since the alarm (None = static snapshot at full radius / instant smoke)."""
    h = {n: 0.0 for n in G}
    for f in fires or []:
        src = f.get("node")
        if src not in G:
            continue
        radius = fire_radius(f, t)
        dynamic = t is not None and f.get("spread_m_per_s")
        inten = float(f.get("intensity", 1.0))
        c = G.nodes[src]
        fl = c["floor"]

        # 1. same-floor radial spread; a "contained" fire stays in its fire compartment (fire doors):
        #    other compartments only get light smoke (COMPARTMENT_LEAK x, below the exposure threshold)
        comp = c.get("compartment") if f.get("contained") else None
        for n, d in G.nodes(data=True):
            if d["floor"] != fl:
                continue
            dist = math.hypot(d["x"] - c["x"], d["y"] - c["y"])
            if dist <= radius:
                v = inten * (1.0 - dist / radius)
                if comp is not None and d.get("compartment") != comp:
                    v *= C.COMPARTMENT_LEAK
                h[n] = max(h[n], v)

        # 2. smoke rises through any stair shaft touched by the zone
        touched = {}
        for n, d in G.nodes(data=True):
            if d["floor"] == fl and h[n] > 0 and stair_shaft(n):
                touched[stair_shaft(n)] = max(touched.get(stair_shaft(n), 0.0), h[n])
        for n, d in G.nodes(data=True):
            s = stair_shaft(n)
            if s in touched and d["floor"] > fl:
                up = d["floor"] - fl
                if dynamic and t < up * C.STAIR_SMOKE_DELAY_S:
                    continue   # smoke has not climbed this far yet
                if up <= C.SMOKE_STAIR_MAX_FLOORS:
                    h[n] = max(h[n], touched[s] * (C.SMOKE_STAIR_DECAY ** up))
    return h


def apply_hazard(G, fires, t=None):
    """Writes `hazard` onto nodes and edges (edge = max of its two ends)."""
    nh = node_hazards(G, fires, t)
    for n in G:
        G.nodes[n]["hazard"] = round(nh[n], 4)
    for u, v, d in G.edges(data=True):
        d["hazard"] = round(max(nh[u], nh[v]), 4)
    return G
