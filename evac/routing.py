"""
evac/routing.py
Route choice for evacuees, with HARD safety rules applied before any cost:

  - edges with hazard >= HAZARD_IMPASSABLE are removed
  - during a fire, ordinary lifts are removed (LIFTS_DISABLED_IN_FIRE)
  - otherwise lifts are kept for step-free users only (LIFTS_ONLY_FOR_STEP_FREE)
  - step-free (mobility-impaired) agents never use stairs
A cost function can make a route slower, but it can never send anyone through
a removed edge.

Methods:
  nearest        shortest walking distance to any exit (hazard-blind apart from the hard rules)
  safe_shortest  travel time x hazard penalty (rule-based, same penalty shape as generate_dataset.py)
  ml             the existing Random Forest edge-cost model (needs scikit-learn + models/)
  reactive       starts like safe_shortest, then reroutes people once queues have formed
                 (current congestion only - no prediction). See congestion_route_table().

All exits are searched at once with one multi-source Dijkstra from the exits, which
gives EVERY node its best path in a single pass.
"""

import networkx as nx
from . import config as C
from .building import exits

METHODS = ("nearest", "safe_shortest", "ml", "reactive")


def _allowed(d, fire_active, step_free):
    if d.get("hazard", 0.0) >= C.HAZARD_IMPASSABLE:
        return False
    if d["etype"] == "lift":
        if fire_active and C.LIFTS_DISABLED_IN_FIRE:
            return False
        if not step_free and C.LIFTS_ONLY_FOR_STEP_FREE:
            return False
    if d["etype"] == "stair" and step_free:
        return False
    return True


def _hazard_penalty(h):
    return 1.0 + 12.0 * h * h + 60.0 * max(0.0, h - 0.7) ** 3


_ml_cache = {}


def _ml_weights(G):
    """Edge costs from the existing RF model, keyed by edge. Cached per hazard state."""
    key = tuple(round(d.get("hazard", 0.0), 3) for _, _, d in G.edges(data=True))
    if key in _ml_cache:
        return _ml_cache[key]
    from ml_routing_model import get_predictor  # existing project module

    pred = get_predictor()
    edges = list(G.edges(data=True))
    recs = [
        {"distance": d["base_distance"], "hazard_level": d.get("hazard", 0.0),
         "crowd_density": 0.0, "accessibility": 1.0}
        for _, _, d in edges
    ]
    costs = pred.batch_predict(recs)
    w = {}
    for (u, v, _), c in zip(edges, costs):
        w[(u, v)] = w[(v, u)] = float(c)
    _ml_cache.clear()
    _ml_cache[key] = w
    return w


def weight_fn(G, method, fire_active, step_free):
    if method == "ml":
        mlw = _ml_weights(G)

    def w(u, v, d):
        if not _allowed(d, fire_active, step_free):
            return None  # networkx: None hides the edge
        t = d["length_m"] / d["free_speed"]
        if method == "nearest":
            return d["length_m"]
        if method == "safe_shortest":
            return t * _hazard_penalty(d.get("hazard", 0.0))
        if method == "ml":
            return mlw[(u, v)]
        raise ValueError(f"unknown method {method}")

    return w


def route_table(G, method, fire_active, step_free):
    """{node: path-to-best-exit} for every node that can reach an exit."""
    ex = [e for e in exits(G) if G.nodes[e].get("hazard", 0.0) < C.HAZARD_IMPASSABLE]
    if not ex:
        return {}
    _dist, paths = nx.multi_source_dijkstra(G, ex, weight=weight_fn(G, method, fire_active, step_free))
    # paths run exit -> node; reverse so they run node -> exit
    return {n: list(reversed(p)) for n, p in paths.items()}


def next_hop_signs(G, method="safe_shortest", fire_active=True, step_free=False):
    """Building-wide 'go this way' arrow for every node (kiosk signage)."""
    table = route_table(G, method, fire_active, step_free)
    return {n: (p[1] if len(p) > 1 else None) for n, p in table.items()}


def congestion_route_table(G, fire_active, step_free, queue_count, occupancy):
    """
    Routes using CURRENT congestion (used by the 'reactive' method).
    Cost of entering edge a->b = travel time at current density x hazard penalty
                                 + expected wait in the queue at a (queued people / flow capacity).
    queue_count: {(a, b): people waiting at a to enter a->b}
    occupancy:   {sorted edge key: people currently on the edge}
    The search runs outward from the exits, so when networkx relaxes u -> v the
    evacuee actually walks v -> u and queues at v.
    """
    from .physics import weidmann_speed_factor

    ex = [e for e in exits(G) if G.nodes[e].get("hazard", 0.0) < C.HAZARD_IMPASSABLE]
    if not ex:
        return {}

    def w(u, v, d):
        if not _allowed(d, fire_active, step_free):
            return None
        key = (u, v) if u <= v else (v, u)
        dens = occupancy.get(key, 0) / (d["length_m"] * d["width_m"])
        t = d["length_m"] / (d["free_speed"] * weidmann_speed_factor(dens))
        wait = queue_count.get((v, u), 0) / d["flow_cap"]
        return t * _hazard_penalty(d.get("hazard", 0.0)) + wait

    _dist, paths = nx.multi_source_dijkstra(G, ex, weight=w)
    return {n: list(reversed(p)) for n, p in paths.items()}
