"""
evac/optimum.py
Theoretical best evacuation time - a LOWER BOUND no routing method can beat.

Method: quickest-flow on a time-expanded network (Ford & Fulkerson; Hoppe & Tardos 2000).
  - Time is cut into steps of TAU seconds. Each node gets a copy per step.
  - Moving along edge u->v takes ceil(walk_time / TAU) steps and at most eff_cap * TAU
    people may start it per step. People may wait at nodes (holdover arcs).
  - Each person becomes available at their origin at their pre-movement time.
  - Max-flow from all people to the exits within T steps = how many can be out by T.
  - Binary search for the smallest T where everyone is out.

It is a valid lower bound for the simulator's flow model because it relaxes it:
flow is fractional, people are perfectly coordinated, speed never drops below free
speed, and step-free users are allowed every non-forbidden edge. The gap between a
method and this bound measures how much better ANY routing could still do.
"""

import math
from collections import Counter

import networkx as nx

from . import config as C
from .building import exits
from .routing import _allowed

TAU = 2.0


def _feasible_count(G, sources, T, fire_active):
    """Max number of people that can reach an exit by step T."""
    H = nx.DiGraph()
    ex = {e for e in exits(G) if G.nodes[e].get("hazard", 0.0) < C.HAZARD_IMPASSABLE}
    for (node, step), cnt in sources.items():
        if step <= T:
            H.add_edge("S", (node, step), capacity=cnt)
    for n in G:
        for t in range(T):
            H.add_edge((n, t), (n, t + 1))          # wait (no capacity attr = infinite)
        if n in ex:
            for t in range(T + 1):
                H.add_edge((n, t), "X")
    for u, v, d in G.edges(data=True):
        if not _allowed(d, fire_active, step_free=False):
            continue
        # round travel time DOWN (and release times down) so this stays a true lower bound
        k = math.floor((d["length_m"] / d["free_speed"]) / TAU)
        c = d["eff_cap"] * TAU
        for a, b in ((u, v), (v, u)):
            for t in range(T - k + 1):
                if k == 0:   # very short edge: same time layer
                    H.add_edge((a, t), (b, t), capacity=c)
                else:
                    H.add_edge((a, t), (b, t + k), capacity=c)
    if "S" not in H or "X" not in H:
        return 0.0
    return nx.maximum_flow_value(H, "S", "X")


def lower_bound(G, agents, fire_active, hi_s=None):
    """
    agents: objects with .origin, .start, .path (None = cannot be evacuated; excluded).
    Returns the minimum possible last-out time in seconds (to TAU resolution).
    """
    movers = [a for a in agents if a.path is not None]
    if not movers:
        return None
    sources = Counter((a.origin, math.floor(a.start / TAU)) for a in movers)
    need = len(movers) - 1e-6
    lo = max(s for (_n, s) in sources) + 1
    hi = math.ceil((hi_s or C.MAX_TIME_S) / TAU)
    while _feasible_count(G, sources, hi, fire_active) < need:
        if hi * TAU >= C.MAX_TIME_S:
            return None
        hi = min(math.ceil(C.MAX_TIME_S / TAU), hi * 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if _feasible_count(G, sources, mid, fire_active) >= need:
            hi = mid
        else:
            lo = mid + 1
    return hi * TAU
