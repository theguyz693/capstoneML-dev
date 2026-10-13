"""
evac/literature.py
Two PUBLISHED evacuation-routing methods, implemented on our building graph as baselines, so the
look-ahead planner is compared with other researchers' methods and not only with our own simple ones.
(CCRP, Lu, George & Shekhar 2005, is already in evac/planner.py as method 'ccrp'.)

quickest  Dynamic quickest-path route choice
          Kemloh Wagoum, A. U., Seyfried, A. & Holl, S. (2012). Modeling the dynamic route choice of
          pedestrians to assess the criticality of building evacuation. Advances in Complex Systems
          15(7), 1250029. doi:10.1142/S0219525912500294
          Each person re-evaluates when they reach a new decision point (here: every graph node) and
          when they have been stuck in a jam for longer than their patience (10 s, +1 s after every
          attempt that did not lead to a switch). The estimated travel time of a route is the walking
          time plus the time for the jam ahead to clear (network equivalent of the paper's 'reference
          pedestrian' estimate). They switch only if the cost-benefit
              cba = (g_alt - g_cur) / (g_alt + g_cur),  g = 1 / travel time
          exceeds g_min = 0.15 (unfamiliar pedestrians, as in the paper).
          Local and reactive: uses only what the person can see now, no prediction, no coordination.

casper    Capacity-Aware Shortest Path Evacuation Routing
          Shahabi, K. & Wilson, J. P. (2014). CASPER: Intelligent capacity-aware evacuation routing.
          Computers, Environment and Urban Systems 46, 12-24. doi:10.1016/j.compenvurbsys.2014.03.004
          One route per source (room), planned in advance. Sources are processed from the farthest
          from an exit to the nearest; each gets the cheapest path where an edge's cost grows with
          the people ALREADY routed over it plus its own group:
              cost(e, s) = imp(e) / T(flow(e) + flow(s), cap(e))
          CASPER leaves the traffic model T as an input. Our pedestrian instance: an edge carrying F
          people takes at least F / sustainable capacity seconds to pass them all, so
              cost = max(free walking time, F / eff_cap) x hazard penalty.
          Global and planned, but books total load per edge rather than time slots (unlike CCRP).
"""

from collections import defaultdict

import networkx as nx

from . import config as C
from .building import exits
from .physics import weidmann_speed_factor
from .routing import _allowed, _hazard_penalty, congestion_route_table, weight_fn

QP_GAIN_MIN = 0.15          # g_min for unfamiliar pedestrians (Kemloh Wagoum et al. 2012)
QP_PATIENCE_S = 10.0        # initial patience t_min
QP_PATIENCE_STEP_S = 1.0    # + this after each failed attempt


# ====================================================================== quickest path
def queue_counts(queues):
    """{(a, b): people waiting at a to enter a->b}"""
    q = defaultdict(int)
    for lst in queues.values():
        for a in lst:
            if a.path and a.idx < len(a.path) - 1:
                q[(a.path[a.idx], a.path[a.idx + 1])] += 1
    return q


def route_time(G, path, fire_active, step_free, qcount, occ):
    """Estimated time to walk `path` now: travel time at current density x hazard + queue to clear at each node.
    Same cost as congestion_route_table() so current and alternative routes are judged alike."""
    total = 0.0
    for a, b in zip(path, path[1:]):
        d = G.edges[a, b]
        if not _allowed(d, fire_active, step_free):
            return float("inf")
        key = (a, b) if a <= b else (b, a)
        dens = occ.get(key, 0) / (d["length_m"] * d["width_m"])
        t = d["length_m"] / (d["free_speed"] * weidmann_speed_factor(dens))
        total += t * _hazard_penalty(d.get("hazard", 0.0)) + qcount.get((a, b), 0) / d["flow_cap"]
    return total


class QuickestPath:
    """Holds the per-step cache and per-person patience for method 'quickest'."""

    def __init__(self, G, fire_active):
        self.G = G
        self.fire = fire_active
        self._stamp = None
        self._tables = {}
        self._q = None
        self.patience = {}        # agent id -> current patience (s)
        self.next_check = {}      # agent id -> time of next jam check
        self.switches = 0

    def _refresh(self, now, queues, occ):
        if self._stamp != now:
            self._stamp = now
            self._tables = {}
            self._q = queue_counts(queues)

    def consider(self, a, now, queues, occ):
        """Re-evaluate person `a` standing at a.path[a.idx]. Returns True if they switched."""
        if not a.compliant or a.path is None or a.idx >= len(a.path) - 1:
            return False
        self._refresh(now, queues, occ)
        sf = a.step_free
        if sf not in self._tables:
            self._tables[sf] = congestion_route_table(self.G, self.fire, sf, self._q, occ)
        node = a.path[a.idx]
        alt = self._tables[sf].get(node)
        cur = a.path[a.idx:]
        if not alt or alt == cur:
            return False
        if a.idx > 0 and len(alt) > 1 and alt[1] == a.path[a.idx - 1]:
            return False                      # never turn straight back
        t_cur = route_time(self.G, cur, self.fire, sf, self._q, occ)
        t_alt = route_time(self.G, alt, self.fire, sf, self._q, occ)
        if t_cur == float("inf"):
            gain = 1.0
        else:
            g_cur, g_alt = 1.0 / max(t_cur, 1e-6), 1.0 / max(t_alt, 1e-6)
            gain = (g_alt - g_cur) / (g_alt + g_cur)
        if gain > QP_GAIN_MIN:
            a.path = a.path[:a.idx] + alt
            a.exit = alt[-1]
            a.changes += 1
            self.switches += 1
            return True
        return False

    def on_arrival(self, a, now, queues, occ):
        """Entering a new decision point: re-evaluate, reset patience."""
        self.patience[a.id] = QP_PATIENCE_S
        self.next_check[a.id] = now + QP_PATIENCE_S
        self.consider(a, now, queues, occ)

    def jam_check(self, now, queues, occ):
        """People who have waited longer than their patience re-evaluate; failed attempts raise patience."""
        for lst in list(queues.values()):
            for a in list(lst):
                if not a.compliant or now < self.next_check.get(a.id, float("inf")):
                    continue
                if self.consider(a, now, queues, occ):
                    self.patience[a.id] = QP_PATIENCE_S
                else:
                    self.patience[a.id] = self.patience.get(a.id, QP_PATIENCE_S) + QP_PATIENCE_STEP_S
                self.next_check[a.id] = now + self.patience[a.id]


# ====================================================================== CASPER
def casper_plan(G, agents, fire_active):
    """{agent id: path}. One path per (origin room, step-free) source, farthest source first."""
    groups = defaultdict(list)
    for a in agents:
        groups[(a.origin, a.step_free)].append(a)
    ex = [e for e in exits(G) if G.nodes[e].get("hazard", 0.0) < C.HAZARD_IMPASSABLE]
    if not ex:
        return {}
    dist = {}
    for sf in (False, True):
        dist[sf], _ = nx.multi_source_dijkstra(G, ex, weight=weight_fn(G, "safe_shortest", fire_active, sf))
    order = sorted(groups, key=lambda k: (-dist[k[1]].get(k[0], -1.0), k[0], k[1]))
    flow = defaultdict(float)
    out = {}
    for origin, sf in order:
        if origin not in dist[sf]:
            continue                                   # no route at all (handled by the simulator)
        n_s = sum(getattr(a, "size", 1.0) for a in groups[(origin, sf)])

        def w(u, v, d, n_s=n_s, sf=sf):
            if not _allowed(d, fire_active, sf):
                return None
            key = (u, v) if u <= v else (v, u)
            walk = d["length_m"] / d["free_speed"]
            return max(walk, (flow[key] + n_s) / d["eff_cap"]) * _hazard_penalty(d.get("hazard", 0.0))

        try:
            _c, p = nx.multi_source_dijkstra(G, ex, target=origin, weight=w)
        except nx.NetworkXNoPath:
            continue
        path = list(reversed(p))
        for u, v in zip(path, path[1:]):
            flow[(u, v) if u <= v else (v, u)] += n_s
        for a in groups[(origin, sf)]:
            out[a.id] = path
    return out
