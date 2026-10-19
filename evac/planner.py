"""
evac/planner.py
Capacity-aware route planning: the core research contribution.

Idea (Capacity Constrained Route Planner, CCRP - Lu, George & Shekhar, 2005):
  Time is split into slots of SLOT_S seconds. Every edge can admit at most
  floor(eff_cap * SLOT_S) people per slot, where eff_cap is the edge's SUSTAINABLE flow
  (door/stair flow rate or the speed-density limit, whichever is lower). People are planned one at a time; each
  one takes the EARLIEST-ARRIVAL route given the capacity already BOOKED by the
  people planned before them, and then books the slots it uses. So when stair A's
  slots fill up, the next people are sent to stair B before any queue exists.

  Earliest arrival is found with a time-dependent Dijkstra: entering edge e at time t
  means waiting for the first slot >= t with free capacity, then walking it.
  Hazard is folded in exactly like `safe_shortest` (walking time x hazard penalty),
  so with no crowding the planner gives the same routes as `safe_shortest`.

Two uses:
  ccrp       plan everyone once at t = 0, then just simulate (one-shot plan)
  lookahead  re-plan every REPLAN_S seconds from where people really are, booking only
             within a horizon of HORIZON_S seconds (a rolling-horizon / MPC-style loop).
             Anti-oscillation: someone switches route only if the new route is predicted
             to be >= SWITCH_GAIN faster, and never by walking back to the node they came from.
"""

import heapq
import math
from collections import defaultdict

from . import config as C
from .building import exits
from .physics import smoke_speed_factor
from .routing import _allowed, _hazard_penalty


class Reservations:
    """Booked entries per (directed edge, time slot)."""

    def __init__(self, G, now=0.0, horizon=math.inf, walk=None):
        self.G = G
        self.walk = walk or {}     # optional learned walking time per directed edge (lookahead_ml)
        self.book = defaultdict(int)
        self.now = now
        self.horizon = horizon

    def cap(self, d):
        cap = d["eff_cap"] if C.PLANNER_USE_SUSTAINABLE_CAP else d["flow_cap"]
        return max(1, int(cap * C.SLOT_S))

    @staticmethod
    def slot(t):
        # small tolerance: s * SLOT_S computed in floating point (e.g. 3 * 4.1 = 12.299999...) must map
        # back to slot s, or bookings land in the previous (full) slot and capacity is overbooked
        return int(math.floor(t / C.SLOT_S + 1e-9))

    def entry_time(self, u, v, d, t, w=1.0):
        """Earliest time >= t at which one more person (of size w person-equivalents) can enter u->v."""
        s = self.slot(t)
        c = self.cap(d)
        while True:
            if s * C.SLOT_S > self.now + self.horizon:   # beyond horizon: not predicted
                return max(t, s * C.SLOT_S)
            # a person always fits an EMPTY slot, even one larger than the slot (e.g. a bed on a narrow door)
            b = self.book[(u, v, s)]
            if b + w <= c or b == 0:
                return max(t, s * C.SLOT_S)
            s += 1

    def reserve(self, u, v, t, w=1.0):
        self.book[(u, v, self.slot(t))] += w


def _walk_time(d, speed_mult=1.0):
    return d["length_m"] / (d["free_speed"] * speed_mult * smoke_speed_factor(d.get("hazard", 0.0)))


def _leg(d, speed_mult=1.0, walk=None):
    """Generalised time of walking an edge: time x hazard penalty, plus a fixed penalty for
    crossing an exposed edge so waiting in a safe queue is preferred over walking through smoke.
    walk: a learned prediction of the walking time under current crowding (replaces free-flow time)."""
    h = d.get("hazard", 0.0)
    w = C.SAFETY_WEIGHT
    extra = w * C.HAZARD_DETOUR_PENALTY_S if h >= C.HAZARD_EXPOSURE else 0.0
    t = walk / speed_mult if walk is not None else _walk_time(d, speed_mult)
    return t * (1.0 + w * (_hazard_penalty(h) - 1.0)) + extra


def earliest_route(G, res, source, t0, fire_active, step_free, speed_mult=1.0, w=1.0):
    """
    Time-dependent Dijkstra on 'generalised time' (walking time x hazard penalty + waiting).
    Returns (path, arrival_cost, [(u, v, entry_time), ...]) or (None, inf, []).
    """
    exit_set = {e for e in exits(G) if G.nodes[e].get("hazard", 0.0) < C.HAZARD_IMPASSABLE}
    best = {source: t0}
    prev = {}
    entry = {}
    pq = [(t0, source)]
    while pq:
        t, u = heapq.heappop(pq)
        if t > best.get(u, math.inf):
            continue
        if u in exit_set:
            path, legs, n = [u], [], u
            while n != source:
                p = prev[n]
                legs.append((p, n, entry[n]))
                path.append(p)
                n = p
            return path[::-1], t, legs[::-1]
        for v in G[u]:
            d = G.edges[u, v]
            if not _allowed(d, fire_active, step_free):
                continue
            te = res.entry_time(u, v, d, t, w)
            tv = te + _leg(d, speed_mult, res.walk.get((u, v)))
            if tv < best.get(v, math.inf):
                best[v] = tv
                prev[v] = u
                entry[v] = te
                heapq.heappush(pq, (tv, v))
    return None, math.inf, []


def path_cost(G, res, path, t0, fire_active=False, step_free=False, speed_mult=1.0, w=1.0):
    """Predicted generalised arrival time of a FIXED path under the current bookings.
    A path that now crosses a forbidden edge costs infinity (forces a switch)."""
    t = t0
    for u, v in zip(path, path[1:]):
        d = G.edges[u, v]
        if not _allowed(d, fire_active, step_free):
            return math.inf
        t = res.entry_time(u, v, d, t, w) + _leg(d, speed_mult, res.walk.get((u, v)))
    return t


def book_path(G, res, path, t0, speed_mult=1.0, w=1.0):
    t = t0
    for u, v in zip(path, path[1:]):
        d = G.edges[u, v]
        te = res.entry_time(u, v, d, t, w)
        res.reserve(u, v, te, w)
        t = te + _leg(d, speed_mult, res.walk.get((u, v)))
    return t


def plan_all(G, agents, fire_active, now=0.0, horizon=math.inf, walk=None):
    """One-shot CCRP: plan agents in order of when they are ready to move."""
    res = Reservations(G, now, horizon, walk)
    order = sorted(agents, key=lambda a: (a.start, a.id))
    out = {}
    for a in order:
        w = getattr(a, "size", 1.0)
        path, _cost, legs = earliest_route(G, res, a.origin, a.start, fire_active, a.step_free, a.speed_mult, w)
        if path is None:
            continue
        for u, v, te in legs:
            res.reserve(u, v, te, w)
        out[a.id] = path
    return out


def replan(G, agents, moving, standing, now, fire_active, horizon, switch_gain, walk=None):
    """
    Rolling-horizon re-plan at time `now`.
      1. Book what is already committed: people walking an edge keep their remaining route.
      2. Re-plan people standing at a node (queued, or not yet started), earliest-ready first.
         A person switches only if the new route is predicted >= switch_gain faster, and
         never by stepping back to the node they just came from.
    Returns number of route changes.
    """
    res = Reservations(G, now, horizon, walk)
    for a in moving:
        u, v = a.path[a.idx], a.path[a.idx + 1]
        d = G.edges[u, v]
        remaining = max(0.0, d["length_m"] - a.pos) / (d["free_speed"] * a.speed_mult
                                                         * smoke_speed_factor(d.get("hazard", 0.0)))
        book_path(G, res, a.path[a.idx + 1:], now + remaining, a.speed_mult, getattr(a, "size", 1.0))

    # people who ignore guidance are not re-planned, but the planner sees where they are
    # heading and books their capacity first
    for a in standing:
        if not getattr(a, "compliant", True) and C.PLANNER_BOOK_NONCOMPLIANT:
            book_path(G, res, a.path[a.idx:], max(a.start, now), a.speed_mult, getattr(a, "size", 1.0))

    changes = 0
    for a in sorted((a for a in standing if getattr(a, "compliant", True)),
                    key=lambda a: (max(a.start, now), a.id)):
        node = a.path[a.idx]
        t0 = max(a.start, now)
        cur = a.path[a.idx:]
        w = getattr(a, "size", 1.0)
        cur_cost = path_cost(G, res, cur, t0, fire_active, a.step_free, a.speed_mult, w)
        new, new_cost, _ = earliest_route(G, res, node, t0, fire_active, a.step_free, a.speed_mult, w)
        back = a.idx > 0 and new is not None and len(new) > 1 and new[1] == a.path[a.idx - 1]
        gain = math.inf if cur_cost == math.inf else (cur_cost - new_cost) / max(1e-9, cur_cost - t0)
        if new is not None and new != cur and not back and gain >= switch_gain:
            a.path = a.path[:a.idx] + new
            a.exit = new[-1]
            a.changes += 1
            changes += 1
            book_path(G, res, new, t0, a.speed_mult, w)
        elif cur_cost < math.inf:
            book_path(G, res, cur, t0, a.speed_mult, w)
    return changes


def _leg_time(d, speed_mult=1.0):
    """Plain walking time of an edge (no hazard penalty) - for time-to-exit estimates."""
    return _walk_time(d, speed_mult)


def _book_time(G, res, path, t0, speed_mult, w, book=True):
    t = t0
    for u, v in zip(path, path[1:]):
        d = G.edges[u, v]
        te = res.entry_time(u, v, d, t, w)
        if book:
            res.reserve(u, v, te, w)
        t = te + _leg_time(d, speed_mult)
    return t


def predict_arrivals(G, agents, moving, standing, now, horizon=math.inf):
    """Time-to-exit for everyone standing at a node, from the SAME capacity bookings the planner uses:
    people already walking and people who ignore guidance are booked first, then everyone else in order of
    readiness, each on the route they currently have (people walking an edge get an estimate too). Unlike a sum of per-corridor predictions, this accounts
    for the crowd that will arrive at each door and stair BEFORE the person does. Returns {agent id: seconds}."""
    res = Reservations(G, now, horizon)
    out = {}
    for a in moving:
        u, v = a.path[a.idx], a.path[a.idx + 1]
        d = G.edges[u, v]
        rem = max(0.0, d["length_m"] - a.pos) / (d["free_speed"] * a.speed_mult * smoke_speed_factor(d.get("hazard", 0.0)))
        out[a.id] = _book_time(G, res, a.path[a.idx + 1:], now + rem, a.speed_mult, getattr(a, "size", 1.0)) - now
    for a in sorted(standing, key=lambda a: (not getattr(a, "compliant", True), max(a.start, now), a.id)):
        t0 = max(a.start, now)
        out[a.id] = _book_time(G, res, a.path[a.idx:], t0, a.speed_mult, getattr(a, "size", 1.0)) - now
    return out

