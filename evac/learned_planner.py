"""
evac/learned_planner.py
ML in the loop: re-plan routes using PREDICTED edge delays (Session 6A).

Every REPLAN_S seconds the controller:
  1. measures the current state of every directed edge u->v: people queued at u for it,
     people at u, people on it, density, and how many people's CURRENT routes still use it
     ("planned users" - the forward-looking signal a simple queue formula does not have);
  2. predicts the seconds a newcomer would need to get through each edge, with one of:
        formula   queue_ahead / capacity + length / speed-at-density   (textbook queueing)
        ml        the LightGBM delay model trained on simulator data   (models/delay_lgbm.joblib)
        ml_upper  the model's 95% upper quantile - a cautious planner that plans for bad luck
  3. routes everyone still standing at a node by Dijkstra on those predicted delays
     (x hazard penalty, + the same hazard detour penalty as the capacity planner), switching a
     person only if the predicted gain is >= SWITCH_GAIN and never by walking back;
  4. after every BATCH switches it updates the planned-user counts and re-predicts, so the
     plan accounts for the people it has just moved (prevents everyone jumping to one stair).

Same planner, three predictors -> the difference in evacuation results is the value of the predictor.
"""

import math
import os
from collections import defaultdict

import networkx as nx
import numpy as np

from . import config as C
from .building import exits
from .physics import weidmann_speed_factor
from .routing import _allowed, _hazard_penalty

HERE = os.path.dirname(os.path.abspath(__file__))
MODELDIR = os.path.join(os.path.dirname(HERE), "models")
BATCH = 10
_models = {}

ET = ("stair", "room_door", "exit_door", "corridor")


def _load(kind):
    if kind not in _models:
        import joblib
        name = {"ml": "delay_lgbm.joblib", "ml_upper": "delay_lgbm_q95.joblib"}[kind]
        path = os.path.join(MODELDIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing - run: python -m evac.ml_study")
        _models[kind] = joblib.load(path)
    return _models[kind]


class EdgeIndex:
    """Directed edge list + static feature columns, built once per graph."""

    def __init__(self, G):
        self.edges = [(u, v) for u, v in G.edges()] + [(v, u) for u, v in G.edges()]
        self.pos = {e: i for i, e in enumerate(self.edges)}
        n = len(self.edges)
        self.static = np.zeros((n, 8))
        for i, (u, v) in enumerate(self.edges):
            d = G.edges[u, v]
            self.static[i, :4] = [d["etype"] == et for et in ET]
            self.static[i, 4:] = [d["length_m"], d["width_m"], d["eff_cap"], d["free_speed"]]


def _features(G, idx, queues, occ, planned):
    n = len(idx.edges)
    dyn = np.zeros((n, 6))
    qa = defaultdict(int)
    for node, q in queues.items():
        for a in q:
            if a.idx < len(a.path) - 1:
                qa[(node, a.path[a.idx + 1])] += 1
    for i, (u, v) in enumerate(idx.edges):
        d = G.edges[u, v]
        k = (u, v) if u <= v else (v, u)
        dyn[i] = [d.get("hazard", 0.0), qa.get((u, v), 0), len(queues.get(u, ())), occ.get(k, 0),
                  occ.get(k, 0) / (d["length_m"] * d["width_m"]), planned.get((u, v), 0)]
    # column order must match delay_model.FEATURES
    return np.hstack([idx.static, dyn])


def _predict(kind, X):
    if kind == "formula":
        length, width, cap, speed = X[:, 4], X[:, 5], X[:, 6], X[:, 7]
        dens = X[:, 12]
        f = np.array([weidmann_speed_factor(x) for x in dens])
        return X[:, 9] / cap + length / (speed * f)
    import pandas as pd
    from .delay_model import FEATURES
    return np.maximum(0.1, _load(kind).predict(pd.DataFrame(X, columns=FEATURES)))


def _planned(agents):
    p = defaultdict(int)
    for a in agents:
        if a.path is None or a.state in ("evacuated", "needs_assistance", "stranded"):
            continue
        for i in range(a.idx, len(a.path) - 1):
            p[(a.path[i], a.path[i + 1])] += 1
    return p


def _table(G, idx, pred, fire_active, step_free):
    ex = [e for e in exits(G) if G.nodes[e].get("hazard", 0.0) < C.HAZARD_IMPASSABLE]

    def w(u, v, d):              # search runs from the exits: the person walks v -> u
        if not _allowed(d, fire_active, step_free):
            return None
        h = d.get("hazard", 0.0)
        extra = C.HAZARD_DETOUR_PENALTY_S if h >= C.HAZARD_EXPOSURE else 0.0
        return pred[idx.pos[(v, u)]] * _hazard_penalty(h) + extra

    dist, paths = nx.multi_source_dijkstra(G, ex, weight=w)
    return dist, {n: list(reversed(p)) for n, p in paths.items()}


def _cost(G, idx, pred, path, fire_active, step_free):
    c = 0.0
    for u, v in zip(path, path[1:]):
        d = G.edges[u, v]
        if not _allowed(d, fire_active, step_free):
            return math.inf
        h = d.get("hazard", 0.0)
        c += pred[idx.pos[(u, v)]] * _hazard_penalty(h) + (C.HAZARD_DETOUR_PENALTY_S if h >= C.HAZARD_EXPOSURE else 0.0)
    return c


def replan(G, idx, kind, agents, queues, occ, waiting, fire_active, switch_gain=None):
    switch_gain = C.SWITCH_GAIN if switch_gain is None else switch_gain
    standing = [a for q in queues.values() for a in q if a.compliant] + [a for a in waiting if a.compliant]
    if not standing:
        return 0
    planned = _planned(agents)
    changes, since = 0, 0
    pred = _predict(kind, _features(G, idx, queues, occ, planned))
    tables = {}
    for a in sorted(standing, key=lambda a: (a.start, a.id)):
        if since >= BATCH:
            pred = _predict(kind, _features(G, idx, queues, occ, planned))
            tables, since = {}, 0
        if a.step_free not in tables:
            tables[a.step_free] = _table(G, idx, pred, fire_active, a.step_free)
        _dist, tab = tables[a.step_free]
        node = a.path[a.idx]
        new = tab.get(node)
        cur = a.path[a.idx:]
        if not new or new == cur:
            continue
        if a.idx > 0 and len(new) > 1 and new[1] == a.path[a.idx - 1]:
            continue
        c_cur = _cost(G, idx, pred, cur, fire_active, a.step_free)
        c_new = _cost(G, idx, pred, new, fire_active, a.step_free)
        if c_cur == math.inf or (c_cur - c_new) / max(1e-9, c_cur) >= switch_gain:
            for i in range(len(cur) - 1):
                planned[(cur[i], cur[i + 1])] -= 1
            for i in range(len(new) - 1):
                planned[(new[i], new[i + 1])] += 1
            a.path = a.path[:a.idx] + new
            a.exit = new[-1]
            a.changes += 1
            changes += 1
            since += 1
    return changes


def walk_predictions(G, idx, agents, queues, occ):
    """Learned WALKING time per directed edge for the capacity planner (lookahead_ml): the delay
    model with the queue features set to zero, because capacity booking already handles queues.
    What remains is the slowdown from crowding on the edge and from people planning to use it."""
    X = _features(G, idx, queues, occ, _planned(agents))
    X[:, 9] = 0.0      # queue_ahead
    X[:, 10] = 0.0     # queue_at_node
    p = _predict("ml", X)
    return {e: float(p[i]) for i, e in enumerate(idx.edges)}
