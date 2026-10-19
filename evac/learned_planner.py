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
from .physics import weidmann_speed_factor, smoke_speed_factor, edge_density
from .routing import _allowed, _hazard_penalty

HERE = os.path.dirname(os.path.abspath(__file__))
MODELDIR = os.path.join(os.path.dirname(HERE), "models")
BATCH = 10
_models = {}

ET = ("stair", "room_door", "exit_door", "corridor")


def _load(kind):
    """kind: 'ml', 'ml_upper' (95 % quantile) or 'ml_lower' (5 %, v3 only); config.DELAY_MODEL picks the version
    (v1 = campus-trained, Session 6; v2 = both buildings, Session 7; v3 = calibrated simulator + random
    geometries, tested on real data - Improving ML, evac/ml_v3.py)."""
    ver = getattr(C, "DELAY_MODEL", "v1")
    key = (kind, ver)
    if key not in _models:
        import joblib
        suffix = "" if ver == "v1" else f"_{ver}"
        name = {"ml": f"delay_lgbm{suffix}.joblib", "ml_upper": f"delay_lgbm{suffix}_q95.joblib",
                "ml_lower": f"delay_lgbm{suffix}_q05.joblib"}[kind]
        path = os.path.join(MODELDIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing - run: python -m evac.ml_study (v1) / python -m evac.ml_transfer (v2)")
        _models[key] = joblib.load(path)
    return _models[key]


class EdgeIndex:
    """Directed edge list + static feature columns, built once per graph."""

    def __init__(self, G):
        self.edges = [(u, v) for u, v in G.edges()] + [(v, u) for u, v in G.edges()]
        self.pos = {e: i for i, e in enumerate(self.edges)}
        n = len(self.edges)
        self.static = np.zeros((n, 9))
        for i, (u, v) in enumerate(self.edges):
            d = G.edges[u, v]
            self.static[i, :5] = [d["etype"] == et for et in ET + ("ramp",)]
            self.static[i, 5:] = [d["length_m"], d["width_m"], d["eff_cap"], d["free_speed"]]


COLS = ["is_stair", "is_room_door", "is_exit_door", "is_corridor", "is_ramp", "length_m", "width_m", "eff_cap",
        "free_speed", "hazard", "queue_ahead", "queue_at_node", "on_edge", "density", "planned_users",
        "mover_speed", "mover_size"]


def _features(G, idx, queues, occ, planned):
    """One row per directed edge; every column any model version may use (named, see COLS)."""
    qa = defaultdict(int)
    for node, q in queues.items():
        for a in q:
            if a.idx < len(a.path) - 1:
                qa[(node, a.path[a.idx + 1])] += 1
    qnode = {u: len(q) for u, q in queues.items()}
    return features_from_counts(G, idx, qa, qnode, occ, planned)


def features_from_counts(G, idx, qa, qnode, occ, planned, mover_speed=1.0, mover_size=1.0):
    """Same rows from plain counts: qa {(u, v): people queued at u for u->v}, qnode {u: people at u},
    occ {sorted edge: people on it}, planned {(u, v): people whose route uses u->v}."""
    import pandas as pd
    n = len(idx.edges)
    dyn = np.zeros((n, 6))
    for i, (u, v) in enumerate(idx.edges):
        d = G.edges[u, v]
        k = (u, v) if u <= v else (v, u)
        dyn[i] = [d.get("hazard", 0.0), qa.get((u, v), 0), qnode.get(u, 0), occ.get(k, 0),
                  edge_density(d, occ.get(k, 0)), planned.get((u, v), 0)]
    mover = np.tile([mover_speed, mover_size], (n, 1))     # default: a standard person (speed x1, size 1)
    return pd.DataFrame(np.hstack([idx.static, dyn, mover]), columns=COLS)


V3_EXTRA = ["t_formula", "t_queue", "t_walk_free", "onedge_per_cap", "planned_per_cap"]


def formula_time(X):
    """Textbook queueing estimate (seconds): people queued ahead / capacity + length / speed at the current
    density and smoke. The physics part of the hybrid model."""
    f = np.array([weidmann_speed_factor(x) for x in X["density"]])
    s = np.array([smoke_speed_factor(h) for h in X["hazard"]])
    mv = X["mover_speed"].to_numpy() if "mover_speed" in X else 1.0
    return (X["queue_ahead"] / X["eff_cap"] + X["length_m"] / (X["free_speed"] * mv * f * s)).to_numpy()


def add_physics(X):
    """v3 inputs: the raw features + physics-derived ones (formula time, queue time, free walking time, people
    on the edge / planned per unit of capacity). Dimensionless-style inputs let the model carry over to widths,
    lengths and crowd sizes it has not seen."""
    X = X.copy()
    X["t_formula"] = formula_time(X)
    X["t_queue"] = X["queue_ahead"] / X["eff_cap"]
    X["t_walk_free"] = X["length_m"] / (X["free_speed"] * X["mover_speed"])
    X["onedge_per_cap"] = X["on_edge"] / X["eff_cap"]
    X["planned_per_cap"] = X["planned_users"] / X["eff_cap"]
    return X


def hybrid_predict(model, X):
    """v3 is a HYBRID model: it learns a correction factor on top of the physics formula,
    time = formula x exp(model(x)), so where data is thin it falls back towards the physics."""
    Xp = add_physics(X)
    return np.maximum(Xp["t_formula"].to_numpy(), 0.3) * np.exp(model.predict(Xp[list(model.feature_name_)]))


def _predict(kind, X):
    if kind == "formula":
        return formula_time(X)
    m = _load(kind)
    if getattr(C, "DELAY_MODEL", "v1") == "v3":
        return np.maximum(0.1, hybrid_predict(m, X))
    cols = list(getattr(m, "feature_name_", None) or __import__("evac.delay_model", fromlist=["FEATURES"]).FEATURES)
    return np.maximum(0.1, m.predict(X[cols]))


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
    X["queue_ahead"] = 0.0
    X["queue_at_node"] = 0.0
    p = _predict("ml", X)
    return {e: float(p[i]) for i, e in enumerate(idx.edges)}


def _real_margin():
    """Widening factor for the 5-95 % range, from conformal calibration on REAL crossings (evac/ml_v3.py)."""
    import json
    p = os.path.join(MODELDIR, f"delay_model_{getattr(C, 'DELAY_MODEL', 'v1')}_meta.json")
    try:
        with open(p, encoding="utf-8") as f:
            return float(math.exp(json.load(f).get("real_log_margin", 0.0)))
    except (OSError, ValueError):
        return 1.0


def route_eta(G, origin, fire_active, step_free=False, state=None, mover_speed=1.0, mover_size=1.0):
    """Navigation query for one person (dashboard 'Evac Route' and kiosk signs):
    route = Dijkstra on PREDICTED crossing times (learned model) x hazard penalty (+ exposure penalty), edges
    on fire removed (rule, not ML); time-to-exit = sum of predicted times along the route, with a range from
    the 5 % / 95 % quantile models. state: optional crowd snapshot from the simulator
    {"qa": {(u, v): n}, "qnode": {u: n}, "occ": {edge: n}, "planned": {(u, v): n}}.
    Note: the crowd is taken as it is NOW (it does not evolve along the route), so with heavy crowds the range
    is a rough guide - the capacity planner (lookahead) is what coordinates many people."""
    st = state or {}
    idx = EdgeIndex(G)
    X = features_from_counts(G, idx, st.get("qa", {}), st.get("qnode", {}), st.get("occ", {}),
                             st.get("planned", {}), mover_speed, mover_size)
    med = _predict("ml", X)
    try:
        lo, hi = np.minimum(_predict("ml_lower", X), med), np.maximum(_predict("ml_upper", X), med)
        lo, hi = lo / _real_margin(), hi * _real_margin()
    except (FileNotFoundError, KeyError):
        lo = hi = None
    _dist, tab = _table(G, idx, med, fire_active, step_free)
    path = tab.get(origin)
    if not path:
        return {"path": None}
    pos = [idx.pos[(u, v)] for u, v in zip(path, path[1:])]
    legs = [{"from": u, "to": v, "etype": G.edges[u, v]["etype"], "hazard": round(G.edges[u, v].get("hazard", 0.0), 2),
             "pred_s": round(float(med[i]), 1)} for (u, v), i in zip(zip(path, path[1:]), pos)]
    out = {"path": path, "exit": path[-1], "eta_s": round(float(med[pos].sum()), 1), "legs": legs,
           "model": getattr(C, "DELAY_MODEL", "v1")}
    if lo is not None:
        out["eta_lo_s"] = round(float(lo[pos].sum()), 1)
        out["eta_hi_s"] = round(float(hi[pos].sum()), 1)
    return out
