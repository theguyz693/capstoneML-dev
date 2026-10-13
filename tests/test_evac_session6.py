"""
Session 6 checks: ML-in-the-loop planner, learned walk times, GA exit plans, sensitivity context
manager, ablation switches, conformal intervals.
    python tests/test_evac_session6.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evac import config as C
from evac import physics
from evac.building import exits, load_campus
from evac.simulator import run

G0 = load_campus()
SC = json.load(open(os.path.join(ROOT, "evac", "scenarios.json"), encoding="utf-8"))
HAS_MODEL = os.path.exists(os.path.join(ROOT, "models", "delay_lgbm.joblib"))
SMALL = {"n_agents": 60, "spawn": "floor", "floor": 3, "fires": []}


def test_learned_planner_runs_and_evacuates_everyone():
    kinds = ["pred_formula"] + (["pred_ml", "pred_ml_upper", "lookahead_ml"] if HAS_MODEL else [])
    for k in kinds:
        m, _ = run(G0.copy(), SMALL, k, 0)
        assert m["evacuated"] == 60 and m["timed_out"] == 0, k


def test_learned_planner_respects_fire():
    sc = {**SMALL, "fires": [{"node": "f3_stairs 1", "radius": 200, "intensity": 1.0}]}
    for k in ["pred_formula"] + (["pred_ml"] if HAS_MODEL else []):
        m, _ = run(G0.copy(), sc, k, 1)
        assert m["evacuated"] == 60 and m["exposed_agents"] == 0, k


def test_walk_predictions_positive():
    if not HAS_MODEL:
        return
    from evac.learned_planner import EdgeIndex, walk_predictions
    idx = EdgeIndex(G0)
    w = walk_predictions(G0, idx, [], {}, {})
    assert len(w) == 2 * G0.number_of_edges() and min(w.values()) > 0


def test_same_seed_is_deterministic_for_learned_methods():
    a = run(G0.copy(), SMALL, "pred_formula", 3)[0]["evac_time_last_s"]
    b = run(G0.copy(), SMALL, "pred_formula", 3)[0]["evac_time_last_s"]
    assert a == b


def test_exit_plan_hook_sends_people_to_planned_exit():
    ex = sorted(exits(G0))
    m0, recs0 = run(G0.copy(), SMALL, "safe_shortest", 0)
    plan = {f"{r['origin']}#{s}": ex[0] for r in recs0 for s in range(4)}
    m, recs = run(G0.copy(), {**SMALL, "exit_plan": plan}, "safe_shortest", 0)
    assert all(r["exit"] == ex[0] for r in recs if r["t_out"] is not None)


def test_ga_never_worse_than_its_seed_plan():
    from evac.ga import optimise
    sc = {"n_agents": 40, "spawn": "floor", "floor": 3, "fires": []}
    base = run(G0.copy(), sc, "nearest", 0)[0]["evac_time_last_s"]
    m, plan = optimise(sc, 0, pop=4, gens=2, G0=G0, verbose=False)
    assert m["evac_time_last_s"] <= base + 1e-9 and plan


def test_sensitivity_context_restores_config():
    from evac.sensitivity import PARAMS, Assumptions
    before = (C.FREE_WALK_SPEED, dict(C.SPECIFIC_FLOW), dict(C.WIDTH_M), physics.PEAK_FLOW_PER_SPEED)
    vals = {p[0]: p[2] for p in PARAMS}
    with Assumptions(vals):
        assert C.FREE_WALK_SPEED == vals["FREE_WALK_SPEED"]
        assert C.SPECIFIC_FLOW["stair"] == vals["STAIR_FLOW"]
    assert (C.FREE_WALK_SPEED, dict(C.SPECIFIC_FLOW), dict(C.WIDTH_M), physics.PEAK_FLOW_PER_SPEED) == before


def test_ablation_switches_change_behaviour_and_default_is_on():
    assert C.PLANNER_USE_SUSTAINABLE_CAP and C.PLANNER_BOOK_NONCOMPLIANT
    sc = SC["floor3_200_compliance60_herding"]
    full = run(G0.copy(), sc, "lookahead", 0)[0]
    C.PLANNER_BOOK_NONCOMPLIANT = False
    try:
        abl = run(G0.copy(), sc, "lookahead", 0)[0]
    finally:
        C.PLANNER_BOOK_NONCOMPLIANT = True
    assert abl["evacuated"] == full["evacuated"]
    assert abl["evac_time_last_s"] != full["evac_time_last_s"] or abl["route_changes"] != full["route_changes"]


def test_slot_booking_is_robust_to_float_slot_length():
    # regression: 3 * 4.1 = 12.2999... once mapped to slot 2 and overbooked capacity
    from evac.planner import Reservations
    saved = C.SLOT_S
    try:
        C.SLOT_S = 4.1
        assert Reservations.slot(3 * C.SLOT_S) == 3
        res = Reservations(G0)
        u, v = next(iter(G0.edges()))
        d = G0.edges[u, v]
        for _ in range(res.cap(d) * 3):
            res.reserve(u, v, res.entry_time(u, v, d, 0.0))
        assert max(res.book.values()) <= res.cap(d)
    finally:
        C.SLOT_S = saved


def test_model_meta_matches_features():
    p = os.path.join(ROOT, "models", "delay_model_meta.json")
    if not os.path.exists(p):
        return
    meta = json.load(open(p))
    from evac.delay_model import FEATURES
    assert meta["features"] == FEATURES and meta["alpha"] == 0.1


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} checks passed")
