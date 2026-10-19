"""
Session 7 checks: imperfect knowledge, printed plans, fire compartments, refuge / defend in place,
staff-limited bed movement, model versions.
    python tests/test_evac_session7.py
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evac import config as C
from evac.building import exits, load_campus
from evac.fixed_plan import assign, make_plan
from evac.hazard import node_hazards
from evac.hospital import load_hospital, scenarios
from evac.simulator import run, spawn

G = load_campus()
H = load_hospital()
SC = json.load(open(os.path.join(ROOT, "evac", "scenarios.json"), encoding="utf-8"))
HS = scenarios()
F3 = SC["floor3_full_200"]


def test_perfect_knowledge_is_unchanged():
    from evac import config as C
    C.use_params("sfpe")                     # regression value from before the calibration (handbook parameters)
    try:
        assert run(load_campus(), F3, "lookahead", 0)[0]["evac_time_last_s"] == 213.0
    finally:
        C.use_params("juelich")


def test_planner_that_sees_nobody_equals_shortest_path():
    a = run(G.copy(), {**F3, "sense_miss": 1.0}, "lookahead", 0)[0]["evac_time_last_s"]
    b = run(G.copy(), F3, "safe_shortest", 0)[0]["evac_time_last_s"]
    assert a == b


def test_noisy_sensing_is_deterministic_and_everyone_gets_out():
    for kw in ({"sense_miss": 0.3}, {"sense_false": 0.5}, {"sense_dead_floor": 3}):
        m1 = run(G.copy(), {**F3, **kw}, "lookahead", 1)[0]
        m2 = run(G.copy(), {**F3, **kw}, "lookahead", 1)[0]
        assert m1["evac_time_last_s"] == m2["evac_time_last_s"] and m1["evacuated"] == 200, kw


def test_printed_plan_is_valid():
    plan = make_plan(G, F3, samples=2)
    ex = set(exits(G))
    for key, opts in plan.items():
        assert abs(sum(s for _, s in opts) - 1.0) < 1e-3
        for p, _ in opts:
            assert p[0] == key.split("|")[0] and p[-1] in ex
            assert all(G.has_edge(u, v) for u, v in zip(p, p[1:]))
    m = run(G.copy(), {**F3, "route_plan": plan, "route_plan_mode": "split"}, "fixed_plan", 0)[0]
    assert m["evacuated"] == 200


def test_printed_plan_never_sends_people_through_smoke():
    from evac.hazard import apply_hazard
    plan = make_plan(G, F3, samples=2)
    sc = SC["floor3_full_200_fire_stairs1"]
    Gf = apply_hazard(G.copy(), sc["fires"])
    agents = spawn(Gf, sc, random.Random(0))
    from evac.routing import route_table
    tab = route_table(Gf, "safe_shortest", True, False)
    for a in agents:
        a.path = tab.get(a.origin)
    assign(Gf, agents, plan, "split", True)
    for a in agents:
        assert all(Gf.edges[u, v]["hazard"] < C.HAZARD_EXPOSURE for u, v in zip(a.path, a.path[1:]))


def test_fire_doors_contain_the_fire():
    f = {"node": "f2_store room", "radius": 2000, "intensity": 1.0}
    open_h = node_hazards(H, [f])
    shut_h = node_hazards(H, [{**f, "contained": True}])
    east = [n for n in H if H.nodes[n].get("compartment") == "f2-E"]
    assert max(open_h[n] for n in east) > C.HAZARD_EXPOSURE
    assert max(shut_h[n] for n in east) < C.HAZARD_EXPOSURE


def test_refuge_and_defend_in_place():
    base = HS["hosp_night_wards_fire"]
    sc = {**base, "fires": [{**base["fires"][0], "contained": True}],
          "refuge": [{"compartment": "f2-W", "target": "f2_j8"}], "in_place": "rest"}
    m, recs = run(H.copy(), sc, "safe_shortest", 0)
    moved = [r for r in recs if r["state"] != "in_place"]
    assert m["in_place"] > 0 and moved
    assert all(H.nodes[r["origin"]]["compartment"] == "f2-W" for r in moved)
    assert all(r["exit"] == "f2_j8" for r in moved if r["t_out"] is not None)


def test_fewer_staff_means_beds_wait_longer():
    base = HS["hosp_night_wards_fire"]
    beds = {}
    for staff in (None, {"2": 2, "3": 2}):
        sc = {**base, "fires": []}
        if staff:
            sc["staff"] = staff
        _m, recs = run(H.copy(), sc, "safe_shortest", 0)
        beds[bool(staff)] = max(r["t_out"] for r in recs if r["profile"] == "bed" and r["t_out"] is not None)
    assert beds[True] > beds[False] * 1.5


def test_model_versions():
    from evac import learned_planner as L
    if not os.path.exists(os.path.join(ROOT, "models", "delay_lgbm_v2.joblib")):
        return
    idx = L.EdgeIndex(H)
    saved = C.DELAY_MODEL
    try:
        out = {}
        for v in ("v1", "v2"):
            C.DELAY_MODEL = v
            out[v] = L.walk_predictions(H, idx, [], {}, {})
        assert out["v1"] != out["v2"] and min(out["v2"].values()) > 0
    finally:
        C.DELAY_MODEL = saved


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} checks passed")
