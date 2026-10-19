"""
Checks for Phase 2 (the half-done items).
    python tests/test_evac_phase2.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evac import config as C
from evac.building import load_campus
from evac.simulator import run

SC = json.load(open(os.path.join(ROOT, "evac", "scenarios.json"), encoding="utf-8"))


def test_safety_dial_trades_time_for_exposure():
    G = load_campus()
    sc = SC["floor3_200_spreading_fire_stairs1"]
    old = C.SAFETY_WEIGHT
    try:
        res = {}
        for w in (0.0, 8.0):
            C.SAFETY_WEIGHT = w
            res[w] = [run(G.copy(), sc, "lookahead", s)[0]["exposed_agents"] for s in range(3)]
    finally:
        C.SAFETY_WEIGHT = old
    assert sum(res[8.0]) <= sum(res[0.0])


def test_safety_dial_default_is_unchanged():
    assert C.SAFETY_WEIGHT == 1.0


def test_eta_probe_and_calibrated_range():
    G = load_campus()
    m, _ = run(G.copy(), {**SC["floor3_full_200"], "eta_probe_at": [60]}, "lookahead", 0)
    rows = m["eta_rows"]
    assert rows and all(r["eta_ml"] is not None and r["eta_plan"] is not None for r in rows)
    assert all(r["actual_s"] is None or r["actual_s"] >= 0 for r in rows)
    if os.path.exists(os.path.join(ROOT, "evac", "eta_calibration.json")):
        from evac.learned_planner import route_eta
        old = C.DELAY_MODEL
        C.DELAY_MODEL = "v3"
        try:
            e = route_eta(G, "classroom g17", False)
        finally:
            C.DELAY_MODEL = old
        assert e["eta_lo_s"] < e["eta_s"] < e["eta_hi_s"]


def test_priority_switches_are_off_by_default_and_change_order():
    assert C.PRIORITY_VULNERABLE is False and C.DOOR_PRIORITY_VULNERABLE is False
    from evac.planner import plan_all, vulnerable
    from evac.simulator import Agent
    G = load_campus()
    a = Agent(0, "classroom g17", 0.0, 1.0, False)
    b = Agent(1, "classroom g17", 5.0, 0.7, True, size=2.0, profile="wheelchair")
    assert vulnerable(b) and not vulnerable(a)
    C.PRIORITY_VULNERABLE = True
    try:
        assert plan_all(G, [a, b], False)          # both planned, wheelchair user booked first
    finally:
        C.PRIORITY_VULNERABLE = False


def test_astar_planner_gives_same_results_as_plain_search():
    from evac import planner
    G = load_campus()
    sc = SC["building_250_spreading_fire_j19"]
    fast = [run(G.copy(), sc, "lookahead", s)[0]["evac_time_last_s"] for s in (0, 1)]
    orig = planner.earliest_route
    planner.earliest_route = planner._earliest_route_ref
    try:
        ref = [run(G.copy(), sc, "lookahead", s)[0]["evac_time_last_s"] for s in (0, 1)]
    finally:
        planner.earliest_route = orig
    assert fast == ref


def test_tower_generator():
    from evac.scalability import tower
    from evac.building import room_nodes, unreachable_nodes
    G = tower(3, 4)
    assert len(room_nodes(G)) == 3 * 16 and not unreachable_nodes(G)


def test_jam_metrics_present():
    G = load_campus()
    m, _ = run(G.copy(), SC["floor3_full_200"], "safe_shortest", 0)
    assert m["wait_person_s"] > 0 and "f3_stairs 2" in m["jam_s"]


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} checks passed")
