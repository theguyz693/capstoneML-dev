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


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} checks passed")
