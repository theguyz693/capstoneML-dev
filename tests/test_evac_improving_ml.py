"""
Checks for the Improving-ML session: calibrated parameters, smoke slowdown, ML v3 route + time-to-exit.
    python tests/test_evac_improving_ml.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evac import config as C
from evac import physics
from evac.building import build_graph, load_building


def test_param_sets_switch():
    assert C.use_params("sfpe") == "sfpe"
    assert C.FREE_WALK_SPEED == 1.25 and C.SPECIFIC_FLOW["exit_door"] == 1.3 and C.DOOR_APPROACH_WIDTH_M is None
    name = C.use_params("juelich")
    if name == "juelich":           # calibrated_params.json present
        assert 1.2 < C.FREE_WALK_SPEED < 1.6
        assert 1.5 < C.SPECIFIC_FLOW["exit_door"] < 2.5
        assert C.DOOR_APPROACH_WIDTH_M == 2.0
    assert C.use_params("sfpe") == "sfpe"     # and back: no state leaks
    assert C.SPECIFIC_FLOW["exit_door"] == 1.3
    C.use_params("juelich")


def test_smoke_slows_walking():
    assert physics.smoke_speed_factor(0.0) == 1.0
    assert 0.7 < physics.smoke_speed_factor(0.3) < 0.8           # Ks = 3 /m -> ~76 %
    assert physics.smoke_speed_factor(1.0) >= C.SMOKE_MIN_FRACTION
    assert physics.smoke_speed_factor(0.9) < physics.smoke_speed_factor(0.5) < physics.smoke_speed_factor(0.1)


def test_smoke_makes_crossing_slower_in_simulator():
    from evac.simulator import run
    U = 1.0 / C.METERS_PER_UNIT
    G = build_graph({"room": (0, 0, 0), "j1": (20 * U, 0, 0), "test exit": (21 * U, 0, 0)},
                    [("room", "j1"), ("j1", "test exit")], kinds={"j1": "junction"})
    sc = {"spawn": "single", "origin": "room", "n_agents": 1, "pre_movement_max_s": 0.0, "speed_spread": 0.0}
    clear, _ = run(G.copy(), {**sc, "fires": []}, "nearest", 0)
    smoky, _ = run(G.copy(), {**sc, "fires": [{"node": "j1", "radius": 30 * U, "intensity": 0.6}]}, "nearest", 0)
    assert smoky["evac_time_last_s"] > clear["evac_time_last_s"] * 1.2


def test_calibrated_door_passes_measured_flow():
    """A 1 m door in the calibrated simulator should pass ~1.6-2.2 persons/s (Juelich/Seyfried ~1.9)."""
    if C.use_params("juelich") != "juelich":
        return
    from evac.calibration import sim_bottleneck
    flow, _ = sim_bottleneck(1.0, 120)
    assert 1.6 < flow < 2.2, flow


def test_snapshot_and_route_eta():
    import json
    from evac.simulator import run
    from evac.learned_planner import route_eta
    if not os.path.exists(os.path.join(ROOT, "models", "delay_lgbm_v3.joblib")):
        return
    G = load_building("campus")
    sc = json.load(open(os.path.join(ROOT, "evac", "scenarios.json"), encoding="utf-8"))["floor3_full_200"]
    m, _ = run(G.copy(), {**sc, "stop_at_s": 40.0}, "lookahead", 0)
    st = m["state"]
    assert st["inside"] > 0 and sum(st["qnode"].values()) > 0
    old = C.DELAY_MODEL
    C.DELAY_MODEL = "v3"
    try:
        empty = route_eta(G, "classroom g17", False)
        busy = route_eta(G, "f3_lab 3" if "f3_lab 3" in G else sorted(n for n in G if G.nodes[n]["floor"] == 3
                                                                       and G.nodes[n]["kind"] == "room")[0],
                         False, state=st)
    finally:
        C.DELAY_MODEL = old
    assert empty["path"][0] == "classroom g17" and empty["eta_lo_s"] <= empty["eta_s"] <= empty["eta_hi_s"]
    assert busy["path"] and busy["eta_s"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
