"""
Session 2 checks: verification tests pass, reactive method is reproducible and actually reroutes.
    python tests/test_evac_session2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evac import verification as V
from evac.building import load_campus
from evac.simulator import run

G0 = load_campus()
HALL = {"n_agents": 120, "spawn": "single", "origin": "f1_sem hall 1", "fires": []}


def test_free_walk():
    assert V.test_free_walk()["pass"]


def test_door_flow_linear_in_width():
    assert V.test_door_flow()["pass"]


def test_stairs_slower_than_corridor():
    assert V.test_stairs()["pass"]


def test_speed_density_follows_weidmann():
    assert V.test_fundamental_diagram()["pass"]


def test_reactive_reproducible():
    a, _ = run(G0.copy(), HALL, "reactive", seed=3)
    b, _ = run(G0.copy(), HALL, "reactive", seed=3)
    a.pop("compute_s"); b.pop("compute_s")
    assert a == b


def test_reactive_reroutes_under_congestion():
    m, _ = run(G0.copy(), HALL, "reactive", seed=0)
    s, _ = run(G0.copy(), HALL, "safe_shortest", seed=0)
    assert m["route_changes"] > 0 and s["route_changes"] == 0
    assert m["evac_time_last_s"] < s["evac_time_last_s"]


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} checks passed")
