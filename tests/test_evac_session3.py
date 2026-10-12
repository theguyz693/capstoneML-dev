"""
Session 3 checks: capacity-aware planning.
    python tests/test_evac_session3.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evac.building import load_campus
from evac.simulator import run

G0 = load_campus()
HALL = {"n_agents": 120, "spawn": "single", "origin": "f1_sem hall 1", "fires": []}
FIRE_250 = {"n_agents": 250, "spawn": "building", "step_free_frac": 0.05,
            "fires": [{"node": "j19", "radius": 780, "intensity": 1.0}]}


def m(sc, method, seed=0, **kw):
    return run(G0.copy(), {**sc, **kw}, method, seed)[0]


def test_ccrp_beats_plain_routing_when_crowded():
    assert m(HALL, "ccrp")["evac_time_last_s"] < 0.8 * m(HALL, "safe_shortest")["evac_time_last_s"]


def test_lookahead_flips_far_less_than_reactive():
    assert m(HALL, "lookahead")["route_changes"] * 20 < m(HALL, "reactive")["route_changes"]


def test_planner_is_not_less_safe_than_safe_routing():
    for seed in (0, 1):
        assert m(FIRE_250, "ccrp", seed)["exposed_agents"] <= m(FIRE_250, "safe_shortest", seed)["exposed_agents"]


def test_planner_reproducible():
    a, b = m(HALL, "lookahead", 4), m(HALL, "lookahead", 4)
    a.pop("compute_s"); b.pop("compute_s")
    assert a == b


def test_optimum_is_a_lower_bound():
    lb = m(HALL, "optimum")["evac_time_last_s"]
    for method in ("safe_shortest", "reactive", "ccrp", "lookahead"):
        assert lb <= m(HALL, method)["evac_time_last_s"]


def test_short_horizon_is_worse_than_long():
    assert m(HALL, "lookahead", horizon_s=30)["evac_time_last_s"] > m(HALL, "lookahead", horizon_s=300)["evac_time_last_s"]


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} checks passed")
