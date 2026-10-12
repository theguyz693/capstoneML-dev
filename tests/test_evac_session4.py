"""
Session 4 checks: behaviour, spreading fire, delay logging.
    python tests/test_evac_session4.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evac.building import load_campus
from evac.hazard import apply_hazard
from evac.simulator import run, spawn
import random

G0 = load_campus()
F3 = {"n_agents": 200, "spawn": "floor", "floor": 3, "fires": []}
SPREAD = {**F3, "fires": [{"node": "stairs 1", "radius": 150, "intensity": 1.0,
                           "spread_m_per_s": 0.5, "max_radius": 900}]}


def m(sc, method, seed=0):
    return run(G0.copy(), sc, method, seed)[0]


def test_same_people_whatever_the_compliance():
    a = spawn(G0, {**F3, "compliance": 1.0, "_seed": 3}, random.Random(3))
    b = spawn(G0, {**F3, "compliance": 0.4, "_seed": 3}, random.Random(3))
    assert [(x.origin, x.start) for x in a] == [(x.origin, x.start) for x in b]
    assert sum(x.compliant for x in b) < len(b)


def test_lookahead_beats_one_shot_plan_when_people_disobey():
    sc = {**F3, "compliance": 0.6}
    assert m(sc, "lookahead")["evac_time_last_s"] < m(sc, "ccrp")["evac_time_last_s"]


def test_fire_grows_over_time():
    G = apply_hazard(G0.copy(), SPREAD["fires"], 0.0)
    early = sum(d["hazard"] > 0 for _, d in G.nodes(data=True))
    G = apply_hazard(G, SPREAD["fires"], 120.0)
    late = sum(d["hazard"] > 0 for _, d in G.nodes(data=True))
    assert late > early


def test_smoke_climbs_shaft_with_delay():
    G = apply_hazard(G0.copy(), [{"node": "stairs 1", "radius": 300, "intensity": 1.0,
                                  "spread_m_per_s": 0.5}], 10.0)
    assert G.nodes["f3_stairs 1"]["hazard"] == 0          # not yet at floor 3 after 10 s
    G = apply_hazard(G, [{"node": "stairs 1", "radius": 300, "intensity": 1.0, "spread_m_per_s": 0.5}], 200.0)
    assert G.nodes["f3_stairs 1"]["hazard"] > 0


def test_replanning_cuts_exposure_in_spreading_fire():
    assert m(SPREAD, "lookahead")["exposed_agents"] < m(SPREAD, "safe_shortest")["exposed_agents"]


def test_edge_logging_produces_labels():
    met = m({**F3, "n_agents": 60, "log_edges": True}, "safe_shortest")
    rows = met["edge_rows"]
    assert len(rows) > 100 and all(r["target_time_s"] > 0 for r in rows)


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} checks passed")
