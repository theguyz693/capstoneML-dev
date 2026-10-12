"""
Session 5 checks: replay frames, kiosk signs, weak-spot candidates, research API.
    python tests/test_evac_session5.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evac.building import load_campus
from evac.hazard import apply_hazard
from evac.routing import next_hop_signs
from evac.simulator import run
from evac.weakspots import candidates

G0 = load_campus()


def test_replay_frames_recorded():
    m, _ = run(G0.copy(), {"n_agents": 40, "spawn": "floor", "floor": 2, "fires": [], "record_tracks": 2}, "ccrp", 0)
    assert len(m["frames"]) > 10 and m["frames"][0]["pts"]


def test_every_reachable_node_has_a_sign():
    G = apply_hazard(G0.copy(), [{"node": "j19", "radius": 400, "intensity": 1.0}])
    arrows = next_hop_signs(G)
    assert len(arrows) > 400
    assert all(G.has_edge(n, nxt) for n, nxt in arrows.items() if nxt)


def test_signs_avoid_fire():
    G = apply_hazard(G0.copy(), [{"node": "j6", "radius": 500, "intensity": 1.0}])
    arrows = next_hop_signs(G)
    assert all(G.edges[n, nxt]["hazard"] < 0.92 for n, nxt in arrows.items() if nxt)


def test_weakspot_candidates_cover_every_stairwell_and_exit():
    c = candidates(G0)
    assert sum(k == "stair_shaft" for k, _ in c) == 4
    assert sum(k == "exit_door" for k, _ in c) == 4


def test_research_api():
    try:
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
    except ImportError:
        print("  (fastapi not installed - API test skipped)")
        return
    from evac.api import router
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    assert len(c.get("/api/research/building").json()["nodes"]) == 452
    r = c.post("/api/research/run", json={"scenario": "lecture_hall_f1_120", "methods": ["ccrp"]}).json()
    assert r["results"]["ccrp"]["frames"]
    k = c.post("/api/research/signs", json={"fire_node": "j19", "kiosk_node": "f3_sem hall 1"}).json()["kiosk"]
    assert k["path"][-1] == k["exit"]


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} checks passed")
