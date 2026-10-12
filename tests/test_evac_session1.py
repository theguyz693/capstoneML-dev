"""
Session 1 checks. Run from the repo root:
    python -m pytest tests -q        (or)        python tests/test_evac_session1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evac.building import load_campus, exits, unreachable_nodes
from evac.hazard import apply_hazard
from evac.routing import route_table
from evac.simulator import run

FIRE = [{"node": "j19", "radius": 780, "intensity": 1.0}]
G0 = load_campus()


def test_graph_size():
    assert G0.number_of_nodes() == 452 and G0.number_of_edges() == 636
    assert len(exits(G0)) == 4


def test_same_seed_same_result():
    sc = {"n_agents": 80, "spawn": "building", "step_free_frac": 0.1, "fires": FIRE}
    a, _ = run(G0.copy(), sc, "safe_shortest", seed=7)
    b, _ = run(G0.copy(), sc, "safe_shortest", seed=7)
    a.pop("compute_s"); b.pop("compute_s")
    assert a == b


def _uses(path, G, etype):
    return any(G.edges[u, v]["etype"] == etype for u, v in zip(path, path[1:]))


def test_no_lifts_in_fire():
    G = apply_hazard(G0.copy(), FIRE)
    for step_free in (False, True):
        for p in route_table(G, "safe_shortest", fire_active=True, step_free=step_free).values():
            assert not _uses(p, G, "lift")


def test_lifts_only_for_step_free_without_fire():
    G = apply_hazard(G0.copy(), [])
    for p in route_table(G, "safe_shortest", fire_active=False, step_free=False).values():
        assert not _uses(p, G, "lift")


def test_step_free_never_uses_stairs():
    for fires in ([], FIRE):
        G = apply_hazard(G0.copy(), fires)
        for p in route_table(G, "safe_shortest", fire_active=bool(fires), step_free=True).values():
            assert not _uses(p, G, "stair")


def test_step_free_upper_floor_in_fire_needs_assistance():
    sc = {"n_agents": 40, "spawn": "floor", "floor": 4, "step_free_frac": 1.0, "fires": FIRE}
    m, _ = run(G0.copy(), sc, "safe_shortest", seed=1)
    assert m["needs_assistance"] == 40 and m["evacuated"] == 0


def test_smoke_rises_through_stair_shaft():
    G = apply_hazard(G0.copy(), [{"node": "stairs 1", "radius": 300, "intensity": 1.0}])
    assert G.nodes["f1_stairs 1"]["hazard"] > 0
    assert G.nodes["f3_stairs 1"]["hazard"] < G.nodes["f1_stairs 1"]["hazard"]


def test_never_routed_through_impassable_edge():
    G = apply_hazard(G0.copy(), FIRE)
    for method in ("nearest", "safe_shortest"):
        for p in route_table(G, method, True, False).values():
            assert all(G.edges[u, v]["hazard"] < 0.92 for u, v in zip(p, p[1:]))


def test_known_isolated_node_is_reported():
    assert unreachable_nodes(G0) == ["electrical room 2"]


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} checks passed")
