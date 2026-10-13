"""
Checks for the published baselines (evac/literature.py) and the JuPedSim cross-check geometry.
    python tests/test_evac_literature.py
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evac.building import load_campus
from evac.literature import casper_plan
from evac.routing import route_table
from evac.simulator import run, spawn

SC = json.load(open(os.path.join(ROOT, "evac", "scenarios.json"), encoding="utf-8"))


def test_casper_one_route_per_room():
    G = load_campus()
    agents = spawn(G, {**SC["floor3_full_200"], "_seed": 0}, random.Random(0))
    plan = casper_plan(G, agents, fire_active=False)
    assert len(plan) == len(agents)
    by_room = {}
    for a in agents:
        by_room.setdefault(a.origin, set()).add(tuple(plan[a.id]))
    assert all(len(p) == 1 for p in by_room.values())          # one path per source
    assert all(plan[a.id][0] == a.origin for a in agents)


def test_casper_spreads_load_over_stairs():
    """Unlike shortest path, CASPER should not send all 200 floor-3 people down the same stairs."""
    G = load_campus()
    agents = spawn(G, {**SC["floor3_full_200"], "_seed": 0}, random.Random(0))
    plan = casper_plan(G, agents, False)
    tab = route_table(G, "safe_shortest", False, False)

    def stairs_used(paths):
        return {n.split()[-1] for p in paths for n in p if "stairs" in n and n.startswith("f3_")}
    assert len(stairs_used(plan.values())) > len(stairs_used(tab[a.origin] for a in agents))


def test_new_methods_run_and_are_repeatable():
    G = load_campus()
    for method in ("quickest", "casper"):
        m1, _ = run(G.copy(), SC["floor3_full_200"], method, 1)
        m2, _ = run(G.copy(), SC["floor3_full_200"], method, 1)
        assert m1["evacuated"] == m1["n_agents"] == 200, method
        assert m1["evac_time_last_s"] == m2["evac_time_last_s"], method


def test_quickest_switches_and_accounts_for_everyone():
    G = load_campus()
    m, _ = run(G.copy(), SC["floor3_200_spreading_fire_stairs1"], "quickest", 0)
    assert m["evacuated"] + m["needs_assistance"] + m["stranded"] + m["timed_out"] == m["n_agents"]
    m2, _ = run(G.copy(), SC["floor3_full_200"], "quickest", 0)
    assert m2["route_changes"] > 0                             # people do react to the jams


def test_jupedsim_geometry_is_connected():
    try:
        import jupedsim  # noqa: F401
        from shapely.geometry import Point
    except ImportError:
        print("  (skipped: jupedsim not installed)")
        return
    from evac.jupedsim_check import floor_geometry, floor_graph
    G = floor_graph()
    geo, _pockets, P = floor_geometry(G)
    assert geo.geom_type == "Polygon"
    for n in G:                                                 # every room and exit is reachable walkable area
        if G.nodes[n]["kind"] in ("room", "exit"):
            assert geo.buffer(0.01).contains(Point(P[n])), n


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} checks passed")
