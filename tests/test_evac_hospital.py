"""
Hospital (second building) checks: import, scale, ramp, occupant profiles, capacity use, campus unchanged.
    python tests/test_evac_hospital.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evac import config as C
from evac.building import exits, load_campus, unreachable_nodes
from evac.hospital import DATA, load_hospital, scenarios
from evac.routing import route_table
from evac.simulator import run, spawn

H = load_hospital()
S = scenarios()


def test_hospital_graph_loads_and_everything_reaches_an_exit():
    assert H.number_of_nodes() == 124
    assert len(exits(H)) == 5
    assert unreachable_nodes(H) == []


def test_floors_are_consistently_scaled():
    doc = json.load(open(DATA, encoding="utf-8"))
    for v in doc["scale"].values():
        assert max(v["shaft_fit_residuals_m"]) < 0.5
    # building is ~54.7 m long on the drawings: corridor extent on each floor must be close to that
    for fl in range(4):
        xs = [H.nodes[n]["x"] * C.METERS_PER_UNIT for n in H if H.nodes[n]["floor"] == fl]
        assert 40 < max(xs) - min(xs) < 60, (fl, max(xs) - min(xs))


def test_ramp_is_a_step_free_route_and_stairs_are_not():
    assert sum(d["etype"] == "ramp" for *_, d in H.edges(data=True)) == 3
    tab = route_table(H, "safe_shortest", True, step_free=True)     # fire: lifts out
    for path in tab.values():
        if path:
            for u, v in zip(path, path[1:]):
                assert H.edges[u, v]["etype"] not in ("stair", "lift")
    assert tab.get("f3_general ward 1")                               # reachable via the ramp


def test_groups_spawn_counts_rooms_and_profiles():
    import random
    sc = S["hosp_day_peak"]
    agents = spawn(H, sc, random.Random(0))
    assert len(agents) == sum(g["n"] for g in sc["groups"])
    assert all(a.origin.startswith("g_") for a in agents[:150])
    beds = [a for a in agents if a.profile == "bed"]
    assert beds and all(a.step_free and a.size == 4.0 and a.origin.startswith(("f2_", "f3_")) for a in beds)


def test_bed_uses_more_capacity_than_a_walker():
    base = {"building": "hospital", "spawn": "groups", "fires": [], "speed_spread": 0.0}
    walk = dict(C.PROFILES["walk"])
    try:
        C.PROFILES["test_big"] = {**walk, "size": 4.0}
        C.PROFILES["test_one"] = {**walk, "size": 1.0}
        t = {}
        for p in ("test_one", "test_big"):
            sc = {**base, "groups": [{"where": "^g_waiting", "n": 60, "mix": {p: 1.0}}]}
            t[p] = run(H.copy(), sc, "safe_shortest", 0)[0]["evac_time_last_s"]
        assert t["test_big"] > t["test_one"] * 1.5, t
    finally:
        C.PROFILES.pop("test_big", None)
        C.PROFILES.pop("test_one", None)


def test_planner_handles_people_bigger_than_a_slot():
    from evac.planner import Reservations
    res = Reservations(H)
    u, v = next((u, v) for u, v, d in H.edges(data=True) if d["etype"] == "room_door")
    d = H.edges[u, v]
    t1 = res.entry_time(u, v, d, 0.0, w=50.0)          # far bigger than one slot: still gets an empty slot
    res.reserve(u, v, t1, 50.0)
    t2 = res.entry_time(u, v, d, 0.0, w=1.0)
    assert t1 == 0.0 and t2 > t1


def test_every_hospital_scenario_runs_with_the_planner():
    for name, sc in S.items():
        m, recs = run(H.copy(), sc, "lookahead", 0)
        assert m["evacuated"] + m["needs_assistance"] + m["stranded"] == m["n_agents"], name
        assert all("profile" in r for r in recs)


def test_ramp_is_a_weakspot_candidate_only_in_the_hospital():
    from evac.weakspots import candidates
    assert any(k == "ramp" for k, _ in candidates(H))
    assert not any(k == "ramp" for k, _ in candidates(load_campus()))


def test_campus_results_unchanged():
    sc = json.load(open(os.path.join(ROOT, "evac", "scenarios.json"), encoding="utf-8"))["floor3_full_200"]
    assert run(load_campus(), sc, "lookahead", 0)[0]["evac_time_last_s"] == 213.0


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} checks passed")
