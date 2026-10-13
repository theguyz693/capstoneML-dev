"""
Session 8 checks: IMO verification tests, validation helpers, field-drill script, what-if API.
    python tests/test_evac_session8.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evac import validation as V


def test_imo_tests_pass():
    for t in V.v3_imo():
        assert t["pass"], t


def test_door_flow_is_below_lab_measurements_and_imo_limit():
    for w, f in V.v1_bottleneck()["sim"]:
        assert f <= 1.33 * w + 0.02
        lab = dict(V.SEYFRIED).get(w)
        if lab:
            assert f < lab


def test_field_drill_script():
    p = os.path.join(tempfile.mkdtemp(), "drills.csv")
    with open(p, "w", encoding="utf-8") as f:
        f.write("date,building,room,people,door_width_m,exit_used,first_out_room_s,last_out_room_s,last_out_building_s,notes\n"
                "2026-10-20,campus,f1_sem hall 1,40,1.0,north,5,40,95,test\n")
    out = V.drill(p, seeds=range(2))
    assert len(out) == 1 and out[0][3] > 0


def test_whatif_api():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from evac.api import router
    app = FastAPI()
    app.include_router(router)
    c = TestClient(app)
    els = c.get("/api/research/whatif/elements?building=hospital").json()
    ramp = next(e["id"] for e in els if e["kind"] == "ramp")
    r = c.post("/api/research/whatif", json={"building": "hospital", "scenario": "hosp_day_fire_stairs1_f2",
                                             "blocked": [ramp], "seeds": 1}).json()
    assert r["modified"]["safe_shortest"]["cut_off"] > r["baseline"]["safe_shortest"]["cut_off"] + 20
    assert c.post("/api/research/whatif", json={"blocked": ["nope"]}).status_code == 400
    r = c.post("/api/research/whatif", json={"scenario": "floor3_full_200", "people_factor": 0.5, "seeds": 1}).json()
    assert r["people"] == 100 and r["baseline"]["safe_shortest"]["jams"]


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
    print(f"{len(fns)} checks passed")
