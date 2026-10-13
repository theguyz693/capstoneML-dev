"""
evac/api.py
HTTP endpoints for the research engine (FastAPI router). Mounted by research_server.py,
so the existing server.py stays untouched.

  GET  /api/research/building     nodes, edges, exits (for drawing the plan)
  GET  /api/research/scenarios    predefined scenarios
  POST /api/research/run          run one scenario with several methods; returns scoreboard + replay frames
  POST /api/research/signs        building-wide 'go this way' arrow for every node (kiosk signage)
  GET  /api/research/weakspots    latest weak-spot ranking (docs/weakspots.json)
  GET  /api/research/results      key findings + figure list (docs/final_results.md)
  GET  /api/research/figure/NAME  one result figure (PNG)
"""

import json
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .building import load_building, exits
from .hazard import apply_hazard
from .routing import next_hop_signs, route_table
from .simulator import run

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
router = APIRouter(prefix="/api/research", tags=["research"])
_G = {}
BUILDINGS = ("campus", "hospital")
ALLOWED = ("nearest", "safe_shortest", "reactive", "ccrp", "lookahead", "ml", "pred_formula", "pred_ml", "pred_ml_upper", "lookahead_ml")


def _check(building):
    if building not in BUILDINGS:
        raise HTTPException(400, f"unknown building {building}")
    return building


def G0(building="campus"):
    if building not in _G:
        _G[building] = load_building(_check(building))
    return _G[building]


def _scenarios(building="campus"):
    if _check(building) == "hospital":
        from .hospital import scenarios as hs
        return hs()
    with open(os.path.join(HERE, "scenarios.json"), encoding="utf-8") as f:
        return json.load(f)


@router.get("/building")
def building(building: str = "campus"):
    G = G0(building)
    return {
        "nodes": [{"id": n, "x": d["x"], "y": d["y"], "z": d["z"], "floor": d["floor"], "kind": d["kind"]}
                  for n, d in G.nodes(data=True)],
        "edges": [[u, v, d["etype"]] for u, v, d in G.edges(data=True)],
        "exits": exits(G),
        "floors": max(d["floor"] for _n, d in G.nodes(data=True)) + 1,
    }


@router.get("/scenarios")
def scenarios(building: str = "campus"):
    return _scenarios(building)


class RunRequest(BaseModel):
    building: str = "campus"
    scenario: str = "floor3_full_200"
    methods: List[str] = ["safe_shortest", "reactive", "lookahead"]
    seed: int = 0
    compliance: Optional[float] = None
    herding: Optional[float] = None
    n_agents: Optional[int] = None
    fire_node: Optional[str] = None          # overrides the scenario's fire
    fire_radius: float = 400.0
    fire_spreading: bool = False
    no_fire: bool = False                    # run the scenario without any fire
    frame_every_s: float = 2.0


@router.post("/run")
def run_scenario(req: RunRequest):
    scen = _scenarios(req.building)
    G0b = G0(req.building)
    if req.scenario not in scen:
        raise HTTPException(400, f"unknown scenario {req.scenario}")
    sc = dict(scen[req.scenario])
    for k in ("compliance", "herding", "n_agents"):
        if getattr(req, k) is not None:
            sc[k] = getattr(req, k)
    if req.fire_node:
        if req.fire_node not in G0b:
            raise HTTPException(400, f"unknown node {req.fire_node}")
        f = {"node": req.fire_node, "radius": req.fire_radius, "intensity": 1.0}
        if req.fire_spreading:
            f.update(radius=150, spread_m_per_s=0.5, max_radius=max(req.fire_radius, 600))
        sc["fires"] = [f]
    if req.no_fire:
        sc["fires"] = []
    sc["record_tracks"] = req.frame_every_s
    out = {}
    for m in req.methods:
        if m not in ALLOWED:
            raise HTTPException(400, f"unknown method {m}")
        metrics, recs = run(G0b.copy(), sc, m, req.seed)
        frames = metrics.pop("frames", [])
        times = sorted(r["t_out"] for r in recs if r["t_out"] is not None)
        n = metrics["n_agents"]
        inside = [{"t": fr["t"], "inside": n - sum(1 for x in times if x <= fr["t"])} for fr in frames]
        out[m] = {"metrics": metrics, "frames": frames, "inside": inside}
    G = apply_hazard(G0b.copy(), sc.get("fires", []))
    return {"scenario": req.scenario, "description": sc.get("description", ""), "fires": sc.get("fires", []),
            "static_hazard": {n: G.nodes[n]["hazard"] for n in G if G.nodes[n]["hazard"] > 0}, "results": out}


class SignRequest(BaseModel):
    building: str = "campus"
    fire_node: Optional[str] = None
    fire_radius: float = 400.0
    step_free: bool = False
    kiosk_node: Optional[str] = None


@router.post("/signs")
def signs(req: SignRequest):
    fires = [{"node": req.fire_node, "radius": req.fire_radius, "intensity": 1.0}] if req.fire_node else []
    G = apply_hazard(G0(req.building).copy(), fires)
    arrows = next_hop_signs(G, "safe_shortest", fire_active=bool(fires), step_free=req.step_free)
    res = {"arrows": arrows, "hazard": {n: G.nodes[n]["hazard"] for n in G if G.nodes[n]["hazard"] > 0}}
    if req.kiosk_node:
        tab = route_table(G, "safe_shortest", bool(fires), req.step_free)
        path = tab.get(req.kiosk_node)
        if path is None:
            res["kiosk"] = {"node": req.kiosk_node, "path": None,
                            "message": "No safe route. Stay in a protected area and wait for assistance."}
        else:
            stairs = [n for n in path if G.nodes[n]["kind"] == "stair"]
            ramp = [n for n in path if G.nodes[n]["kind"] == "ramp"]
            msg = f"Go towards {path[1] if len(path) > 1 else path[0]}"
            if stairs:
                msg += f", take {stairs[0].split('_')[-1]} down"
            elif ramp and G.nodes[req.kiosk_node]["floor"] > 0:
                msg += ", take the ramp down"
            msg += f", exit via {path[-1]}."
            res["kiosk"] = {"node": req.kiosk_node, "path": path, "next": path[1] if len(path) > 1 else None,
                            "exit": path[-1], "message": msg}
    return res


@router.get("/weakspots")
def weakspots(building: str = "campus"):
    p = os.path.join(ROOT, "docs", "hospital_weakspots.json" if _check(building) == "hospital" else "weakspots.json")
    if not os.path.exists(p):
        raise HTTPException(404, "run: python -m evac.weakspots")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


FIGURES = [  # (file, title, one-line takeaway) shown on the Results tab, in presentation order
    ("fig12_main.png", "Main comparison", "Every scenario, change vs shortest safe path, 30 paired seeds, 95% CI."),
    ("fig15_tipping.png", "Tipping point", "Coordination starts to pay off once queues form (~100 people on a floor)."),
    ("fig14_mlloop.png", "ML in the loop", "Prediction quality matters little; coordination matters a lot."),
    ("fig16_freshness.png", "Data freshness", "In a growing fire, stale crowd data costs safety, not time."),
    ("fig17_ablation.png", "Ablations", "Capacity booking and sustainable stair capacity are the essential parts."),
    ("fig13_fairness.png", "Fairness", "Who pays: step-free users, upper floors, lower floors."),
    ("fig10_sensitivity.png", "Sensitivity", "The ranking survives changes to every physical assumption."),
    ("fig6_model_comparison.png", "Delay model comparison", "Grouped cross-validation: learned models vs textbook formula."),
    ("fig8_shap.png", "What the model looks at", "SHAP: crowd density and planned users dominate."),
    ("fig9_conformal.png", "Uncertainty", "Conformal intervals: calibrated 90% ranges for predicted delays."),
    ("fig11_tuning.png", "Automatic tuning", "Bayesian optimisation of planner settings, with a held-out check."),
]


HOSPITAL_FIGURES = [
    ("fig18_hospital_main.png", "Hospital: main comparison", "Change in last walking person out, and exposure, per scenario."),
    ("fig19_hospital_groups.png", "Hospital: patient groups", "Mean exit time for staff, walking patients, wheelchair users, beds."),
    ("fig20_hospital_weakspots.png", "Hospital: weak spots", "Which single failure cuts people off - the ramp."),
    ("fig21_hospital_assumptions.png", "Hospital: assumptions", "Bed and ramp assumptions vs the routing conclusion."),
]


@router.get("/results")
def results(building: str = "campus"):
    """Key findings (parsed from docs/final_results.md or hospital_results.md) and figures for the Results tab."""
    hosp = _check(building) == "hospital"
    p = os.path.join(ROOT, "docs", "hospital_results.md" if hosp else "final_results.md")
    findings = []
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            text = f.read()
        block = text.split("## Key findings", 1)[-1].split("\n## ", 1)[0]
        for line in block.splitlines():
            line = line.strip()
            head = line.split(". ", 1)
            if len(head) == 2 and head[0].isdigit():
                findings.append(head[1])
    figs = [{"file": f, "title": t, "note": n} for f, t, n in (HOSPITAL_FIGURES if hosp else FIGURES)
            if os.path.exists(os.path.join(ROOT, "docs", "figures", f))]
    return {"findings": findings, "figures": figs}


@router.get("/figure/{name}")
def figure(name: str):
    from fastapi.responses import FileResponse
    if name not in {f for f, _, _ in FIGURES + HOSPITAL_FIGURES}:
        raise HTTPException(404, "unknown figure")
    return FileResponse(os.path.join(ROOT, "docs", "figures", name), media_type="image/png")


# ------------------------------------------------------------------ what-if (Session 8)
@router.get("/whatif/elements")
def whatif_elements(building: str = "campus"):
    """Elements a building manager can take out of service: whole stairwells, the ramp, exit doors."""
    from .weakspots import candidates, _label
    G = G0(building)
    out = []
    for kind, edges in candidates(G):
        if kind == "corridor":
            continue
        out.append({"id": f"{kind}:{_label(G, edges, kind)}", "label": _label(G, edges, kind).replace("g_", ""),
                    "kind": kind, "edges": [list(e) for e in edges]})
    return out


class WhatIfRequest(BaseModel):
    building: str = "campus"
    scenario: str = "floor3_full_200"
    people_factor: float = 1.0               # 0.5 = half the people, 2 = double
    blocked: List[str] = []                  # ids from /whatif/elements
    fire_node: Optional[str] = None
    no_fire: bool = False
    seeds: int = 2


@router.post("/whatif")
def whatif(req: WhatIfRequest):
    import statistics
    from .fixed_plan import _scaled
    G = G0(req.building)
    scen = _scenarios(req.building)
    if req.scenario not in scen:
        raise HTTPException(400, f"unknown scenario {req.scenario}")
    if not 0.1 <= req.people_factor <= 3.0 or not 1 <= req.seeds <= 5:
        raise HTTPException(400, "people_factor 0.1-3, seeds 1-5")
    sc = _scaled(scen[req.scenario], req.people_factor, with_fire=True)
    for k in ("compliance", "herding", "familiar_exit"):
        if k in scen[req.scenario]:
            sc[k] = scen[req.scenario][k]
    if req.no_fire:
        sc["fires"] = []
    elif req.fire_node:
        if req.fire_node not in G:
            raise HTTPException(400, f"unknown node {req.fire_node}")
        sc["fires"] = [{"node": req.fire_node, "radius": 400, "intensity": 1.0}]
    elems = {e["id"]: e for e in whatif_elements(req.building)}
    bad = [b for b in req.blocked if b not in elems]
    if bad:
        raise HTTPException(400, f"unknown element {bad[0]}")
    Gb = G.copy()
    for b in req.blocked:
        Gb.remove_edges_from([tuple(e) for e in elems[b]["edges"]])

    def measure(graph):
        out = {}
        for m in ("safe_shortest", "lookahead"):
            runs = [run(graph.copy(), sc, m, s) for s in range(req.seeds)]
            ms = [r[0] for r in runs]
            peaks = {}
            for x in ms:
                for n, q in x["queue_peaks"].items():
                    peaks[n] = max(peaks.get(n, 0), q)
            last = [x["evac_time_last_s"] for x in ms if x["evac_time_last_s"] is not None]
            out[m] = {"last": round(statistics.mean(last), 1) if last else None,
                      "mean": round(statistics.mean(x["evac_time_mean_s"] for x in ms if x["evac_time_mean_s"]), 1),
                      "exposed": round(statistics.mean(x["exposed_agents"] for x in ms), 1),
                      "cut_off": round(statistics.mean(x["needs_assistance"] + x["stranded"] + x["timed_out"] for x in ms), 1),
                      "people": ms[0]["n_agents"],
                      "jams": sorted(peaks.items(), key=lambda kv: -kv[1])[:6]}
        return out
    return {"baseline": measure(G), "modified": measure(Gb), "blocked_edges": [e for b in req.blocked for e in elems[b]["edges"]],
            "people": sc.get("n_agents") or sum(g["n"] for g in sc.get("groups", [])),
            "fires": sc.get("fires", [])}
