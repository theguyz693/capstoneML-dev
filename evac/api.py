"""
evac/api.py
HTTP endpoints for the research engine (FastAPI router). Mounted by research_server.py,
so the existing server.py stays untouched.

  GET  /api/research/building     nodes, edges, exits (for drawing the plan)
  GET  /api/research/scenarios    predefined scenarios
  POST /api/research/run          run one scenario with several methods; returns scoreboard + replay frames
  POST /api/research/signs        building-wide 'go this way' arrow for every node (kiosk signage)
  GET  /api/research/weakspots    latest weak-spot ranking (docs/weakspots.json)
"""

import json
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .building import load_campus, exits
from .hazard import apply_hazard
from .routing import next_hop_signs, route_table
from .simulator import run

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
router = APIRouter(prefix="/api/research", tags=["research"])
_G = None
ALLOWED = ("nearest", "safe_shortest", "reactive", "ccrp", "lookahead", "ml", "pred_formula", "pred_ml", "pred_ml_upper", "lookahead_ml")


def G0():
    global _G
    if _G is None:
        _G = load_campus()
    return _G


def _scenarios():
    with open(os.path.join(HERE, "scenarios.json"), encoding="utf-8") as f:
        return json.load(f)


@router.get("/building")
def building():
    G = G0()
    return {
        "nodes": [{"id": n, "x": d["x"], "y": d["y"], "z": d["z"], "floor": d["floor"], "kind": d["kind"]}
                  for n, d in G.nodes(data=True)],
        "edges": [[u, v, d["etype"]] for u, v, d in G.edges(data=True)],
        "exits": exits(G),
    }


@router.get("/scenarios")
def scenarios():
    return _scenarios()


class RunRequest(BaseModel):
    scenario: str = "floor3_full_200"
    methods: List[str] = ["safe_shortest", "reactive", "lookahead"]
    seed: int = 0
    compliance: Optional[float] = None
    herding: Optional[float] = None
    n_agents: Optional[int] = None
    fire_node: Optional[str] = None          # overrides the scenario's fire
    fire_radius: float = 400.0
    fire_spreading: bool = False
    frame_every_s: float = 2.0


@router.post("/run")
def run_scenario(req: RunRequest):
    scen = _scenarios()
    if req.scenario not in scen:
        raise HTTPException(400, f"unknown scenario {req.scenario}")
    sc = dict(scen[req.scenario])
    for k in ("compliance", "herding", "n_agents"):
        if getattr(req, k) is not None:
            sc[k] = getattr(req, k)
    if req.fire_node:
        if req.fire_node not in G0():
            raise HTTPException(400, f"unknown node {req.fire_node}")
        f = {"node": req.fire_node, "radius": req.fire_radius, "intensity": 1.0}
        if req.fire_spreading:
            f.update(radius=150, spread_m_per_s=0.5, max_radius=max(req.fire_radius, 600))
        sc["fires"] = [f]
    sc["record_tracks"] = req.frame_every_s
    out = {}
    for m in req.methods:
        if m not in ALLOWED:
            raise HTTPException(400, f"unknown method {m}")
        metrics, recs = run(G0().copy(), sc, m, req.seed)
        frames = metrics.pop("frames", [])
        times = sorted(r["t_out"] for r in recs if r["t_out"] is not None)
        n = metrics["n_agents"]
        inside = [{"t": fr["t"], "inside": n - sum(1 for x in times if x <= fr["t"])} for fr in frames]
        out[m] = {"metrics": metrics, "frames": frames, "inside": inside}
    G = apply_hazard(G0().copy(), sc.get("fires", []))
    return {"scenario": req.scenario, "description": sc.get("description", ""), "fires": sc.get("fires", []),
            "static_hazard": {n: G.nodes[n]["hazard"] for n in G if G.nodes[n]["hazard"] > 0}, "results": out}


class SignRequest(BaseModel):
    fire_node: Optional[str] = None
    fire_radius: float = 400.0
    step_free: bool = False
    kiosk_node: Optional[str] = None


@router.post("/signs")
def signs(req: SignRequest):
    fires = [{"node": req.fire_node, "radius": req.fire_radius, "intensity": 1.0}] if req.fire_node else []
    G = apply_hazard(G0().copy(), fires)
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
            msg = f"Go towards {path[1] if len(path) > 1 else path[0]}"
            if stairs:
                msg += f", take {stairs[0].split('_')[-1]} down"
            msg += f", exit via {path[-1]}."
            res["kiosk"] = {"node": req.kiosk_node, "path": path, "next": path[1] if len(path) > 1 else None,
                            "exit": path[-1], "message": msg}
    return res


@router.get("/weakspots")
def weakspots():
    p = os.path.join(ROOT, "docs", "weakspots.json")
    if not os.path.exists(p):
        raise HTTPException(404, "run: python -m evac.weakspots")
    with open(p, encoding="utf-8") as f:
        return json.load(f)
