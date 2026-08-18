"""
server.py
Feature 1 ML Routing — FastAPI backend with 2D Floor-by-Floor Route Segmentation.

Endpoints:
  POST /api/evacuate   — run ML evacuation routing with dynamic inputs & 2D floor segments
  GET  /api/scenarios  — list predefined scenario templates (for UI dropdowns)
  GET  /api/nodes      — all graph node IDs grouped by floor (for UI dropdowns)

The backend calls the actual Feature 1 pipeline:
  ml_edge_updater.py → ml_evacuation_router.py → Random Forest → NetworkX Dijkstra

No ML logic is duplicated here.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from ml_edge_updater import (
    build_ground_floor_graph,
    build_full_campus_graph,
    apply_environmental_scenario,
)
from ml_evacuation_router import find_optimal_evacuation_routes, GROUND_EXITS
from ml_routing_model import get_predictor
import campus_data as cd

# ─────────────────────────────────────────────────────────────────────────────
# 2D Floor Plan Definitions & Image Mappings
# ─────────────────────────────────────────────────────────────────────────────
FLOOR_DEFS = {
    0: {
        "name": "Ground Floor",
        "nodes": cd.nodes_g_raw,
        "image": "/static/floor_plan_hd.jpg",
        "width": 4096,
        "height": 2878,
    },
    1: {
        "name": "Floor 1",
        "nodes": cd.nodes_f1_raw,
        "image": "/static/first_floor_hd.jpg",
        "width": 7709,
        "height": 5394,
    },
    2: {
        "name": "Floor 2",
        "nodes": cd.nodes_f2,
        "image": "/static/second_floor_hd.jpg",
        "width": 9292,
        "height": 6472,
    },
    3: {
        "name": "Floor 3",
        "nodes": cd.nodes_f3,
        "image": "/static/second_floor_hd.jpg",
        "width": 9292,
        "height": 6472,
    },
    4: {
        "name": "Floor 4",
        "nodes": cd.nodes_f4,
        "image": "/static/fourth_floor_hd.jpg",
        "width": 1980,
        "height": 1530,
    },
    5: {
        "name": "Floor 5",
        "nodes": cd.nodes_f5,
        "image": "/static/fourth_floor_hd.jpg",
        "width": 1980,
        "height": 1530,
    },
    6: {
        "name": "Floor 6",
        "nodes": cd.nodes_f6,
        "image": "/static/sixth_floor_hd.jpg",
        "width": 1980,
        "height": 1530,
    },
}

# Inverted mapping: node_id -> (floor_num, raw_x, raw_y)
NODE_TO_FLOOR_2D: Dict[str, tuple] = {}
for fl_idx, fld in FLOOR_DEFS.items():
    for n_id, (nx, ny) in fld["nodes"].items():
        NODE_TO_FLOOR_2D[n_id] = (fl_idx, nx, ny)


def segment_route_for_2d(path: List[str], hazard_node: Optional[str] = None, hazard_radius: float = 280.0) -> List[Dict[str, Any]]:
    """
    Splits the authoritative ML evacuation route into sequential per-floor segments
    annotated with 2D floor-plan coordinates, transitions, actions, and hazard positions.
    """
    if not path:
        return []

    # Map each node in the path to its floor and 2D pixel coordinates
    node_infos = []
    for n in path:
        fl_idx, x, y = NODE_TO_FLOOR_2D.get(n, (0, 0, 0))
        is_stair = "stair" in n.lower()
        is_lift = "lift" in n.lower()
        node_infos.append({
            "id": n,
            "floor": fl_idx,
            "x": x,
            "y": y,
            "is_stair": is_stair,
            "is_lift": is_lift,
        })

    # Group contiguous nodes by floor
    grouped_segments = []
    curr_fl = None
    curr_nodes = []

    for info in node_infos:
        if info["floor"] != curr_fl:
            if curr_nodes:
                grouped_segments.append((curr_fl, curr_nodes))
            curr_fl = info["floor"]
            curr_nodes = [info]
        else:
            curr_nodes.append(info)
    if curr_nodes:
        grouped_segments.append((curr_fl, curr_nodes))

    origin_floor = node_infos[0]["floor"]
    exit_floor = node_infos[-1]["floor"]
    total_segments = len(grouped_segments)

    result_segments = []
    for seg_idx, (fl_idx, seg_nodes) in enumerate(grouped_segments):
        fld = FLOOR_DEFS[fl_idx]
        is_origin = (seg_idx == 0)
        is_exit = (seg_idx == total_segments - 1)

        # Classify nodes for rendering
        annotated_nodes = []
        for n_idx, node_data in enumerate(seg_nodes):
            role = "waypoint"
            label = node_data["id"]

            if is_origin and n_idx == 0:
                role = "origin"
                label = f"Start: {node_data['id']}"
            elif is_exit and n_idx == len(seg_nodes) - 1:
                role = "exit"
                label = f"Exit: {node_data['id']}"
            elif node_data["is_stair"]:
                role = "transition_stair"
                label = f"{node_data['id']} (Stairs)"
            elif node_data["is_lift"]:
                role = "transition_lift"
                label = f"{node_data['id']} (Lift)"

            annotated_nodes.append({
                "id": node_data["id"],
                "x": node_data["x"],
                "y": node_data["y"],
                "role": role,
                "label": label,
            })

        # Generate floor header and instructional action text
        fl_name = fld["name"]
        if is_origin and is_exit:
            badge = "YOU ARE HERE & EXIT"
            status_title = f"{fl_name.upper()} — YOU ARE HERE"
            action_desc = f"Evacuate directly to {path[-1]}"
        elif is_origin:
            badge = "YOU ARE HERE"
            status_title = f"{fl_name.upper()} — YOU ARE HERE"
            last_node = seg_nodes[-1]["id"]
            action_desc = f"Proceed to {last_node} to transfer to next floor"
        elif is_exit:
            badge = "FINAL EXIT"
            status_title = f"{fl_name.upper()} — EXIT BUILDING"
            action_desc = f"Exit building via {path[-1]}"
        else:
            badge = f"STEP {seg_idx + 1}"
            status_title = f"{fl_name.upper()} — CONTINUE ↓"
            last_node = seg_nodes[-1]["id"]
            action_desc = f"Transfer through {last_node}"

        # Hazard marker if hazard is on this floor
        hazard_info = None
        if hazard_node and hazard_node in NODE_TO_FLOOR_2D:
            h_fl, hx, hy = NODE_TO_FLOOR_2D[hazard_node]
            if h_fl == fl_idx:
                hazard_info = {
                    "node": hazard_node,
                    "x": hx,
                    "y": hy,
                    "radius": hazard_radius,
                }

        result_segments.append({
            "segment_index": seg_idx + 1,
            "floor_index": fl_idx,
            "floor_name": fl_name,
            "image_url": fld["image"],
            "image_width": fld["width"],
            "image_height": fld["height"],
            "badge": badge,
            "status_title": status_title,
            "action_desc": action_desc,
            "is_origin_floor": is_origin,
            "is_exit_floor": is_exit,
            "nodes": annotated_nodes,
            "hazard": hazard_info,
        })

    return result_segments


# ─────────────────────────────────────────────────────────────────────────────
# Scenario templates — used ONLY for UI dropdown defaults
# ─────────────────────────────────────────────────────────────────────────────
SCENARIO_TEMPLATES = {
    "normal": {
        "label": "Normal — No Emergency",
        "description": "Baseline routing using purely physical distance.",
        "defaults": {
            "emergency_type": "normal",
            "hazard_node": "",
            "hazard_intensity": 0.0,
            "hazard_radius": 300.0,
            "crowd_node": "",
            "crowd_intensity": 0.0,
            "crowd_radius": 300.0,
            "mobility_impaired": False,
            "multi_floor": False,
        },
    },
    "fire": {
        "label": "Fire / Hazard",
        "description": "Fire or hazardous material outbreak at a specified node.",
        "defaults": {
            "emergency_type": "fire",
            "hazard_node": "j6",
            "hazard_intensity": 1.0,
            "hazard_radius": 280.0,
            "crowd_node": "",
            "crowd_intensity": 0.0,
            "crowd_radius": 300.0,
            "mobility_impaired": False,
            "multi_floor": False,
        },
    },
    "crowd": {
        "label": "Crowd Congestion",
        "description": "Crowd bottleneck at a specified node.",
        "defaults": {
            "emergency_type": "crowd",
            "hazard_node": "",
            "hazard_intensity": 0.0,
            "hazard_radius": 300.0,
            "crowd_node": "j2",
            "crowd_intensity": 0.95,
            "crowd_radius": 300.0,
            "mobility_impaired": False,
            "multi_floor": False,
        },
    },
    "accessibility": {
        "label": "Accessibility / Mobility-Impaired",
        "description": "Step-free route. Staircases receive severe penalty. Elevators preferred.",
        "defaults": {
            "emergency_type": "accessibility",
            "hazard_node": "",
            "hazard_intensity": 0.0,
            "hazard_radius": 300.0,
            "crowd_node": "",
            "crowd_intensity": 0.0,
            "crowd_radius": 300.0,
            "mobility_impaired": True,
            "multi_floor": True,
        },
    },
    "multifloor_fire": {
        "label": "Multi-Floor: Stairwell Blocked",
        "description": "A stairwell on an upper floor is blocked by fire/smoke.",
        "defaults": {
            "emergency_type": "fire",
            "hazard_node": "f1_stairs 1",
            "hazard_intensity": 1.0,
            "hazard_radius": 300.0,
            "crowd_node": "",
            "crowd_intensity": 0.0,
            "crowd_radius": 300.0,
            "mobility_impaired": False,
            "multi_floor": True,
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Emergency Evacuation ML Routing API", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class EvacuateRequest(BaseModel):
    origin: str = Field(..., description="Graph node ID where the person is located.")
    emergency_type: str = Field(
        "normal",
        description="One of: normal | fire | crowd | accessibility | multifloor_fire"
    )
    hazard_node: Optional[str] = Field(None, description="Center node of hazard.")
    hazard_intensity: float = Field(1.0, ge=0.0, le=1.0)
    hazard_radius: float = Field(280.0, ge=10.0, le=2000.0)

    crowd_node: Optional[str] = Field(None, description="Center node of crowd.")
    crowd_intensity: float = Field(0.0, ge=0.0, le=1.0)
    crowd_radius: float = Field(300.0, ge=10.0, le=2000.0)

    mobility_impaired: bool = Field(False)
    multi_floor: bool = Field(False)


class RouteResult(BaseModel):
    primary_route: list
    recommended_exit: str
    physical_distance: float
    ml_cost: float
    peak_hazard: float
    average_crowd: float
    status: str

    origin: str
    emergency_type: str
    hazard_node: Optional[str]
    crowd_node: Optional[str]
    mobility_impaired: bool

    path_changed_from_baseline: bool
    baseline_route: list
    baseline_exit: str

    hazard_zone_node: Optional[str] = None
    hazard_zone_radius: Optional[float] = None

    # 2D Floor Plan Segmentation for Visualisation
    floor_segments: List[Dict[str, Any]] = []


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    """Serve the main Three.js + 2D floor plans UI (index.html)."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"),
                        media_type="text/html")


@app.get("/campus_data.js")
def serve_campus_data():
    """Serve the campus graph JS file."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "campus_data.js"),
                        media_type="application/javascript")


@app.get("/api/scenarios")
def list_scenarios():
    """Return scenario templates with labels and default field values."""
    return {
        key: {
            "label": val["label"],
            "description": val["description"],
            "defaults": val["defaults"],
        }
        for key, val in SCENARIO_TEMPLATES.items()
    }


@app.get("/api/nodes")
def list_nodes():
    """
    Return all valid node IDs organized by floor and category for UI dropdowns.
    """
    exits_set = {"east entrance/exit", "east main entrance", "north entrance/exit", "north main entrance"}

    floor_room_groups = {}
    for fl_idx, fld in FLOOR_DEFS.items():
        fl_nodes = fld["nodes"]
        rooms = [
            n for n in sorted(fl_nodes.keys())
            if not n.startswith("j") and not n.startswith(f"f{fl_idx}_j")
            and n not in exits_set
            and "stair" not in n.lower()
            and "lift" not in n.lower()
        ]
        floor_room_groups[fld["name"]] = rooms

    # Junctions across floors
    junctions_by_floor = {}
    for fl_idx, fld in FLOOR_DEFS.items():
        fl_nodes = fld["nodes"]
        juncs = [
            n for n in sorted(fl_nodes.keys())
            if (n.startswith("j") and fl_idx == 0) or n.startswith(f"f{fl_idx}_j")
        ]
        junctions_by_floor[fld["name"]] = juncs

    all_ground = sorted(cd.nodes_g_raw.keys())
    hazard_nodes_all = sorted([n for n in NODE_TO_FLOOR_2D.keys() if "j" in n or "stair" in n.lower()])

    return {
        "floor_rooms": floor_room_groups,
        "floor_junctions": junctions_by_floor,
        "all_ground_nodes": all_ground,
        "hazard_nodes_all": hazard_nodes_all,
        "exits": sorted(exits_set),
    }


@app.post("/api/evacuate", response_model=RouteResult)
def run_evacuation(req: EvacuateRequest):
    """
    Run Feature 1 ML evacuation routing with fully dynamic inputs and 2D floor segments.
    """
    origin = req.origin.strip()
    emergency_type = req.emergency_type.lower()

    # Determine graph scope (multi-floor if origin is on upper floor or requested)
    origin_fl = NODE_TO_FLOOR_2D.get(origin, (0, 0, 0))[0]
    use_multi = req.multi_floor or (origin_fl > 0) or emergency_type in ("accessibility", "multifloor_fire")

    if use_multi:
        G_baseline = build_full_campus_graph()
        G_dynamic  = build_full_campus_graph()
    else:
        G_baseline = build_ground_floor_graph()
        G_dynamic  = build_ground_floor_graph()

    if origin not in G_baseline.nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Origin node '{origin}' not found in graph.",
        )

    # Dynamic hazard & crowd sources
    hazard_sources: List[dict] = []
    crowd_sources:  List[dict] = []

    if emergency_type in ("fire", "multifloor_fire") and req.hazard_node:
        hazard_node = req.hazard_node.strip()
        if hazard_node not in G_dynamic.nodes:
            raise HTTPException(
                status_code=400,
                detail=f"Hazard node '{hazard_node}' not found in graph.",
            )
        hazard_sources.append({
            "node":      hazard_node,
            "radius":    req.hazard_radius,
            "intensity": req.hazard_intensity,
        })

    if emergency_type == "crowd" and req.crowd_node:
        crowd_node = req.crowd_node.strip()
        if crowd_node not in G_dynamic.nodes:
            raise HTTPException(
                status_code=400,
                detail=f"Crowd node '{crowd_node}' not found in graph.",
            )
        crowd_sources.append({
            "node":      crowd_node,
            "radius":    req.crowd_radius,
            "intensity": req.crowd_intensity,
        })

    mobility_impaired = req.mobility_impaired or (emergency_type == "accessibility")

    # Apply environment & predict edge weights
    predictor = get_predictor()
    apply_environmental_scenario(
        G_dynamic,
        hazard_sources=hazard_sources,
        crowd_sources=crowd_sources,
        mobility_impaired=mobility_impaired,
        predictor=predictor,
    )

    # Baseline & ML dynamic routes
    baseline_routes = find_optimal_evacuation_routes(G_baseline, origin)
    if not baseline_routes:
        raise HTTPException(status_code=500, detail=f"No baseline routes found from '{origin}'.")

    dynamic_routes = find_optimal_evacuation_routes(G_dynamic, origin)
    if not dynamic_routes:
        raise HTTPException(status_code=500, detail=f"No ML dynamic routes found from '{origin}'.")

    best_dynamic  = dynamic_routes[0]
    best_baseline = baseline_routes[0]
    path_changed  = best_dynamic["path"] != best_baseline["path"]

    viz_hazard_node   = hazard_sources[0]["node"]   if hazard_sources else None
    viz_hazard_radius = hazard_sources[0]["radius"]  if hazard_sources else None

    # Compute 2D floor segments for visualization
    floor_segments = segment_route_for_2d(
        best_dynamic["path"],
        hazard_node=viz_hazard_node,
        hazard_radius=viz_hazard_radius or 280.0,
    )

    return RouteResult(
        primary_route=best_dynamic["path"],
        recommended_exit=best_dynamic["destination_exit"],
        physical_distance=best_dynamic["physical_distance"],
        ml_cost=best_dynamic["predicted_cost"],
        peak_hazard=best_dynamic["max_hazard"],
        average_crowd=best_dynamic["avg_crowd"],
        status="SAFE" if best_dynamic["is_safe"] else "DANGER",

        origin=origin,
        emergency_type=emergency_type,
        hazard_node=req.hazard_node,
        crowd_node=req.crowd_node,
        mobility_impaired=mobility_impaired,

        path_changed_from_baseline=path_changed,
        baseline_route=best_baseline["path"],
        baseline_exit=best_baseline["destination_exit"],

        hazard_zone_node=viz_hazard_node,
        hazard_zone_radius=viz_hazard_radius,

        floor_segments=floor_segments,
    )


if __name__ == "__main__":
    import uvicorn
    print("Starting Emergency Evacuation ML Routing Server (v2.5 — with 2D Floor Plans)")
    print("Open http://localhost:8000 in your browser.")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
