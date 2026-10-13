"""
evac/hospital.py
Second building: 100-bedded Oncology Block (Government Hospital, Tiruppur), ground + 3 floors.

The team digitised the architectural drawings (HOS-TIR-ARCH-DD-1001..1004) into node/edge files,
one pixel coordinate frame per floor image. This module turns those into the engine's format:

  python -m evac.hospital --import <folder with the team's hospital files>
      -> data/hospital/hospital_graph.json   (metres, one shared frame, kinds, data fixes, provenance)

  from evac.hospital import load_hospital
  G = load_hospital()                        # same graph format as load_campus()

Scale and alignment (no guessing):
  - Ground floor: the drawing's overall dimension line, 54,675 mm, spans pixels 387..5719 of
    hospital_ground_floor.jpg  ->  0.010254 m per pixel.
  - Upper floors: their images are cropped differently, so the scale is FITTED: the six vertical
    shafts every floor shares (stairs 1, stairs 2, lift, ramp, bed lifts 1-2) are matched to the
    ground floor by least squares (scale + shift). The fit residuals are stored in the JSON; they
    are <= 0.2 m, i.e. the team's node placement is consistent across floors.

Reading the team's files: values are extracted with json / ast.literal_eval only - their scripts
(which open matplotlib windows) are never executed.
"""

import argparse
import ast
import json
import math
import os

from . import config as C
from .building import build_graph

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "hospital", "hospital_graph.json")

FLOORS = (("g", 0, "ground"), ("f1", 1, "first"), ("f2", 2, "second"), ("f3", 3, "third"))
SHAFTS = ("stairs 1", "stairs 2", "lift", "ramp up", "bed lift 1", "bed lift 2")
DIM_MM, DIM_PX = 54675.0, (387, 5719)        # ground-floor overall dimension line

# Outside points. The main entrance has 3 steps up to the porch (+450 mm); ramps 1 and 2 are the
# step-free approach ramps beside it; the east "ramp down" and the staff entry are separate doors.
EXITS = ("g_main entrance", "g_ramp 1", "g_ramp 2", "g_ramp down exit", "g_staff entry")

# Data fixes (listed in the JSON and in docs/hospital_results.md):
FIXES = [
    (("f2_general ward 2", "f2_j5"), "general ward 2 was reachable only by walking through wards 1 and 3; "
                                      "added its door to the corridor junction below it (~5 m)"),
    (("f3_general ward 2", "f3_j5"), "same on the third floor"),
]


def _literal(path, names):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def _read_team_files(folder):
    nodes, edges = {}, []
    with open(os.path.join(folder, "ground_floor", "hospital_ground_nodes.json"), encoding="utf-8") as f:
        nodes.update({k: tuple(v) for k, v in json.load(f).items()})
    with open(os.path.join(folder, "ground_floor", "hospital_ground_edges.json"), encoding="utf-8") as f:
        edges += [tuple(e) for e in json.load(f)]
    for pre, _lvl, word in FLOORS[1:]:
        d = _literal(os.path.join(folder, f"{word}_floor", f"hospital_{word}_data.py"), {f"nodes_{pre}", f"edges_{pre}"})
        nodes.update({k: tuple(v) for k, v in d[f"nodes_{pre}"].items()})
        edges += [tuple(e) for e in d[f"edges_{pre}"]]
    vertical = [tuple(e) for e in _literal(os.path.join(folder, "hospital_vertical_edges.py"), {"vertical_edges"})["vertical_edges"]]
    return nodes, edges, vertical


def _fit(src, dst):
    """dst ~ s * src + t (uniform scale, no rotation), least squares."""
    n = len(src)
    mx = [sum(p[i] for p in src) / n for i in (0, 1)]
    my = [sum(p[i] for p in dst) / n for i in (0, 1)]
    num = sum((a[i] - mx[i]) * (b[i] - my[i]) for a, b in zip(src, dst) for i in (0, 1))
    den = sum((a[i] - mx[i]) ** 2 for a in src for i in (0, 1))
    s = num / den
    t = [my[i] - s * mx[i] for i in (0, 1)]
    res = [math.dist((s * a[0] + t[0], s * a[1] + t[1]), b) for a, b in zip(src, dst)]
    return s, t, res


def import_team_files(folder, out=DATA):
    nodes_px, edges, vertical = _read_team_files(folder)
    missing = [e for e in edges + vertical if e[0] not in nodes_px or e[1] not in nodes_px]
    if missing:
        raise ValueError(f"edges refer to unknown nodes: {missing[:5]}")
    s_g = DIM_MM / 1000.0 / (DIM_PX[1] - DIM_PX[0])
    ground = [(nodes_px[f"g_{a}"][0] * s_g, nodes_px[f"g_{a}"][1] * s_g) for a in SHAFTS]
    transform = {"g": (s_g, [0.0, 0.0], [0.0] * len(SHAFTS))}
    for pre, _lvl, _w in FLOORS[1:]:
        transform[pre] = _fit([nodes_px[f"{pre}_{a}"] for a in SHAFTS], ground)
    nodes = {}
    for name, (x, y) in nodes_px.items():
        pre = name.split("_", 1)[0]
        lvl = {p: l for p, l, _ in FLOORS}[pre]
        s, t, _ = transform[pre]
        nodes[name] = [round(s * x + t[0], 3), round(s * y + t[1], 3), lvl]
    fixed = [list(e) for e, _why in FIXES if e[0] in nodes and e[1] in nodes and
             tuple(e) not in set(edges) and tuple(e[::-1]) not in set(edges)]
    kinds = {n: "exit" for n in EXITS}
    kinds.update({n: "ramp" for n in nodes if n.endswith("_ramp up")})
    doc = {
        "building": "100-bedded Oncology Block, Government Hospital, Tiruppur (ground + 3 floors)",
        "source": "team-digitised nodes/edges over drawings HOS-TIR-ARCH-DD-1001..1004",
        "units": "metres; one frame for all floors (ground-floor drawing frame)",
        "scale": {pre: {"m_per_px": round(transform[pre][0], 6),
                        "shaft_fit_residuals_m": [round(r, 2) for r in transform[pre][2]]} for pre, _l, _w in FLOORS},
        "scale_note": f"ground: {DIM_MM:.0f} mm dimension line over pixels {DIM_PX[0]}-{DIM_PX[1]}; "
                      "upper floors: least-squares fit of the 6 shared shafts to the ground floor",
        "nodes": nodes,
        "edges": [list(e) for e in edges + vertical] + fixed,
        "kinds": kinds,
        "fixes": [{"edge": list(e), "why": why} for e, why in FIXES],
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return doc


def load_hospital(path=DATA):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    u = 1.0 / C.METERS_PER_UNIT                     # engine works in graph units; metres -> units
    nodes = {n: (x * u, y * u, fl) for n, (x, y, fl) in doc["nodes"].items()}
    G = build_graph(nodes, [tuple(e) for e in doc["edges"]], floor_height=350, kinds=doc["kinds"])
    G.graph["name"] = "hospital"
    # Fire compartments (Session 7): each floor is split into a West and an East compartment at the
    # bed-lift core, where the 3500 mm corridor has fire doors in a typical hospital layout. ASSUMPTION -
    # the drawings do not mark fire doors. Used only by fires marked "contained": true.
    for fl in range(4):
        pre = "g" if fl == 0 else f"f{fl}"
        core = G.nodes[f"{pre}_bed lift 1"]["x"]
        for n, d in G.nodes(data=True):
            if d["floor"] == fl:
                d["compartment"] = f"{pre}-{'W' if d['x'] < core else 'E'}"
    return G


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--import", dest="folder", required=True, help="folder with the team's hospital files")
    a = ap.parse_args(argv)
    doc = import_team_files(a.folder)
    print(f"{len(doc['nodes'])} nodes, {len(doc['edges'])} edges -> {DATA}")
    for pre, v in doc["scale"].items():
        print(f"  {pre}: {v['m_per_px']} m/px, shaft residuals {v['shaft_fit_residuals_m']} m")


if __name__ == "__main__":
    main()


def scenarios(path=os.path.join(HERE, "scenarios_hospital.json")):
    """Hospital scenarios; an entry with "base" inherits everything from that entry, then overrides."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for name, sc in raw.items():
        base = dict(raw[sc["base"]]) if "base" in sc else {}
        base.update({k: v for k, v in sc.items() if k != "base"})
        out[name] = base
    return out
