"""
evac/building.py
Loads a building into a NetworkX graph with physical attributes the simulator needs.

The geometry (node positions, floor alignment) mirrors
ml_edge_updater.build_full_campus_graph() exactly, so routes and the 3D view
stay consistent. On top of that, every node gets a `kind` and every edge gets:
  etype        corridor | room_door | exit_door | stair | lift | ramp
  length_m     walking length in metres
  width_m      effective width
  flow_cap     max persons per second that can ENTER the edge
  eff_cap      sustainable persons/second (min of flow_cap and the speed-density limit)
  jam_cap      max persons that fit on the edge at jam density
  free_speed   m/s at free flow
  vertical     True for edges between floors

Any building can be plugged in by providing the same dict format as campus_data.py
(nodes_* dicts of name -> (x, y), edges_* lists of (u, v), bridges between floors).
"""

import math
import networkx as nx

from . import config as C
from .physics import density_width, sustainable_flow

EXIT_NODES = [
    "north entrance/exit",
    "east entrance/exit",
    "north main entrance",
    "east main entrance",
]


def node_kind(name: str) -> str:
    low = name.lower()
    if name in EXIT_NODES or "exit" in low or "entrance" in low:
        return "exit"
    if "stairs" in low:
        return "stair"
    if "lift" in low:
        return "lift"
    if name.startswith("j") or "_j" in name:
        return "junction"
    return "room"


def _edge_type(G, u, v) -> str:
    ku, kv = G.nodes[u]["kind"], G.nodes[v]["kind"]
    if G.nodes[u]["floor"] != G.nodes[v]["floor"]:
        if ku == "lift" or kv == "lift":
            return "lift"
        if ku == "ramp" and kv == "ramp":
            return "ramp"
        return "stair"
    if ku == "exit" or kv == "exit":
        return "exit_door"
    if ku == "room" or kv == "room":
        return "room_door"
    return "corridor"


def _attach_physics(G):
    for u, v, d in G.edges(data=True):
        et = _edge_type(G, u, v)
        if et == "stair":
            length = C.STAIR_FLIGHT_LENGTH_M
            speed = C.STAIR_DOWN_SPEED
        elif et == "lift":
            length = C.STAIR_FLIGHT_LENGTH_M
            speed = C.FREE_WALK_SPEED
        elif et == "ramp":
            length = C.RAMP_LENGTH_PER_FLOOR_M
            speed = C.RAMP_SPEED
        else:
            length = max(1.0, d["base_distance"] * C.METERS_PER_UNIT)
            speed = C.FREE_WALK_SPEED
        width = C.WIDTH_M[et]
        dw = density_width(et, width)
        d.update(
            etype=et,
            vertical=(G.nodes[u]["floor"] != G.nodes[v]["floor"]),
            length_m=round(length, 2),
            width_m=width,
            dens_width_m=dw,
            flow_cap=C.SPECIFIC_FLOW[et] * width,
            # what the edge can really sustain: door/stair flow rate OR speed-density limit, whichever is lower
            eff_cap=min(C.SPECIFIC_FLOW[et] * width, sustainable_flow(dw, speed)),
            jam_cap=max(2, int(C.JAM_DENSITY * length * dw)),
            free_speed=speed,
        )
    return G


def set_edge_width(d, width_m):
    """Give one edge a new clear width and recompute its capacities (used for test geometries)."""
    et = d["etype"]
    dw = density_width(et, width_m)
    d.update(width_m=width_m, dens_width_m=dw, flow_cap=C.SPECIFIC_FLOW[et] * width_m,
             eff_cap=min(C.SPECIFIC_FLOW[et] * width_m, sustainable_flow(dw, d["free_speed"])),
             jam_cap=max(2, int(C.JAM_DENSITY * d["length_m"] * dw)))
    return d


def build_graph(nodes, edges, floor_height=350, kinds=None):
    """
    Build ANY building from plain data.
      nodes: {name: (x, y, floor)}  - x, y in the same units as METERS_PER_UNIT
      edges: [(u, v), ...]          - physically walkable connections only
      kinds: optional {name: kind} overrides (exit | stair | lift | ramp | junction | room)
    Node kinds otherwise come from names (exit/entrance, stairs, lift, j*, otherwise room),
    so a new building only needs the same naming conventions as campus_data.py.
    """
    G = nx.Graph()
    for name, (x, y, fl) in nodes.items():
        G.add_node(name, floor=fl, x=x, y=y, z=fl * floor_height)
    for u, v in edges:
        if u in G and v in G and u != v:
            a, b = G.nodes[u], G.nodes[v]
            dist = math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)
            G.add_edge(u, v, base_distance=round(dist, 2))
    for n in G.nodes:
        G.nodes[n]["kind"] = (kinds or {}).get(n) or node_kind(n)
    return _attach_physics(G)


def load_building(name="campus"):
    """'campus' (default), 'hospital', or any building-pack file: a path to a .json file, or the name of one in
    buildings/ (see docs/BUILDING_GUIDE.md and evac/building_pack.py)."""
    import os
    if name == "hospital":
        from .hospital import load_hospital
        return load_hospital()
    if name == "campus":
        return load_campus()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (name, os.path.join(root, "buildings", name), os.path.join(root, "buildings", f"{name}.json")):
        if os.path.isfile(p):
            return load_building_file(p)
    raise ValueError(f"unknown building {name!r} (campus, hospital, or a file in buildings/)")


def load_building_file(path):
    """Load a building-pack JSON file:
        {"name": ..., "units": "m",
         "nodes": {"<name>": [x_m, y_m, floor], ...},
         "edges": [["<u>", "<v>"], ...],
         "kinds": {"<name>": "room|junction|stair|lift|ramp|exit"},     (optional - else taken from the names)
         "widths": {"<u>|<v>": clear_width_m, ...}}                      (optional - else config.WIDTH_M by type)
    Coordinates are metres; floors are integers (0 = ground)."""
    import json
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    u = 1.0 / C.METERS_PER_UNIT
    nodes = {n: (x * u, y * u, int(fl)) for n, (x, y, fl) in doc["nodes"].items()}
    G = build_graph(nodes, [tuple(e) for e in doc["edges"]], kinds=doc.get("kinds"))
    for key, w in (doc.get("widths") or {}).items():
        a, b = key.split("|")
        if G.has_edge(a, b):
            set_edge_width(G.edges[a, b], float(w))
    G.graph["name"] = doc.get("name", path)
    return G


def load_campus():
    """Main Block: Ground + 6 floors. Geometry identical to build_full_campus_graph()."""
    import campus_data as cd

    G = nx.Graph()
    FLOOR_HEIGHT = 350
    ground_anchor = cd.nodes_g_raw.get("stairs 1", (1179, 1174))

    for name, (x, y) in cd.nodes_g_raw.items():
        G.add_node(name, floor=0, x=x, y=y, z=0)

    upper = [
        (cd.nodes_f1_raw, cd.edges_f1, "f1_stairs 1", 1, 1.5),
        (cd.nodes_f2, cd.edges_f2, "f2_stairs 1", 2, 1.8),
        (cd.nodes_f3, cd.edges_f3, "f3_stairs 1", 3, 1.8),
        (cd.nodes_f4, cd.edges_f4, "f4_stairs 1", 4, 0.4),
        (cd.nodes_f5, cd.edges_f5, "f5_stairs 1", 5, 0.4),
        (cd.nodes_f6, cd.edges_f6, "f6_stairs 1", 6, 0.4),
    ]
    for nodes, _e, anchor, level, scale in upper:
        ax, ay = nodes.get(anchor, (0, 0))
        for name, (x, y) in nodes.items():
            fx = ground_anchor[0] + (x - ax) / scale
            fy = ground_anchor[1] + (y - ay) / scale
            G.add_node(name, floor=level, x=fx, y=fy, z=level * FLOOR_HEIGHT)

    all_edges = list(cd.edges_g)
    for _n, edges, *_ in upper:
        all_edges.extend(edges)
    all_edges.extend(cd.bridges)

    for u, v in all_edges:
        if u in G and v in G and u != v:
            a, b = G.nodes[u], G.nodes[v]
            dist = math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)
            G.add_edge(u, v, base_distance=round(dist, 2))

    for n in G.nodes:
        G.nodes[n]["kind"] = node_kind(n)
    return _attach_physics(G)


def exits(G):
    return [n for n in G if G.nodes[n]["kind"] == "exit"]


def room_nodes(G):
    """Rooms that can actually reach an exit (spawn points)."""
    ex = exits(G)
    reach = set()
    for e in ex:
        reach |= nx.node_connected_component(G, e)
    return sorted(n for n in G if G.nodes[n]["kind"] == "room" and n in reach)


def unreachable_nodes(G):
    ex = exits(G)
    reach = set()
    for e in ex:
        reach |= nx.node_connected_component(G, e)
    return sorted(n for n in G if n not in reach)


def stair_shaft(name: str):
    """'f3_stairs 2' -> '2', 'stairs 4' -> '4'. None if not a stair node."""
    if "stairs" not in name.lower():
        return None
    return name.split()[-1]
