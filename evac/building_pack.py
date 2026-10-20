"""
evac/building_pack.py
Phase 2, item 8: a public "building pack" - any floor plan in one plain JSON file - and the tools around it.

  python -m evac.building_pack export                 write buildings/campus.json and buildings/hospital.json
  python -m evac.building_pack check  FILE            is the file a valid, evacuable building? (lists problems)
  python -m evac.building_pack run    FILE [--people 150] [--fire NODE] [--seeds 3]
                                                      shortest path vs look-ahead on that building, in one go

File format (metres; see docs/BUILDING_GUIDE.md):
  {"name": "...", "units": "m",
   "nodes":  {"<name>": [x, y, floor], ...},
   "edges":  [["<u>", "<v>"], ...],
   "kinds":  {"<name>": "room|junction|stair|lift|ramp|exit"},       optional, else from the name
   "widths": {"<u>|<v>": metres}}                                       optional, else config.WIDTH_M by type
"""

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
PACK = os.path.join(ROOT, "buildings")


def export(G, name, description=""):
    """Graph -> building-pack JSON (metres)."""
    from evac import config as C
    from evac.building import node_kind
    m = C.METERS_PER_UNIT
    nodes = {n: [round(d["x"] * m, 3), round(d["y"] * m, 3), int(d["floor"])] for n, d in G.nodes(data=True)}
    kinds = {n: d["kind"] for n, d in G.nodes(data=True) if d["kind"] != node_kind(n)}
    widths = {f"{u}|{v}": d["width_m"] for u, v, d in G.edges(data=True) if d["width_m"] != C.WIDTH_M[d["etype"]]}
    doc = {"name": name, "description": description, "units": "m", "nodes": nodes,
           "edges": [[u, v] for u, v in G.edges()]}
    if kinds:
        doc["kinds"] = kinds
    if widths:
        doc["widths"] = widths
    return doc


def check(G):
    """Problems that would break or distort an evacuation study. Returns (errors, warnings, summary)."""
    import networkx as nx
    from evac.building import exits, room_nodes, unreachable_nodes
    from evac.routing import route_table
    errors, warnings = [], []
    ex = exits(G)
    if not ex:
        errors.append("no exit nodes (name them '... exit' / '... entrance' or give kind 'exit')")
    if ex and any(G.nodes[e]["floor"] != 0 for e in ex):
        warnings.append("exits above the ground floor: " + ", ".join(e for e in ex if G.nodes[e]["floor"] != 0))
    un = unreachable_nodes(G)
    if un:
        errors.append(f"{len(un)} nodes cannot reach any exit: " + ", ".join(un[:8]) + (" ..." if len(un) > 8 else ""))
    rooms = room_nodes(G)
    if not rooms:
        errors.append("no rooms that can reach an exit (nobody to evacuate)")
    floors = sorted({d["floor"] for _n, d in G.nodes(data=True)})
    for fl in floors[1:]:
        down = [(u, v) for u, v, d in G.edges(data=True) if {G.nodes[u]["floor"], G.nodes[v]["floor"]} == {fl, fl - 1}]
        kinds = {G.edges[e]["etype"] for e in down}
        if not down:
            errors.append(f"floor {fl} has no stair/ramp/lift down to floor {fl - 1}")
        elif "stair" in kinds and sum(G.edges[e]["etype"] == "stair" for e in down) < 2:
            warnings.append(f"floor {fl} has only one stair down (no alternative if it is lost)")
        if down and not kinds & {"ramp", "lift"}:
            warnings.append(f"floor {fl}: no ramp or lift - step-free people cannot leave in a fire")
    sf = route_table(G, "safe_shortest", True, True)
    no_sf = [r for r in rooms if r not in sf]
    if no_sf:
        warnings.append(f"{len(no_sf)} rooms have no step-free way out in a fire (lifts off): "
                        + ", ".join(no_sf[:5]) + (" ..." if len(no_sf) > 5 else ""))
    zero = [(u, v) for u, v, d in G.edges(data=True) if d["base_distance"] <= 0]
    if zero:
        errors.append(f"{len(zero)} edges join nodes at the same position")
    if not nx.is_connected(G):
        warnings.append(f"the graph has {nx.number_connected_components(G)} separate parts")
    summary = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "floors": len(floors), "rooms": len(rooms),
               "exits": len(ex), "edge_types": {t: sum(d['etype'] == t for _u, _v, d in G.edges(data=True))
                                                for t in sorted({d['etype'] for _u, _v, d in G.edges(data=True)})}}
    return errors, warnings, summary


def quick_run(G, people=150, fire=None, seeds=3, fire_radius_m=8.0):
    from evac import config as C
    from evac.simulator import run
    sc = {"n_agents": people, "spawn": "building", "step_free_frac": 0.05,
          "fires": [{"node": fire, "radius": fire_radius_m / C.METERS_PER_UNIT, "intensity": 1.0}] if fire else []}
    out = {}
    for m in ("safe_shortest", "lookahead"):
        ms = [run(G.copy(), sc, m, s)[0] for s in range(seeds)]
        out[m] = {"last_out_s": statistics.mean(x["evac_time_last_s"] or 0 for x in ms),
                  "mean_s": statistics.mean(x["evac_time_mean_s"] or 0 for x in ms),
                  "exposed": statistics.mean(x["exposed_agents"] for x in ms),
                  "not_out": statistics.mean(x["n_agents"] - x["evacuated"] for x in ms)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    c = sub.add_parser("check")
    c.add_argument("file")
    r = sub.add_parser("run")
    r.add_argument("file")
    r.add_argument("--people", type=int, default=150)
    r.add_argument("--fire", default=None)
    r.add_argument("--fire-radius", type=float, default=8.0, help="metres")
    r.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    from evac.building import load_building
    if a.cmd == "export":
        os.makedirs(PACK, exist_ok=True)
        for name, desc in (("campus", "PES University main block, ground + 6 floors (scale ASSUMED 0.04 m/px)"),
                           ("hospital", "100-bed oncology hospital, ground + 3 floors (scale from the drawings)")):
            doc = export(load_building(name), name, desc)
            with open(os.path.join(PACK, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=0)
            print(f"buildings/{name}.json: {len(doc['nodes'])} nodes, {len(doc['edges'])} edges")
        return
    G = load_building(a.file)
    if a.cmd == "check":
        errors, warnings, summary = check(G)
        print(json.dumps(summary))
        for e in errors:
            print("ERROR  ", e)
        for w in warnings:
            print("warning", w)
        print("OK - ready to simulate" if not errors else f"{len(errors)} error(s)")
        sys.exit(1 if errors else 0)
    errors, _w, _s = check(G)
    if errors:
        sys.exit("fix the errors first: python -m evac.building_pack check " + a.file)
    res = quick_run(G, a.people, a.fire, a.seeds, a.fire_radius)
    for m, r_ in res.items():
        print(f"{m:14s} last out {r_['last_out_s']:.0f} s, mean {r_['mean_s']:.0f} s, exposed {r_['exposed']:.1f}, "
              f"not out {r_['not_out']:.1f} (step-free people with no lift/ramp in a fire, or cut off by it)")


if __name__ == "__main__":
    main()
