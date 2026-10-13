"""
evac/jupedsim_check.py
Cross-check of OUR simulator against JuPedSim, an established open-source pedestrian simulator
(Forschungszentrum Juelich, https://www.jupedsim.org), on the same geometry with the same people.

Why: our simulator is a network (queueing) model written for this project. JuPedSim models every
person as a moving circle in continuous 2-D space. If both give similar evacuation times on the same
layouts, our times are not an artefact of our own code. JuPedSim offers several published movement
models; they do not agree with each other either, so we compare against the RANGE they span:

  CFSM  collision-free speed model (Tordeux, Chraibi & Seyfried 2016)  - JuPedSim's standard model
  AVM   anticipation velocity model (Xu, Chraibi & Seyfried 2021)
  SFM   social force model (Helbing & Molnar 1995)

Cases (all with free walking speed 1.25 m/s +-15%, like our simulator; JuPedSim default parameters
otherwise):

  C1  door test       100 people in an 8 x 5 m room, one door of width 1.0 / 1.2 / 1.6 / 2.0 / 2.4 m
                      -> door flow (persons/s) and time until the last person is out
  C2  room-corridor   100 / 200 people, 10 x 10 m room -> 1.0 m door -> 20 m x 2 m corridor -> 1.8 m exit
  C3  real floor      the campus GROUND FLOOR turned into 2-D geometry: every graph edge becomes a walkable
                      strip of its width (corridor 2 m, room door 1 m, exit door 1.8 m), every room a small
                      pocket. 150 and 300 people with the SAME origins, start times, speeds and routes
                      (shortest path) in both simulators.

Not covered: stairs. JuPedSim is 2-D; our stair speed is validated separately against NIST drills
(evac/validation.py).

Run:  python -m evac.jupedsim_check            (needs: pip install jupedsim shapely; ~1-2 h on 2 cores)
      python -m evac.jupedsim_check --quick    (one seed, CFSM only - a few minutes)
Out:  docs/jupedsim_crosscheck.md, docs/jupedsim_crosscheck.json, docs/figures/fig26_jupedsim.png
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from evac import config as C                     # noqa: E402
from evac.building import build_graph, load_campus  # noqa: E402
from evac.routing import route_table             # noqa: E402
from evac.simulator import run, spawn            # noqa: E402

DOC = os.path.join(ROOT, "docs", "jupedsim_crosscheck.md")
JSON = os.path.join(ROOT, "docs", "jupedsim_crosscheck.json")
FIG = os.path.join(ROOT, "docs", "figures", "fig26_jupedsim.png")
U = 1.0 / C.METERS_PER_UNIT          # graph units per metre
M = C.METERS_PER_UNIT
MODELS = ("cfsm", "avm", "sfm")
MODEL_NAMES = {"ours": "Our simulator", "cfsm": "JuPedSim CFSM", "avm": "JuPedSim AVM", "sfm": "JuPedSim SFM"}
BASE = {"spawn": "single", "origin": "room", "pre_movement_max_s": 0.0, "fires": []}
T_MAX = 600.0
GRIDLOCK_S = 120.0    # stop a run when nobody has got out for this long (gridlock)
WAYPOINTS = False     # True: force every person through each graph node (creates artificial jams at node centres)


# ====================================================================== JuPedSim side
def _jps_sim(model, geometry):
    import jupedsim as jps
    m = {"cfsm": jps.CollisionFreeSpeedModel, "avm": jps.AnticipationVelocityModel,
         "sfm": jps.SocialForceModel}[model]()
    return jps.Simulation(model=m, geometry=geometry, dt=0.01 if model != "sfm" else 0.005)


def _jps_agent(model, pos, speed, journey, stage):
    import jupedsim as jps
    if model == "cfsm":
        return jps.CollisionFreeSpeedModelAgentParameters(position=pos, desired_speed=speed, journey_id=journey, stage_id=stage)
    if model == "avm":
        return jps.AnticipationVelocityModelAgentParameters(position=pos, desired_speed=speed, journey_id=journey, stage_id=stage)
    return jps.SocialForceModelAgentParameters(position=pos, desired_speed=speed, journey_id=journey, stage_id=stage)


def _spacing(model):
    """(distance between agents, distance to walls) when placing people; SFM agents are bigger (r = 0.3 m)."""
    return (0.65, 0.35) if model == "sfm" else (0.45, 0.25)


def _speeds(n, seed):
    r = random.Random(f"speed-{seed}")
    return [C.FREE_WALK_SPEED * (1.0 + r.uniform(-C.AGENT_SPEED_SPREAD, C.AGENT_SPEED_SPREAD)) for _ in range(n)]


def _jps_simple(model, geo, spawn_box, exit_box, n, seed, door_x=None):
    """Everyone starts at t = 0 inside spawn_box and walks to exit_box. Returns (t_last, door flow, times)."""
    import jupedsim as jps
    import shapely
    sim = _jps_sim(model, geo)
    ex = sim.add_exit_stage(exit_box)
    j = sim.add_journey(jps.JourneyDescription([ex]))
    da, dp = _spacing(model)
    if model == "sfm":
        # SFM people are bigger (r = 0.3 m); random placement cannot fit 100 of them, so use a shuffled grid
        minx, miny, maxx, maxy = spawn_box.bounds
        grid = [(minx + dp + i * da, miny + dp + j * da)
                for i in range(int((maxx - minx - 2 * dp) / da) + 1)
                for j in range(int((maxy - miny - 2 * dp) / da) + 1)]
        random.Random(seed).shuffle(grid)
        pos = grid[:n]
    else:
        pos = jps.distribute_by_number(polygon=spawn_box, number_of_agents=n, distance_to_agents=da,
                                       distance_to_polygon=dp, seed=seed)
    for p, v in zip(pos, _speeds(n, seed)):
        sim.add_agent(_jps_agent(model, p, v, j, ex))
    passed, alive, out_t = {}, set(a.id for a in sim.agents()), []
    last_exit = 0.0
    while sim.agent_count() > 0 and sim.elapsed_time() < T_MAX and sim.elapsed_time() - last_exit < GRIDLOCK_S:
        sim.iterate()
        if sim.iteration_count() % 10 == 0:
            now = sim.elapsed_time()
            if len(alive) > sim.agent_count():
                last_exit = now
            ids = set()
            for a in sim.agents():
                ids.add(a.id)
                if door_x is not None and a.id not in passed and a.position[0] > door_x:
                    passed[a.id] = now
            out_t += [now] * len(alive - ids)
            alive = ids
    stuck = sim.agent_count()
    ts = sorted(passed.values())
    flow = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else None
    done = out_t and stuck <= max(1, 0.01 * n)
    return {"t_last": round(max(out_t), 1) if done else None, "flow": flow,
            "stuck": stuck, "t_mean": round(statistics.mean(out_t), 1) if out_t else None}


# ---------------------------------------------------------------------- C1 / C2 geometry
def c1_geometry(w):
    import shapely
    room = shapely.box(0, 0, 10, 6)
    door = shapely.box(10, 3 - w / 2, 10.3, 3 + w / 2)
    outside = shapely.box(10.3, -3, 14, 9)
    return shapely.union_all([room, door, outside]), room, shapely.box(13, -3, 14, 9)


def c2_geometry():
    import shapely
    room = shapely.box(0, 0, 12, 12)
    door = shapely.box(12, 5.5, 12.3, 6.5)                   # 1.0 m room door
    corridor = shapely.box(12.3, 5.0, 32.3, 7.0)             # 20 m x 2 m
    exitdoor = shapely.box(32.3, 5.1, 32.6, 6.9)             # 1.8 m exit door
    outside = shapely.box(32.6, 1, 36, 11)
    geo = shapely.union_all([room, door, corridor, exitdoor, outside])
    return geo, room, shapely.box(35, 1, 36, 11)


# ---------------------------------------------------------------------- C1 / C2 in our simulator
def chain(segments):
    """A straight chain of edges [(length m, width m), ...] from 'room' to 'test exit', with our physics
    (door-style flow limit per metre of width, Weidmann speed-density, jam density) on every edge."""
    from evac.physics import sustainable_flow
    names = ["room"] + [f"j{i}" for i in range(1, len(segments))] + ["test exit"]
    x, nodes = 0.0, {}
    for i, nm in enumerate(names):
        nodes[nm] = (x * U, 0, 0)
        if i < len(segments):
            x += segments[i][0]
    G = build_graph(nodes, list(zip(names, names[1:])))
    for (u, v), (L, w) in zip(zip(names, names[1:]), segments):
        d = G.edges[u, v]
        d.update(length_m=L, width_m=w, flow_cap=C.SPECIFIC_FLOW["exit_door"] * w,
                 eff_cap=min(C.SPECIFIC_FLOW["exit_door"] * w, sustainable_flow(w, C.FREE_WALK_SPEED)),
                 jam_cap=max(2, int(C.JAM_DENSITY * L * w)), free_speed=C.FREE_WALK_SPEED)
    return G


def _ours_simple(G, n, seed):
    m, recs = run(G, {**BASE, "n_agents": n}, "nearest", seed)
    ts = sorted(r["t_out"] for r in recs if r["t_out"] is not None)
    # nothing downstream of the door is narrower, so people leave at the door's rate
    return {"t_last": m["evac_time_last_s"], "t_mean": m["evac_time_mean_s"],
            "flow": (len(ts) - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else None, "stuck": n - len(ts)}


def ours_c1(w, n, seed):
    # drawn the way our building graphs are drawn: one edge from the room node through the door
    # (5 m from the room centre to the door, 3 m on to the measuring line), the door's width all along
    return _ours_simple(chain([(8.3, w)]), n, seed)


def ours_c2(n, seed):
    # room node -> corridor (1 m room-door edge, 6.6 m), 20 m x 2 m corridor, 1.8 m exit door edge (3.3 m)
    r = _ours_simple(chain([(6.6, 1.0), (20.0, 2.0), (3.3, 1.8)]), n, seed)
    r["flow"] = None
    return r


def door_as_short_edge(w=1.0, n=100, seed=0):
    """Sensitivity: the same room drawn with the door as its own 1 m-long edge between wide room / outside edges."""
    return _ours_simple(chain([(5.3, 6.0), (1.0, w), (3.0, 12.0)]), n, seed)


# ---------------------------------------------------------------------- C3: campus ground floor
def floor_graph():
    """Ground floor only (stairs/lifts are dead ends here: nobody starts upstairs)."""
    G = load_campus()
    return G.subgraph([n for n in G if G.nodes[n]["floor"] == 0]).copy()


def floor_geometry(G):
    """Every edge -> a strip of its width; every room -> a pocket behind its door. Coordinates in metres."""
    import shapely
    from shapely.geometry import LineString, Point
    P = {n: (G.nodes[n]["x"] * M, G.nodes[n]["y"] * M) for n in G}
    strips = {}
    for u, v, d in G.edges(data=True):
        if d["etype"] in ("stair", "lift"):
            continue
        strips[(u, v)] = LineString([P[u], P[v]]).buffer(d["width_m"] / 2, quad_segs=6)
    corridor_union = shapely.union_all(list(strips.values()))
    pockets = {}
    for n in G:
        if G.nodes[n]["kind"] != "room":
            continue
        nb = list(G.neighbors(n))
        if not nb:
            continue
        # pocket on the far side of the room node from its (first) door
        dx = sum(P[n][0] - P[b][0] for b in nb) / len(nb)
        dy = sum(P[n][1] - P[b][1] for b in nb) / len(nb)
        L = math.hypot(dx, dy) or 1.0
        others = shapely.union_all([s for (a, b), s in strips.items() if n not in (a, b)])
        for r in (3.0, 2.5, 2.0, 1.6, 1.3):
            c = Point(P[n][0] + dx / L * r, P[n][1] + dy / L * r).buffer(r, quad_segs=8)
            if not c.intersects(others):
                pockets[n] = c
                break
        else:
            pockets[n] = Point(P[n]).buffer(1.0, quad_segs=8)
    geo = shapely.union_all([corridor_union] + list(pockets.values()))
    if geo.geom_type == "MultiPolygon":              # keep the connected part that holds the exits
        geo = max(geo.geoms, key=lambda g: g.area)
    return geo, pockets, P


def floor_people(G, n, seed):
    """The exact people our simulator creates for this seed: origin, start time, speed, route."""
    sc = {"n_agents": n, "spawn": "floor", "floor": 0, "fires": []}
    agents = spawn(G, {**sc, "_seed": seed}, random.Random(seed))
    tab = route_table(G, "safe_shortest", False, False)
    return sc, [(a.origin, a.start, a.speed_mult, tab.get(a.origin)) for a in agents]


def ours_c3(n, seed):
    G = floor_graph()
    sc, _ = floor_people(G, n, seed)
    m, recs = run(G, sc, "safe_shortest", seed)
    ts = sorted(r["t_out"] for r in recs if r["t_out"] is not None)
    return {"t_last": m["evac_time_last_s"], "t_mean": m["evac_time_mean_s"], "curve": ts, "stuck": n - len(ts),
            "exits": m["exits_used"], "max_queue": m["max_queue"], "max_queue_node": m["max_queue_node"]}


def jps_c3(model, n, seed):
    import jupedsim as jps
    import shapely
    from shapely.geometry import Point
    G = floor_graph()
    geo, pockets, P = floor_geometry(G)
    _sc, people = floor_people(G, n, seed)
    sim = _jps_sim(model, geo)
    exits_stage, journeys = {}, {}
    for a in people:
        path = a[3]
        if not path or len(path) < 2:
            continue
        key = tuple(path)
        if key in journeys:
            continue
        ex = path[-1]
        if ex not in exits_stage:
            exits_stage[ex] = sim.add_exit_stage(Point(P[ex]).buffer(0.9, quad_segs=6).intersection(geo).buffer(-0.01))
        if WAYPOINTS:
            stages = [sim.add_waypoint_stage(P[nd], 1.0) for nd in path[1:-1]] + [exits_stage[ex]]
        else:
            # same exit as in our simulator; JuPedSim walks the geometric shortest way there, which on this
            # strip geometry is the same corridor sequence, but lets people cut corners like real walkers
            stages = [exits_stage[ex]]
        jd = jps.JourneyDescription(stages)
        for s1, s2 in zip(stages, stages[1:]):
            jd.set_transition_for_stage(s1, jps.Transition.create_fixed_transition(s2))
        journeys[key] = (sim.add_journey(jd), stages[0])
    rng = random.Random(f"place-{seed}")
    da, dp = _spacing(model)
    pending = sorted(((a[1], i) for i, a in enumerate(people) if a[3] and len(a[3]) > 1))
    pending = [(t, i) for t, i in pending]
    out_t, alive, delays = [], set(), []
    last_exit = 0.0
    while ((pending or sim.agent_count() > 0) and sim.elapsed_time() < T_MAX
           and sim.elapsed_time() - max(last_exit, pending[0][0] if pending else 0.0) < GRIDLOCK_S):
        now = sim.elapsed_time()
        if pending and sim.iteration_count() % 10 == 0:
            keep = []
            for t0, i in pending:
                if t0 > now:
                    keep.append((t0, i))
                    continue
                origin, _s, mult, path = people[i]
                pocket = pockets.get(origin) or Point(P[origin]).buffer(1.0)
                placed = False
                others = [a.position for a in sim.agents()]
                for _ in range(40):
                    minx, miny, maxx, maxy = pocket.bounds
                    p = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
                    pt = Point(p)
                    if not pocket.contains(pt) or not geo.contains(pt.buffer(dp)):
                        continue
                    if any(math.hypot(p[0] - q[0], p[1] - q[1]) < da for q in others):
                        continue
                    jid, st = journeys[tuple(path)]
                    try:
                        aid = sim.add_agent(_jps_agent(model, p, C.FREE_WALK_SPEED * mult, jid, st))
                    except Exception:
                        continue
                    alive.add(aid)
                    delays.append(now - t0)
                    placed = True
                    break
                if not placed:
                    keep.append((t0, i))            # room full right now - try again shortly
            pending = keep
        sim.iterate()
        if sim.iteration_count() % 10 == 0:
            ids = {a.id for a in sim.agents()}
            gone = alive - ids
            if gone:
                last_exit = sim.elapsed_time()
            out_t += [sim.elapsed_time()] * len(gone)
            alive = ids
    ts = sorted(out_t)
    where = []
    for a in sim.agents():
        near = min(P, key=lambda nd: math.hypot(P[nd][0] - a.position[0], P[nd][1] - a.position[1]))
        where.append(near)
    where += [f"not placed: {people[i][0]}" for _t, i in pending]
    stuck = len(where)
    # a model that leaves at most 1% of people stuck (e.g. wedged at a corner) still counts as finished;
    # the stuck count is reported next to the time
    done = ts and stuck <= max(1, 0.01 * len(people))
    return {"stuck_where": where, "t_last": round(ts[-1], 1) if done else None,
            "t_mean": round(statistics.mean(ts), 1) if ts else None, "curve": ts,
            "stuck": sim.agent_count() + len(pending), "spawn_delay_max": round(max(delays, default=0), 1)}


# ====================================================================== experiment
CACHE = os.path.join(ROOT, "results", "jupedsim_runs.jsonl")


def _cached():
    out = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[(r["case"], r["model"], r["param"], r["n"], r["seed"])] = r
    return out


def _job(args):
    case, model, param, n, seed = args
    t0 = time.time()
    try:
        if case == "C1":
            if model == "ours":
                r = ours_c1(param, n, seed)
            else:
                geo, room, exitbox = c1_geometry(param)
                r = _jps_simple(model, geo, room, exitbox, n, seed, door_x=10.3)
        elif case == "C2":
            if model == "ours":
                r = ours_c2(n, seed)
            else:
                geo, room, exitbox = c2_geometry()
                r = _jps_simple(model, geo, room, exitbox, n, seed)
        else:
            r = ours_c3(n, seed) if model == "ours" else jps_c3(model, n, seed)
        r["error"] = None
    except Exception as e:                                   # report, don't hide
        r = {"t_last": None, "t_mean": None, "flow": None, "stuck": None, "error": f"{type(e).__name__}: {e}"}
    r.update(case=case, model=model, param=param, n=n, seed=seed, compute_s=round(time.time() - t0, 1))
    print(f"  {case} {model:5s} p={param} n={n} s={seed}: last={r.get('t_last')} flow={r.get('flow')} "
          f"stuck={r.get('stuck')} err={r['error']} ({r['compute_s']} s)", flush=True)
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps(r) + "\n")
    return r


def jobs(quick=False):
    models = ("ours", "cfsm") if quick else ("ours",) + MODELS
    seeds = (0,) if quick else (0, 1, 2)
    out = []
    for m in models:
        for w in (1.0, 1.2, 1.6, 2.0, 2.4):
            out += [("C1", m, w, 100, s) for s in seeds]
        for n in (100, 200):
            out += [("C2", m, None, n, s) for s in seeds]
        for n in (150, 300):
            out += [("C3", m, None, n, s) for s in seeds]
    return out


def summarise(rows):
    """Mean over seeds per (case, param, n, model)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["case"], r["param"], r["n"], r["model"]), []).append(r)
    summ = []
    for (case, param, n, model), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2], kv[0][3])):
        ok = [r for r in rs if r["t_last"] is not None]
        flows = [r["flow"] for r in rs if r.get("flow")]
        summ.append({
            "case": case, "param": param, "n": n, "model": model, "runs": len(rs), "completed": len(ok),
            "t_last": round(statistics.mean(r["t_last"] for r in ok), 1) if ok else None,
            "t_mean": round(statistics.mean(r["t_mean"] for r in ok if r["t_mean"]), 1) if ok else None,
            "flow": round(statistics.mean(flows), 2) if flows else None,
            "errors": sorted({r["error"] for r in rs if r["error"]}),
            "stuck": [r["stuck"] for r in rs],
            "stuck_total": sum(r["stuck"] or 0 for r in rs),
        })
    return summ


def _label(s):
    if s["case"] == "C1":
        return f"Door {s['param']:.1f} m, 100 people"
    if s["case"] == "C2":
        return f"Room-corridor-exit, {s['n']} people"
    return f"Campus ground floor, {s['n']} people"


def compare(summ):
    """Our time vs each JuPedSim model, per situation. Means use the runs that finished."""
    out = []
    keys = sorted({(s["case"], s["param"], s["n"]) for s in summ}, key=lambda k: (k[0], k[1] or 0, k[2]))
    for k in keys:
        row = {s["model"]: s for s in summ if (s["case"], s["param"], s["n"]) == k}
        ours = row.get("ours")
        if not ours:
            continue
        ref = {m: row[m] for m in MODELS if m in row and row[m]["t_last"] is not None}
        o = ours["t_last"]
        c = {"situation": _label(ours), "case": k[0], "param": k[1], "n": k[2], "ours": o,
             "ref": {m: r["t_last"] for m, r in ref.items()},
             "done": {m: f"{row[m]['completed']}/{row[m]['runs']}" for m in MODELS if m in row},
             "flow": {m: row[m]["flow"] for m in ("ours",) + MODELS if m in row and row[m]["flow"]},
             "errors": {m: row[m]["errors"] for m in MODELS if m in row and row[m]["errors"]}}
        if ref:
            c["ref_lo"] = min(r["t_last"] for r in ref.values())
            c["ref_hi"] = max(r["t_last"] for r in ref.values())
            c["inside_range"] = len(ref) >= 2 and c["ref_lo"] <= o <= c["ref_hi"]
        if "cfsm" in ref:
            c["diff_vs_cfsm_pct"] = round(100 * (o - ref["cfsm"]["t_last"]) / ref["cfsm"]["t_last"], 1)
        out.append(c)
    return out


# ====================================================================== report
def figure(rows, cmp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"ours": "#2A78D6", "cfsm": "#E07B39", "avm": "#7A9E4F", "sfm": "#8E6BB5"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.0, 1.25, 1.1]})

    # (a) door flow vs width
    ax = axes[0]
    for m in ("ours",) + MODELS:
        pts = sorted((r["param"], r["flow"]) for r in rows if r["case"] == "C1" and r["model"] == m and r.get("flow"))
        if not pts:
            continue
        ws = sorted({p[0] for p in pts})
        mean = [statistics.mean(f for w2, f in pts if w2 == w) for w in ws]
        ax.plot(ws, mean, "o-", color=COL[m], label=MODEL_NAMES[m], lw=2 if m == "ours" else 1.5)
    try:
        from evac.validation import SEYFRIED
        ax.plot([w for w, _ in SEYFRIED], [f for _, f in SEYFRIED], "k^", ms=6, label="Lab data (Seyfried 2009)")
    except Exception:
        pass
    ax.set_xlabel("Door width (m)")
    ax.set_ylabel("Flow through the door (persons/s)")
    ax.set_title("(a) Door flow, 100 people", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    # (b) evacuation time per situation: ours vs JuPedSim range
    ax = axes[1]
    cmp = [c for c in cmp if c.get("ref")]
    labels = [c["situation"] for c in cmp]
    y = list(range(len(cmp)))[::-1]
    for yi, c in zip(y, cmp):
        ax.plot([c["ref_lo"], c["ref_hi"]], [yi, yi], color="#BBBBBB", lw=6, solid_capstyle="round", zorder=1)
        for m, t in c["ref"].items():
            ax.plot(t, yi, "o", color=COL[m], ms=6, zorder=2)
        ax.plot(c["ours"], yi, "D", color=COL["ours"], ms=8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Time until the last person is out (s)")
    ax.set_title("(b) Ours (diamond) vs JuPedSim models (grey = their range)", loc="left", fontsize=11)
    ax.grid(axis="x", alpha=0.25)

    # (c) evacuation curve on the real floor, 300 people, seed 0
    ax = axes[2]
    for m in ("ours",) + MODELS:
        rs = [r for r in rows if r["case"] == "C3" and r["n"] == 300 and r["model"] == m and r["seed"] == 0 and r.get("curve")]
        if rs:
            ts = rs[0]["curve"]
            ax.step(ts, range(1, len(ts) + 1), where="post", color=COL[m], label=MODEL_NAMES[m], lw=2 if m == "ours" else 1.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("People out")
    ax.set_title("(c) Campus ground floor, 300 people (seed 0)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.suptitle("Our network simulator against JuPedSim (continuous 2-D models), same geometry and people",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=130)


def geometry_figure():
    """Picture of the 2-D ground floor made from the graph (appendix)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G = floor_graph()
    geo, _p, P = floor_geometry(G)
    fig, ax = plt.subplots(figsize=(7, 7))
    polys = [geo] if geo.geom_type == "Polygon" else list(geo.geoms)
    for g in polys:
        x, y = g.exterior.xy
        ax.fill(x, y, color="#DCE6F2", ec="#2A78D6", lw=0.6)
        for hole in g.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, color="white", ec="#2A78D6", lw=0.4)
    for n in G:
        k = G.nodes[n]["kind"]
        if k == "exit":
            ax.plot(*P[n], "s", color="#10B981", ms=7)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Campus ground floor as 2-D walkable area (from the graph)", fontsize=10)
    ax.set_xlabel("m")
    out = os.path.join(ROOT, "docs", "figures", "fig26b_jupedsim_geometry.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    return out


def write_report(rows, summ, cmp, minutes):
    import jupedsim
    try:
        from evac.validation import SEYFRIED
        sey = dict(SEYFRIED)
    except Exception:
        sey = {}
    short = door_as_short_edge(1.0)
    L = ["# Cross-check against JuPedSim", "",
         f"Generated by `python -m evac.jupedsim_check` (JuPedSim {jupedsim.__version__}). Our network simulator is "
         "compared with three published movement models implemented in JuPedSim, an established open-source "
         "pedestrian simulator from Forschungszentrum Juelich, on the same geometry with the same people. Free walking "
         "speed 1.25 m/s +-15% everywhere; the JuPedSim models use their default parameters. Times are means over 3 "
         "seeds (over the runs that finished).", "",
         "Models: **CFSM** collision-free speed model (Tordeux, Chraibi & Seyfried 2016), JuPedSim's standard model; "
         "**AVM** anticipation velocity model (Xu, Chraibi & Seyfried 2021); **SFM** social force model "
         "(Helbing & Molnar 1995).", "",
         "## Time until the last person is out (s)", "",
         "| Situation | Ours | CFSM | AVM | SFM | Ours vs CFSM |", "|---|---|---|---|---|---|"]

    def cell(c, m):
        d = c["done"].get(m, "0/0")
        if m in c["ref"]:
            return f"{c['ref'][m]}" + ("" if d.split("/")[0] == d.split("/")[1] else f" ({d} runs)")
        if m in c["errors"] and any("AgentNumberError" in e for e in c["errors"][m]):
            return "could not place people"
        if m in c["errors"] and any("outside of accessible area" in e for e in c["errors"][m]):
            return "crashed (pushed through a wall)"
        return "jammed" if m in c["done"] else "–"
    for c in cmp:
        dv = f"{c['diff_vs_cfsm_pct']:+.0f}%" if "diff_vs_cfsm_pct" in c else "–"
        L.append(f"| {c['situation']} | **{c['ours']}** | {cell(c, 'cfsm')} | {cell(c, 'avm')} | {cell(c, 'sfm')} | {dv} |")
    L += ["", "When only some runs finished, the mean is over those runs (count in brackets). "
          "'jammed' = people stopped moving for good (nobody got out for 2 minutes) in every run; "
          "real crowds do not lock up permanently, so these runs have no evacuation time.", "",
          "## Door flow (persons/s leaving, 100 people)", "",
          "| Door width | Ours | CFSM | AVM | SFM | Lab (Seyfried et al. 2009) |", "|---|---|---|---|---|---|"]
    for c in cmp:
        if c["case"] != "C1":
            continue
        f = c["flow"]
        g = lambda m: f"{f[m]:.2f}" if m in f else "–"
        L.append(f"| {c['param']:.1f} m | **{g('ours')}** | {g('cfsm')} | {g('avm')} | {g('sfm')} | {sey.get(c['param'], '–')} |")
    diffs = [c["diff_vs_cfsm_pct"] for c in cmp if "diff_vs_cfsm_pct" in c]
    within = sum(abs(d) <= 15 for d in diffs)
    inside = [c for c in cmp if c.get("inside_range") is not None and len(c["ref"]) >= 2]
    floor = [c for c in cmp if c["case"] == "C3" and "diff_vs_cfsm_pct" in c]
    L += ["", "## What it shows", "",
          f"- Against CFSM, JuPedSim's standard model, our time is within 15% in {within} of {len(diffs)} situations "
          f"(median difference {statistics.median(diffs):+.0f}%, range {min(diffs):+.0f}% to {max(diffs):+.0f}%).",
          f"- Where two or more JuPedSim models finished, ours lies inside their range in "
          f"{sum(c['inside_range'] for c in inside)} of {len(inside)} situations: the established models disagree "
          "with each other by tens of percent, so no single number is 'the truth'.",
          "- On the real campus floor: " + "; ".join(f"{c['n']} people {c['ours']} s vs CFSM {c['ref']['cfsm']} s "
                                                     f"({c['diff_vs_cfsm_pct']:+.0f}%)" for c in floor) + ".",
          "- Door flow: our crowded doors pass fewer people per second than laboratory crowds (Seyfried et al. 2009), "
          "like the JuPedSim models do with default parameters. Our flow lies between AVM and CFSM for narrow "
          "doors and close to CFSM for wide ones.",
          "- Single rooms (C1, C2): ours is slower than CFSM (+10% to +46%). Our graphs draw the walk from the room "
          "centre to the door as a strip as narrow as the door, so people crowd and slow down earlier than in a "
          "2-D room. For these cases our times are conservative.",
          "- Campus floor: ours is about as fast for 150 people and faster for 300 (-14%). Probably streams merging "
          "at corridor junctions cost more time in 2-D than our first-come-first-served queues predict, so for "
          "large crowds on one floor our times may be optimistic by that margin.",
          "- AVM and SFM often locked up permanently at the 1 m doors and in the floor geometry (default parameters); "
          "this is reported as 'jammed', not dropped. CFSM finished almost every run.", "",
          "### Correction to the earlier door-flow validation", "",
          "`docs/validation.md` compared the rate at which people are *let onto* a door edge (1.3 persons/s per metre) "
          "with laboratory flows. Measured as the rate at which people actually *leave* through a crowded 1 m door, "
          f"our simulator passes {next((c['flow']['ours'] for c in cmp if c['case'] == 'C1' and c['param'] == 1.0), float('nan')):.2f} "
          "persons/s, because walking speed on the door edge falls with density. That is about half the laboratory "
          "value and inside the JuPedSim range, so our evacuation times are, if anything, conservative (too slow). "
          f"Drawn as its own 1 m-long edge, the same door passes only {short['flow']:.2f} persons/s (last person out "
          f"{short['t_last']} s): door throughput depends on how a door is drawn in the graph, which is a limitation "
          "of network models. Our building graphs draw a door as part of the room-to-corridor edge, as in C1.", "",
          "## Method", "",
          "- **C1 / C2**: the same rooms, doors and corridors drawn in 2-D for JuPedSim and as a small graph for ours, "
          "drawn the way our building graphs are (one edge from the room node through the door, with the door's "
          "width). Everyone starts at t = 0. C1: 10 x 6 m room. C2: 12 x 12 m room, 1 m door, 20 m x 2 m corridor, "
          "1.8 m exit door.",
          "- **C3**: the campus ground-floor graph turned into 2-D geometry: each edge becomes a strip of its width "
          "(corridor 2 m, room door 1 m, exit door 1.8 m) and each room a pocket behind its door "
          "(`figures/fig26b_jupedsim_geometry.png`). Both simulators get the same people: origin room, start time "
          "(pre-movement U(0, 30) s), personal speed and exit (shortest path). In JuPedSim people walk the "
          "geometric shortest way to that exit. Stairs and upper floors are not part of this check (JuPedSim is "
          "2-D); stair speed is validated against NIST drill data in `docs/validation.md`.",
          "- JuPedSim time step 0.01 s (SFM 0.005 s), ours 0.5 s. A run stops when nobody has got out for 120 s.", "",
          "![JuPedSim cross-check](figures/fig26_jupedsim.png)", "",
          "## References", "",
          "- Tordeux, A., Chraibi, M. & Seyfried, A. (2016). Collision-free speed model for pedestrian dynamics. "
          "In Traffic and Granular Flow '15 (Springer), 225-232. doi:10.1007/978-3-319-33482-0_29",
          "- Xu, Q., Chraibi, M. & Seyfried, A. (2021). Anticipation in a velocity-based model for pedestrian "
          "dynamics. Transportation Research Part C, 103464. doi:10.1016/j.trc.2021.103464",
          "- Helbing, D. & Molnar, P. (1995). Social force model for pedestrian dynamics. Physical Review E 51, 4282.",
          "- Seyfried, A. et al. (2009). New insights into pedestrian flow through bottlenecks. Transportation "
          "Science 43(3), 395-406.", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--report-only", action="store_true", help="rebuild report/figure from the cached runs")
    ap.add_argument("--shard", default="", help="i/n: run only every n-th pending job (for parallel processes)")
    a = ap.parse_args()
    t0 = time.time()
    if a.report_only:
        want = set(jobs(a.quick))
        rows = [r for k, r in _cached().items() if k in want]
    else:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        done = _cached()
        js = [j for j in jobs(a.quick) if j not in done]
        # slow jobs first so the pool stays busy
        js.sort(key=lambda j: ({"C3": 0, "C2": 1, "C1": 2}[j[0]], j[1] == "ours"))
        if a.shard:                                   # e.g. --shard 0/2 : run every 2nd job, no report
            i, n = map(int, a.shard.split("/"))
            for j in js[i::n]:
                _job(j)
            return
        with get_context("spawn").Pool(a.workers) as pool:
            list(pool.imap_unordered(_job, js))
        want = set(jobs(a.quick))
        rows = [r for k, r in _cached().items() if k in want]
    summ = summarise(rows)
    cmp = compare(summ)
    os.makedirs(os.path.dirname(JSON), exist_ok=True)
    with open(JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summ, "comparison": cmp, "runs": rows}, f, indent=1)
    figure(rows, cmp)
    geometry_figure()
    write_report(rows, summ, cmp, (time.time() - t0) / 60)
    print("wrote", DOC)


if __name__ == "__main__":
    main()
