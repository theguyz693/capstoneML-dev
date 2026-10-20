"""
evac/simulator.py
Headless, seeded, discrete-time evacuation simulator on the building graph.

What it models (each step of DT seconds):
  - pre-movement: each agent starts after a random delay
  - queues: an agent may ENTER an edge only if the edge's flow budget allows
    (flow_cap persons/second) and the edge is not at jam capacity; others wait
    in a first-come-first-served queue at the node
  - speed: falls with density on the edge (Weidmann fundamental diagram)
  - hazard: crossing an edge with hazard >= HAZARD_EXPOSURE counts as exposure
  - step-free agents with no lift/ramp route are marked 'needs_assistance'
    (in a real fire they wait at a refuge for assisted evacuation)
  - behaviour (Session 4): a share of people ignore the system and head for their familiar
    exit; some of those copy the majority exit of the queue they join (herding)
  - spreading fire (Session 4): hazard is recomputed every HAZARD_UPDATE_S; anyone about to
    step onto an edge that has become impassable turns to the safest route from where they stand

Same scenario + same seed -> identical results.
"""

import math
import random
import time
from collections import defaultdict

from . import config as C
from .building import room_nodes
from .hazard import apply_hazard, is_dynamic
from .routing import route_table, congestion_route_table
from . import planner
from . import learned_planner


from .physics import weidmann_speed_factor, smoke_speed_factor, edge_density  # noqa: E402


def _ekey(u, v):
    return (u, v) if u <= v else (v, u)


class Agent:
    __slots__ = ("id", "origin", "path", "idx", "pos", "state", "start", "speed_mult",
                 "step_free", "queued_at", "t_out", "exposed", "max_hazard", "exit", "changes", "t_left",
                 "compliant", "herds", "edge_log", "size", "profile")

    def __init__(self, aid, origin, start, speed_mult, step_free, size=1.0, profile=None):
        self.id = aid
        self.origin = origin
        self.path = None
        self.idx = 0          # index of current node in path
        self.pos = 0.0        # metres along current edge
        self.state = "waiting"
        self.start = start
        self.speed_mult = speed_mult
        self.step_free = step_free
        self.queued_at = None
        self.t_out = None
        self.exposed = False
        self.max_hazard = 0.0
        self.exit = None
        self.changes = 0
        self.t_left = None    # time the agent left its origin room (passed the room door)
        self.compliant = True
        self.herds = False
        self.edge_log = None
        self.size = size      # person-equivalents of door/corridor capacity (a bed ~ 4)
        self.profile = profile or ("step_free" if step_free else "walk")


def _spawn_groups(G, scenario, rng):
    """'spawn': 'groups' - several groups, each with its own rooms and mix of occupant profiles:
       "groups": [{"where": "<regex on room names>", "n": 40, "mix": {"walk": 0.5, "bed": 0.5}}, ...]
    Profiles (speed, step-free, size, pre-movement) are in config.PROFILES."""
    import re
    rooms = room_nodes(G)
    spread = float(scenario.get("speed_spread", C.AGENT_SPEED_SPREAD))
    agents = []
    for g in scenario["groups"]:
        pool = [r for r in rooms if re.search(g["where"], r)]
        if not pool:
            raise ValueError(f"no rooms match {g['where']!r}")
        names, weights = zip(*sorted(g.get("mix", {"walk": 1.0}).items()))
        for _ in range(int(g["n"])):
            prof = rng.choices(names, weights)[0]
            P = C.PROFILES[prof]
            agents.append(Agent(
                aid=len(agents), origin=pool[rng.randrange(len(pool))],
                start=rng.uniform(*P["pre_s"]),
                speed_mult=P["speed"] * (1.0 + rng.uniform(-spread, spread)),
                step_free=P["step_free"], size=P["size"], profile=prof))
    return agents


def spawn(G, scenario, rng):
    """Create agents. scenario keys: n_agents, spawn ('building'|'single'|'floor'|'groups'), origin, floor,
    step_free_frac, groups"""
    if scenario.get("spawn") == "groups":
        agents = _spawn_groups(G, scenario, rng)
        brng = random.Random(f"behaviour-{scenario.get('_seed', 0)}")
        p = float(scenario.get("compliance", C.COMPLIANCE))
        h = float(scenario.get("herding", C.HERDING))
        for a in agents:
            a.compliant = brng.random() < p
            a.herds = (not a.compliant) and brng.random() < h
        return agents
    rooms = room_nodes(G)
    n = int(scenario.get("n_agents", 100))
    mode = scenario.get("spawn", "building")
    if mode == "single":
        origin = scenario.get("origin", "classroom g17")
        origins = [origin] * n
    elif mode == "floor":
        fl = int(scenario.get("floor", 0))
        pool = [r for r in rooms if G.nodes[r]["floor"] == fl] or rooms
        origins = [pool[rng.randrange(len(pool))] for _ in range(n)]
    else:  # whole building, spread across floors
        by_floor = defaultdict(list)
        for r in rooms:
            by_floor[G.nodes[r]["floor"]].append(r)
        floors = sorted(by_floor)
        origins = []
        for i in range(n):
            pool = by_floor[floors[i % len(floors)]]
            origins.append(pool[rng.randrange(len(pool))])
    sf = float(scenario.get("step_free_frac", 0.0))
    pre = float(scenario.get("pre_movement_max_s", C.PRE_MOVEMENT_MAX_S))
    spread = float(scenario.get("speed_spread", C.AGENT_SPEED_SPREAD))
    agents = []
    for i, o in enumerate(origins):
        agents.append(Agent(
            aid=i, origin=o,
            start=rng.uniform(0.0, pre),
            speed_mult=1.0 + rng.uniform(-spread, spread),
            step_free=(rng.random() < sf),
        ))
    # behaviour uses its OWN random stream, so the same seed keeps the same people in the
    # same rooms whatever the compliance setting (paired comparisons stay paired)
    brng = random.Random(f"behaviour-{scenario.get('_seed', 0)}")
    p = float(scenario.get("compliance", C.COMPLIANCE))
    h = float(scenario.get("herding", C.HERDING))
    for a in agents:
        a.compliant = brng.random() < p
        a.herds = (not a.compliant) and brng.random() < h
    return agents


def _reactive_reroute(G, fire_active, queues, occ, waiting_agents):
    """Re-plan agents standing at a node whose remaining route crosses a visible queue."""
    qlen = {n: len(q) for n, q in queues.items() if q}
    hot = {n for n, k in qlen.items() if k >= C.REACTIVE_QUEUE_THRESHOLD}
    if not hot:
        return
    standing = [a for q in queues.values() for a in q] + list(waiting_agents)
    affected = [a for a in standing if a.compliant and a.path and any(n in hot for n in a.path[a.idx:-1])]
    if not affected:
        return
    qcount = {}
    for q in queues.values():
        for a in q:
            e = (a.path[a.idx], a.path[a.idx + 1])
            qcount[e] = qcount.get(e, 0) + 1
    tables = {}
    for a in affected:
        if a.step_free not in tables:
            tables[a.step_free] = congestion_route_table(G, fire_active, a.step_free, qcount, occ)
        new = tables[a.step_free].get(a.path[a.idx])
        if new and new != a.path[a.idx:]:
            a.path = a.path[:a.idx] + new
            a.exit = new[-1]
            a.changes += 1


def _has_edge(path, idx, u, v):
    for i in range(idx, len(path) - 1):
        if path[i] == u and path[i + 1] == v:
            return True
    return False


def _forced_detours(G, fire_active, queues, waiting_agents, own_route):
    """After the fire grows: anyone standing at a node whose NEXT edge is now impassable (or whose
    route crosses one) is re-routed from where they stand. Applies to every method - nobody walks
    into flames - and counts as a forced detour."""
    n = 0
    tables = {}
    standing = [a for q in queues.values() for a in q] + list(waiting_agents)
    for a in standing:
        rest = a.path[a.idx:]
        if all(G.edges[x, y].get("hazard", 0.0) < C.HAZARD_IMPASSABLE for x, y in zip(rest, rest[1:])):
            continue
        node = a.path[a.idx]
        if a.compliant:
            if a.step_free not in tables:
                tables[a.step_free] = route_table(G, "safe_shortest", fire_active, a.step_free)
            new = tables[a.step_free].get(node)
        else:
            new = own_route(a, node)
        if new:
            a.path = a.path[:a.idx] + new
            a.exit = new[-1]
            a.changes += 1
            n += 1
    return n


def crowd_state(agents, queues, occ, t):
    """Crowd snapshot as plain counts (what sensors could report): people queued per edge and per node,
    people on each edge, people whose route uses each edge."""
    qa, qnode, planned = defaultdict(int), {}, defaultdict(int)
    for node, q in queues.items():
        qnode[node] = len(q)
        for a in q:
            if a.idx < len(a.path) - 1:
                qa[(node, a.path[a.idx + 1])] += 1
    for a in agents:
        if a.path and a.state not in ("evacuated", "needs_assistance", "stranded", "in_place"):
            for i in range(a.idx, len(a.path) - 1):
                planned[(a.path[i], a.path[i + 1])] += 1
    return {"qa": dict(qa), "qnode": qnode, "occ": {k: v for k, v in occ.items() if v > 0},
            "planned": dict(planned), "t": t, "inside": sum(a.state in ("waiting", "queued", "moving") for a in agents)}


def _eta_probe(G, agents, queues, moving, occ, t, fire_active, idx_cache):
    """Time-to-exit predictions for everyone standing at a node at time t (Phase 2, evac/eta_study.py):
    'ml' = sum of learned crossing times on the current crowd, 'plan' = capacity bookings of all routes."""
    standing = [a for q in queues.values() for a in q if a.path and a.idx < len(a.path) - 1]
    walking = [a for a in moving if a.path and a.idx < len(a.path) - 1]
    if not standing and not walking:
        return []
    from .physics import smoke_speed_factor as _ssf
    from .planner import predict_arrivals
    plan = predict_arrivals(G, agents, walking, standing, t)
    if "idx" not in idx_cache:
        idx_cache["idx"] = learned_planner.EdgeIndex(G)
    st = crowd_state(agents, queues, occ, t)
    tables = {}
    rows = []
    for a in standing + walking:
        key = (a.step_free, round(a.speed_mult, 1), a.size)
        if key not in tables:
            tables[key] = learned_planner.eta_table(G, idx_cache["idx"], st, fire_active, a.step_free, key[1], a.size)[0]
        if a.state == "moving":                     # rest of the current edge, then from its end node
            u, v = a.path[a.idx], a.path[a.idx + 1]
            d = G.edges[u, v]
            rem = max(0.0, d["length_m"] - a.pos) / (d["free_speed"] * a.speed_mult * _ssf(d.get("hazard", 0.0)))
            node = v
        else:
            rem, node = 0.0, a.path[a.idx]
        ml = tables[key].get(node)
        rows.append({"agent": a.id, "t": t, "node": node, "state": a.state, "profile": a.profile,
                     "compliant": a.compliant, "queue_here": len(queues.get(node, ())), "inside": st["inside"],
                     "eta_ml": None if ml is None else round(rem + ml[1], 1),
                     "eta_ml_lo": None if ml is None or ml[2] is None else round(rem + ml[2], 1),
                     "eta_ml_hi": None if ml is None or ml[3] is None else round(rem + ml[3], 1),
                     "eta_plan": round(plan[a.id], 1) if a.id in plan else None})
    return rows


def _lift_shaft(name):
    return name.split("_", 1)[-1] if "_" in name else name


def configure_evac_lifts(G):
    """Mark evacuation lifts and give lift edges car-like physics: car speed, and a capacity (for the planner)
    of one car-load per round trip from the top floor (cautious: the planner assumes one group per trip)."""
    floors = max(d["floor"] for _n, d in G.nodes(data=True))
    cycle = C.LIFT_DOOR_LOAD_S + 2 * max(1, floors) * C.STAIR_FLIGHT_LENGTH_M / C.LIFT_SPEED
    for u, v, d in G.edges(data=True):
        if d["etype"] != "lift":
            continue
        d["evac_lift"] = any(k in u or k in v for k in C.EVAC_LIFT_NAMES)
        d["free_speed"] = C.LIFT_SPEED
        d["flow_cap"] = d["eff_cap"] = C.LIFT_CAR_CAPACITY / cycle
        d["jam_cap"] = C.LIFT_CAR_CAPACITY
    return G


def _board_lift(G, a, u, d, t, state):
    """Evacuation-lift car: people already in the car ride on to the next floor; a new group boards only when
    the car is back (free_at) or is loading at this floor at this moment and has room."""
    if a.idx > 0 and G.edges[a.path[a.idx - 1], u]["etype"] == "lift":
        return True                                  # still in the car
    sh = _lift_shaft(u)
    st = state.setdefault(sh, {"free_at": 0.0, "batch_t": None, "batch_floor": None, "load": 0.0})
    fl = G.nodes[u]["floor"]
    if (st["batch_t"] is not None and 0 <= t - st["batch_t"] <= C.LIFT_LOAD_WINDOW_S and st["batch_floor"] == fl
            and st["load"] + a.size <= C.LIFT_CAR_CAPACITY):          # car still loading here with room
        st["load"] += a.size
        return True
    if t >= st["free_at"] and a.size <= C.LIFT_CAR_CAPACITY:
        travel = fl * C.STAIR_FLIGHT_LENGTH_M / C.LIFT_SPEED
        st.update(batch_t=t, batch_floor=fl, load=a.size, free_at=t + C.LIFT_DOOR_LOAD_S + 2 * travel)
        return True
    return False


def run(G, scenario, method, seed):
    """Run one evacuation. Returns (metrics dict, per-agent records).
    method='optimum' returns the theoretical lower bound instead of a simulation."""
    if method == "optimum":
        return _optimum(G, scenario, seed)
    t0 = time.perf_counter()
    rng = random.Random(seed)
    fires = scenario.get("fires", [])
    fire_active = bool(fires)
    dynamic = is_dynamic(fires)
    if C.EVAC_LIFTS:
        configure_evac_lifts(G)
    lift_state = {}           # shaft -> {"free_at", "batch_t", "batch_floor", "load"}  (evacuation lifts)
    apply_hazard(G, fires, 0.0 if dynamic else None)

    agents = spawn(G, {**scenario, "_seed": seed}, rng)
    familiar = scenario.get("familiar_exit", C.FAMILIAR_EXIT)
    log_edges = bool(scenario.get("log_edges"))
    edge_rows = []

    # routes: one multi-source Dijkstra per (method, step_free) covers every origin
    base = ("safe_shortest" if method in ("reactive", "ccrp", "lookahead", "lookahead_ml", "fixed_plan",
                                          "quickest", "casper")
            or method.startswith("pred_") else method)
    if method == "lookahead_ml":
        edge_index = learned_planner.EdgeIndex(G)
    if method.startswith("pred_"):
        pred_kind = method[len("pred_"):]          # formula | ml | ml_upper
        edge_index = learned_planner.EdgeIndex(G)
    tables = {
        False: route_table(G, base, fire_active, step_free=False),
        True: route_table(G, base, fire_active, step_free=True),
    }
    for a in agents:
        p = tables[a.step_free].get(a.origin)
        if p is None:
            a.state = "needs_assistance" if a.step_free else "stranded"
        else:
            a.path = p
            a.exit = p[-1]

    def own_route(a, node):
        """Where a non-compliant person goes from `node`: their familiar exit if reachable safely."""
        for targets in ([familiar], None):
            tab = route_table(G, "safe_shortest", fire_active, a.step_free, targets=targets)
            if node in tab:
                return tab[node]
        return None

    for a in agents:
        if a.path is not None and not a.compliant:
            p = own_route(a, a.origin)
            if p:
                a.path, a.exit = p, p[-1]

    # hospital procedures (Session 7): defend in place, move to a refuge, limited staff for beds
    _hospital_procedures(G, scenario, agents, fire_active)

    # optional fixed exit assignment per origin room (used by the genetic-algorithm baseline)
    exit_plan = scenario.get("exit_plan")
    if exit_plan:
        cache = {}
        for a in agents:
            tgt = exit_plan.get(f"{a.origin}#{a.id % 4}", exit_plan.get(a.origin))
            if a.path is None or not a.compliant or tgt is None:
                continue
            key = (tgt, a.step_free)
            if key not in cache:
                cache[key] = route_table(G, "safe_shortest", fire_active, a.step_free, targets=[tgt])
            p = cache[key].get(a.origin)
            if p:
                a.path, a.exit = p, p[-1]

    # printed plan made without live data (Session 7, evac/fixed_plan.py)
    if method == "fixed_plan":
        from .fixed_plan import assign
        assign(G, agents, scenario["route_plan"], scenario.get("route_plan_mode", "split"), fire_active)

    # what the planner can see of the crowd (Session 7, evac/knowledge.py)
    from .knowledge import Sensor
    sensor = Sensor(scenario, seed, G)

    horizon = float(scenario.get("horizon_s", C.HORIZON_S))
    switch_gain = float(scenario.get("switch_gain", C.SWITCH_GAIN))
    planner.FAMILIAR_EXIT = familiar
    if method in ("ccrp", "lookahead", "lookahead_ml"):
        # 'oracle': the planner knows who follows it; otherwise it plans (and books) everyone the same way
        plannable = sensor.sense([a for a in agents if a.path is not None and
                                  (a.compliant or C.PLANNER_COMPLIANCE_MODEL != "oracle")])
        walk0 = (learned_planner.walk_predictions(G, edge_index, agents, {}, {})
                 if method == "lookahead_ml" else None)
        plans = planner.plan_all(G, plannable, fire_active,
                                 horizon=math.inf if method == "ccrp" else horizon, walk=walk0)
        for a in plannable:
            if a.id in plans and a.id >= 0 and a.compliant:   # negative ids are phantom (sensed but not real) people
                a.path = plans[a.id]
                a.exit = a.path[-1]

    # published baselines (evac/literature.py)
    qp = None
    if method == "casper":
        from .literature import casper_plan
        plannable = [a for a in sensor.sense([a for a in agents if a.path is not None and a.compliant]) if a.id >= 0]
        for aid, p in casper_plan(G, plannable, fire_active).items():
            ag = agents[aid]
            ag.path, ag.exit = p, p[-1]
    elif method == "quickest":
        from .literature import QuickestPath
        qp = QuickestPath(G, fire_active)

    budget = defaultdict(float)
    occ = defaultdict(int)
    queues = defaultdict(list)      # node -> agents waiting to leave it
    moving = []
    max_queue, max_queue_node = 0, None
    node_peak = defaultdict(int)          # longest queue seen at each node (jam points)
    jam_s = defaultdict(float)            # seconds each node had a queue of >= JAM_QUEUE people (evac/jam_study.py)
    wait_ps = defaultdict(float)          # person-seconds of waiting at each node
    jam_small_s = defaultdict(float)
    peak_density = 0.0

    def _join(a, node, now):
        """Agent arrives in the queue at `node`. Herding + delay-feature logging happen here."""
        if a.herds and queues[node]:
            votes = defaultdict(int)
            for b in queues[node]:
                votes[b.exit] += 1
            top = max(sorted(votes), key=lambda e: votes[e])
            if top != a.exit:
                tab = route_table(G, "safe_shortest", fire_active, a.step_free, targets=[top])
                p = tab.get(node)
                if p:
                    a.path = a.path[:a.idx] + p
                    a.exit = top
                    a.changes += 1
        if qp is not None:
            qp.on_arrival(a, now, queues, occ)
        queues[node].append(a)
        if log_edges and a.idx < len(a.path) - 1:
            u, v = a.path[a.idx], a.path[a.idx + 1]
            d = G.edges[u, v]
            k = _ekey(u, v)
            ahead = sum(1 for b in queues[node] if b is not a and b.idx < len(b.path) - 1
                        and b.path[b.idx + 1] == v)
            planned = sum(1 for b in agents if b is not a and b.path and b.state != "evacuated"
                          and _has_edge(b.path, b.idx, u, v))
            a.edge_log = {
                "scenario": scenario.get("_name", ""), "seed": seed, "fire_node": fires[0]["node"] if fires else "none",
                "etype": d["etype"], "length_m": d["length_m"], "width_m": d["width_m"],
                "eff_cap": round(d["eff_cap"], 3), "free_speed": d["free_speed"],
                "hazard": d.get("hazard", 0.0), "queue_ahead": ahead, "queue_at_node": len(queues[node]) - 1,
                "on_edge": occ[k], "density": round(edge_density(d, occ[k]), 3),
                "planned_users": planned, "mover_speed": round(a.speed_mult, 3), "mover_size": a.size,
                "_t_join": now,
            }

    trace_key = tuple(sorted(scenario["trace_edge"])) if scenario.get("trace_edge") else None
    edge_trace = []
    t = 0.0
    next_reroute = C.REROUTE_INTERVAL_S
    active = [a for a in agents if a.state == "waiting"]
    next_hazard = C.HAZARD_UPDATE_S
    forced = 0
    track_every = float(scenario.get("record_tracks", 0) or 0)
    next_frame = 0.0
    frames = []

    def snapshot(now):
        pts = []
        for a in agents:
            if a.state in ("evacuated", "needs_assistance", "stranded", "in_place") or a.path is None:
                continue
            if a.state == "moving":
                u, v = a.path[a.idx], a.path[a.idx + 1]
                f = min(1.0, a.pos / G.edges[u, v]["length_m"])
                nu, nv = G.nodes[u], G.nodes[v]
                x, y, z = (nu["x"] + f * (nv["x"] - nu["x"]), nu["y"] + f * (nv["y"] - nu["y"]),
                           nu["z"] + f * (nv["z"] - nu["z"]))
            else:
                n0 = G.nodes[a.path[a.idx]]
                x, y, z = n0["x"], n0["y"], n0["z"]
            pts.append([a.id, round(x), round(y), round(z), {"waiting": 0, "queued": 1, "moving": 2}[a.state],
                        int(a.compliant)])
        frames.append({"t": round(now, 1), "pts": pts,
                       "hazard": {n: G.nodes[n]["hazard"] for n in G if G.nodes[n]["hazard"] > 0} if dynamic else None})
    stop_at = float(scenario.get("stop_at_s", C.MAX_TIME_S))      # snapshot mode (dashboard route queries)
    eta_probe = sorted(scenario.get("eta_probe_at", []))            # time-to-exit study (evac/eta_study.py)
    eta_rows, eta_idx = [], {}
    while t < min(C.MAX_TIME_S, stop_at) and (active or moving or any(queues.values())):
        if track_every and t >= next_frame:
            next_frame += track_every
            snapshot(t)
        if eta_probe and t >= eta_probe[0]:
            eta_probe.pop(0)
            eta_rows += _eta_probe(G, agents, queues, moving, occ, t, fire_active, eta_idx)
            eta_probe = eta_probe or [math.inf]
        if dynamic and t >= next_hazard:
            next_hazard += C.HAZARD_UPDATE_S
            apply_hazard(G, fires, t)
            forced += _forced_detours(G, fire_active, queues, active, own_route)
        if method == "reactive" and t >= next_reroute:
            next_reroute += C.REROUTE_INTERVAL_S
            _reactive_reroute(G, fire_active, queues, occ, [a for a in active if a.compliant])
        elif method == "quickest" and t >= next_reroute:
            next_reroute += 1.0
            qp.jam_check(t, queues, occ)
        elif method.startswith("pred_") and t >= next_reroute:
            next_reroute += C.REPLAN_S
            learned_planner.replan(G, edge_index, pred_kind, agents, queues, occ, active, fire_active, switch_gain)
        elif method in ("lookahead", "lookahead_ml") and t >= next_reroute:
            next_reroute += C.REPLAN_S
            standing = [a for q in queues.values() for a in q] + list(active)
            walk = (learned_planner.walk_predictions(G, edge_index, agents, queues, occ)
                    if method == "lookahead_ml" else None)
            planner.replan(G, agents, sensor.sense(moving), sensor.sense(standing), t, fire_active, horizon,
                           switch_gain, walk=walk)
        # flow budgets refill
        for u, v, d in G.edges(data=True):
            k = _ekey(u, v)
            # keep the fractional remainder so the long-run rate equals flow_cap exactly;
            # the cap stops an idle door from "saving up" a burst
            budget[k] = min(budget[k] + d["flow_cap"] * C.DT, d["flow_cap"] * C.DT + 1.0)

        # agents whose pre-movement time is over join the queue at their origin
        still = []
        for a in active:
            if t >= a.start:
                if len(a.path) == 1:          # spawned on an exit
                    a.state, a.t_out = "evacuated", t
                else:
                    a.state, a.queued_at = "queued", t
                    _join(a, a.path[0], t)
            else:
                still.append(a)
        active = still

        # move agents on edges
        arrived = []
        for a in moving:
            u, v = a.path[a.idx], a.path[a.idx + 1]
            d = G.edges[u, v]
            k = _ekey(u, v)
            dens = edge_density(d, occ[k])
            peak_density = max(peak_density, dens)
            if trace_key is not None and k == trace_key:
                edge_trace.append((round(t, 2), round(dens, 3), round(weidmann_speed_factor(dens) * d["free_speed"], 3)))
            if C.EVAC_LIFTS and d["etype"] == "lift":
                a.pos += d["free_speed"] * C.DT          # riding the car: car speed, whoever you are
            else:
                a.pos += (d["free_speed"] * a.speed_mult * weidmann_speed_factor(dens)
                          * smoke_speed_factor(d.get("hazard", 0.0)) * C.DT)
            if a.pos >= d["length_m"]:
                arrived.append(a)
        for a in arrived:
            u, v = a.path[a.idx], a.path[a.idx + 1]
            occ[_ekey(u, v)] -= a.size
            if log_edges and a.edge_log is not None:
                edge_rows.append({**a.edge_log, "target_time_s": round(t + C.DT - a.edge_log["_t_join"], 2)})
                a.edge_log = None
            moving.remove(a)
            a.idx += 1
            a.pos = 0.0
            if a.idx == len(a.path) - 1:
                a.state, a.t_out = "evacuated", t + C.DT
            else:
                a.state, a.queued_at = "queued", t
                _join(a, v, t)

        # admit queued agents onto their next edge (first come, first served)
        for node in sorted(queues):
            q = queues[node]
            if not q:
                continue
            if C.DOOR_PRIORITY_VULNERABLE:          # marshals let vulnerable people through first
                from .planner import vulnerable
                q.sort(key=lambda a: (not vulnerable(a), a.queued_at, a.id))
            else:
                q.sort(key=lambda a: (a.queued_at, a.id))
            waiting = []
            for a in q:
                u, v = a.path[a.idx], a.path[a.idx + 1]
                d = G.edges[u, v]
                k = _ekey(u, v)
                if d.get("hazard", 0.0) >= C.HAZARD_IMPASSABLE:
                    waiting.append(a)        # will be detoured at the next hazard update
                    continue
                if C.EVAC_LIFTS and d["etype"] == "lift":
                    if not _board_lift(G, a, u, d, t, lift_state):
                        waiting.append(a)
                        continue
                    ok = True                    # in the car: no door budget
                else:
                    ok = budget[k] >= 1.0 and occ[k] < d["jam_cap"]
                if ok:
                    budget[k] -= a.size          # a bed uses ~4 person-equivalents (budget may go negative = door busy)
                    occ[k] += a.size
                    a.state = "moving"
                    if a.t_left is None:
                        a.t_left = t
                    moving.append(a)
                    h = d.get("hazard", 0.0)
                    a.max_hazard = max(a.max_hazard, h)
                    if h >= C.HAZARD_EXPOSURE:
                        a.exposed = True
                else:
                    waiting.append(a)
            queues[node] = waiting
            wait_ps[node] += len(waiting) * C.DT                 # person-seconds spent waiting here
            if len(waiting) >= C.JAM_QUEUE:
                jam_s[node] += C.DT                              # seconds with a jam here
            if len(waiting) >= C.JAM_QUEUE // 2:
                jam_small_s[node] += C.DT                        # ... with a smaller queue (sensitivity)
            if len(waiting) > node_peak[node]:
                node_peak[node] = len(waiting)
            if len(waiting) > max_queue:
                max_queue, max_queue_node = len(waiting), node
        t += C.DT

    out = [a for a in agents if a.state == "evacuated"]
    times = sorted(a.t_out for a in out)
    n = len(agents)

    def pct(p):
        if not times:
            return None
        i = min(len(times) - 1, int(math.ceil(p * len(times))) - 1)
        return round(times[i], 1)

    exit_counts = defaultdict(int)
    for a in out:
        exit_counts[a.exit] += 1

    metrics = {
        "method": method,
        "route_changes": sum(a.changes for a in agents),
        "forced_detours": forced,
        "compliance": float(scenario.get("compliance", C.COMPLIANCE)),
        "herding": float(scenario.get("herding", C.HERDING)),
        "seed": seed,
        "n_agents": n,
        "evacuated": len(out),
        "evac_time_last_s": round(times[-1], 1) if times else None,
        "evac_time_mean_s": round(sum(times) / len(times), 1) if times else None,
        "evac_time_p95_s": pct(0.95),
        "needs_assistance": sum(a.state == "needs_assistance" for a in agents),
        "stranded": sum(a.state == "stranded" for a in agents),
        "timed_out": sum(a.state in ("queued", "moving", "waiting") for a in agents),
        "exposed_agents": sum(a.exposed for a in agents) + _in_place_exposed(G, scenario, agents, t),
        "in_place": sum(a.state == "in_place" for a in agents),
        "max_hazard_crossed": round(max((a.max_hazard for a in agents), default=0.0), 3),
        "max_queue": max_queue,
        "max_queue_node": max_queue_node,
        "jam_s": {n: round(v, 1) for n, v in jam_s.items() if v >= C.JAM_MIN_S},
        "jam_small_s": {n: round(v, 1) for n, v in jam_small_s.items() if v >= C.JAM_MIN_S},
        "wait_person_s": round(sum(wait_ps.values()), 1),
        "queue_peaks": dict(sorted(((n, q) for n, q in node_peak.items() if q >= 5), key=lambda x: -x[1])[:8]),
        "peak_density_pm2": round(peak_density, 2),
        "exits_used": dict(sorted(exit_counts.items())),
        "sim_time_s": round(t, 1),
        "compute_s": round(time.perf_counter() - t0, 3),
    }
    if trace_key is not None:
        metrics["edge_trace"] = edge_trace
    if log_edges:
        metrics["edge_rows"] = edge_rows
    if "stop_at_s" in scenario:            # crowd state at stop_at_s, as counts the learned model can use
        metrics["state"] = crowd_state(agents, queues, occ, t)
    if eta_probe:
        out_t = {a.id: a.t_out for a in agents}
        for r in eta_rows:
            r["actual_s"] = None if out_t[r["agent"]] is None else round(out_t[r["agent"]] - r["t"], 1)
        metrics["eta_rows"] = eta_rows
    if track_every:
        snapshot(t)
        metrics["frames"] = frames
    records = [
        {"agent": a.id, "origin": a.origin, "step_free": a.step_free, "profile": a.profile, "state": a.state,
         "exit": a.exit, "t_left": a.t_left, "t_out": a.t_out, "exposed": a.exposed, "route_changes": a.changes,
         "compliant": a.compliant, "via": ("lift" if a.path and any("lift" in n for n in a.path) else
                 "ramp" if a.path and any("ramp" in n for n in a.path) else
                 "stairs" if a.path and any("stairs" in n for n in a.path) else "level")}
        for a in agents
    ]
    return metrics, records


def _hospital_procedures(G, scenario, agents, fire_active):
    """Scenario keys (all optional):
      "in_place": "<regex>" | "rest"            people in matching rooms (or everyone not sent to a refuge)
                                                stay put (defend in place)
      "refuge": [{"where": "<regex>" | "compartment": "<label>", "target": "<node>"}]
                                                people in matching rooms go to a safe place inside the
                                                building (e.g. the next fire compartment) instead of an exit
      "staff": {"<floor>": n}                   beds on that floor can only move when a team of
                                                STAFF_PER_BED staff is free; teams walk back for the next bed
    """
    import re
    moved = set()
    for r in scenario.get("refuge", []):
        rx = re.compile(r["where"]) if r.get("where") else None
        tabs = {}
        for a in agents:
            if a.state != "waiting":
                continue
            if rx is not None and not rx.search(a.origin):
                continue
            if r.get("compartment") and G.nodes[a.origin].get("compartment") != r["compartment"]:
                continue
            moved.add(a.id)
            if a.step_free not in tabs:
                tabs[a.step_free] = route_table(G, "safe_shortest", fire_active, a.step_free, targets=[r["target"]])
            p = tabs[a.step_free].get(a.origin)
            if p:
                a.path, a.exit = p, p[-1]
    stay = scenario.get("in_place")
    if stay:                      # a regex, or "rest" = everyone not sent to a refuge
        rx = None if stay == "rest" else re.compile(stay)
        for a in agents:
            if a.state == "waiting" and a.id not in moved and (rx is None or rx.search(a.origin)):
                a.state = "in_place"
    staff = scenario.get("staff")
    if staff:
        fires = scenario.get("fires", [])
        fire_comp = {G.nodes[f["node"]].get("compartment") for f in fires if f.get("node") in G}

        def length(path):
            return sum(G.edges[u, v]["length_m"] for u, v in zip(path, path[1:]))
        for fl, n_staff in staff.items():
            teams = [0.0] * max(1, int(n_staff) // C.STAFF_PER_BED)
            beds = [a for a in agents if a.profile == "bed" and a.state == "waiting" and a.path
                    and G.nodes[a.origin]["floor"] == int(fl)]
            # beds in the fire compartment first, then in order of readiness
            beds.sort(key=lambda a: (G.nodes[a.origin].get("compartment") not in fire_comp, a.start, a.id))
            for a in beds:
                k = min(range(len(teams)), key=lambda i: teams[i])
                a.start = max(a.start, teams[k])
                L = length(a.path)
                trip = sum(G.edges[u, v]["length_m"] / (G.edges[u, v]["free_speed"] * a.speed_mult)
                           for u, v in zip(a.path, a.path[1:]))
                teams[k] = a.start + trip + L / C.STAFF_WALK_SPEED


def _in_place_exposed(G, scenario, agents, t_end):
    """People defending in place are exposed if smoke/fire reaches their room while they wait."""
    stay = [a for a in agents if a.state == "in_place"]
    if not stay:
        return 0
    from .hazard import node_hazards
    fires = scenario.get("fires", [])
    times = [None] if not is_dynamic(fires) else [k * 10.0 for k in range(int(max(t_end, 600) // 10) + 1)]
    worst = defaultdict(float)
    for tt in times:
        for n, h in node_hazards(G, fires, tt).items():
            worst[n] = max(worst[n], h)
    n_exp = 0
    for a in stay:
        if worst[a.origin] >= C.HAZARD_EXPOSURE:
            a.exposed = True
            n_exp += 1
    return n_exp


def _optimum(G, scenario, seed):
    """Theoretical best last-out time for the same people (same seed) - see evac/optimum.py."""
    from .optimum import lower_bound
    t0 = time.perf_counter()
    ref, _ = run(G.copy(), scenario, "ccrp", seed)        # an achievable time, used as the search ceiling
    rng = random.Random(seed)
    fires = scenario.get("fires", [])
    apply_hazard(G, fires)
    agents = spawn(G, scenario, rng)
    tables = {sf: route_table(G, "safe_shortest", bool(fires), sf) for sf in (False, True)}
    for a in agents:
        a.path = tables[a.step_free].get(a.origin)
    hi = (ref["evac_time_last_s"] or C.MAX_TIME_S) + 10
    lb = lower_bound(G, agents, bool(fires), hi_s=hi)
    metrics = {"method": "optimum", "route_changes": 0, "seed": seed, "n_agents": len(agents),
               "evacuated": sum(a.path is not None for a in agents), "evac_time_last_s": lb,
               "evac_time_mean_s": None, "evac_time_p95_s": None,
               "needs_assistance": sum(a.path is None and a.step_free for a in agents),
               "stranded": sum(a.path is None and not a.step_free for a in agents), "timed_out": 0,
               "exposed_agents": None, "max_hazard_crossed": None, "max_queue": None,
               "max_queue_node": None, "peak_density_pm2": None, "exits_used": {},
               "sim_time_s": None, "compute_s": round(time.perf_counter() - t0, 3)}
    return metrics, []
