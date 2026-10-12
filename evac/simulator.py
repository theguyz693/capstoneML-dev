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

Same scenario + same seed -> identical results.
"""

import math
import random
import time
from collections import defaultdict

from . import config as C
from .building import room_nodes
from .hazard import apply_hazard
from .routing import route_table


def weidmann_speed_factor(density):
    """Fraction of free speed at a given density (persons/m^2)."""
    if density <= 0.05:
        return 1.0
    if density >= C.JAM_DENSITY:
        return C.MIN_SPEED_FRACTION
    f = 1.0 - math.exp(-1.913 * (1.0 / density - 1.0 / C.JAM_DENSITY))
    return max(C.MIN_SPEED_FRACTION, min(1.0, f))


def _ekey(u, v):
    return (u, v) if u <= v else (v, u)


class Agent:
    __slots__ = ("id", "origin", "path", "idx", "pos", "state", "start", "speed_mult",
                 "step_free", "queued_at", "t_out", "exposed", "max_hazard", "exit")

    def __init__(self, aid, origin, start, speed_mult, step_free):
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


def spawn(G, scenario, rng):
    """Create agents. scenario keys: n_agents, spawn ('building'|'single'|'floor'), origin, floor, step_free_frac"""
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
    agents = []
    for i, o in enumerate(origins):
        agents.append(Agent(
            aid=i, origin=o,
            start=rng.uniform(0.0, C.PRE_MOVEMENT_MAX_S),
            speed_mult=1.0 + rng.uniform(-C.AGENT_SPEED_SPREAD, C.AGENT_SPEED_SPREAD),
            step_free=(rng.random() < sf),
        ))
    return agents


def run(G, scenario, method, seed):
    """Run one evacuation. Returns (metrics dict, per-agent records)."""
    t0 = time.perf_counter()
    rng = random.Random(seed)
    fires = scenario.get("fires", [])
    fire_active = bool(fires)
    apply_hazard(G, fires)

    agents = spawn(G, scenario, rng)

    # routes: one multi-source Dijkstra per (method, step_free) covers every origin
    tables = {
        False: route_table(G, method, fire_active, step_free=False),
        True: route_table(G, method, fire_active, step_free=True),
    }
    for a in agents:
        p = tables[a.step_free].get(a.origin)
        if p is None:
            a.state = "needs_assistance" if a.step_free else "stranded"
        else:
            a.path = p
            a.exit = p[-1]

    budget = defaultdict(float)
    occ = defaultdict(int)
    queues = defaultdict(list)      # node -> agents waiting to leave it
    moving = []
    max_queue, max_queue_node = 0, None
    peak_density = 0.0

    t = 0.0
    active = [a for a in agents if a.state == "waiting"]
    while t < C.MAX_TIME_S and (active or moving or any(queues.values())):
        # flow budgets refill
        for u, v, d in G.edges(data=True):
            k = _ekey(u, v)
            budget[k] = min(budget[k] + d["flow_cap"] * C.DT, max(1.0, d["flow_cap"] * C.DT))

        # agents whose pre-movement time is over join the queue at their origin
        still = []
        for a in active:
            if t >= a.start:
                if len(a.path) == 1:          # spawned on an exit
                    a.state, a.t_out = "evacuated", t
                else:
                    a.state, a.queued_at = "queued", t
                    queues[a.path[0]].append(a)
            else:
                still.append(a)
        active = still

        # move agents on edges
        arrived = []
        for a in moving:
            u, v = a.path[a.idx], a.path[a.idx + 1]
            d = G.edges[u, v]
            k = _ekey(u, v)
            dens = occ[k] / (d["length_m"] * d["width_m"])
            peak_density = max(peak_density, dens)
            a.pos += d["free_speed"] * a.speed_mult * weidmann_speed_factor(dens) * C.DT
            if a.pos >= d["length_m"]:
                arrived.append(a)
        for a in arrived:
            u, v = a.path[a.idx], a.path[a.idx + 1]
            occ[_ekey(u, v)] -= 1
            moving.remove(a)
            a.idx += 1
            a.pos = 0.0
            if a.idx == len(a.path) - 1:
                a.state, a.t_out = "evacuated", t + C.DT
            else:
                a.state, a.queued_at = "queued", t
                queues[v].append(a)

        # admit queued agents onto their next edge (first come, first served)
        for node in sorted(queues):
            q = queues[node]
            if not q:
                continue
            q.sort(key=lambda a: (a.queued_at, a.id))
            waiting = []
            for a in q:
                u, v = a.path[a.idx], a.path[a.idx + 1]
                d = G.edges[u, v]
                k = _ekey(u, v)
                if budget[k] >= 1.0 and occ[k] < d["jam_cap"]:
                    budget[k] -= 1.0
                    occ[k] += 1
                    a.state = "moving"
                    moving.append(a)
                    h = d.get("hazard", 0.0)
                    a.max_hazard = max(a.max_hazard, h)
                    if h >= C.HAZARD_EXPOSURE:
                        a.exposed = True
                else:
                    waiting.append(a)
            queues[node] = waiting
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
        "seed": seed,
        "n_agents": n,
        "evacuated": len(out),
        "evac_time_last_s": round(times[-1], 1) if times else None,
        "evac_time_mean_s": round(sum(times) / len(times), 1) if times else None,
        "evac_time_p95_s": pct(0.95),
        "needs_assistance": sum(a.state == "needs_assistance" for a in agents),
        "stranded": sum(a.state == "stranded" for a in agents),
        "timed_out": sum(a.state in ("queued", "moving", "waiting") for a in agents),
        "exposed_agents": sum(a.exposed for a in agents),
        "max_hazard_crossed": round(max((a.max_hazard for a in agents), default=0.0), 3),
        "max_queue": max_queue,
        "max_queue_node": max_queue_node,
        "peak_density_pm2": round(peak_density, 2),
        "exits_used": dict(sorted(exit_counts.items())),
        "sim_time_s": round(t, 1),
        "compute_s": round(time.perf_counter() - t0, 3),
    }
    records = [
        {"agent": a.id, "origin": a.origin, "step_free": a.step_free, "state": a.state,
         "exit": a.exit, "t_out": a.t_out, "exposed": a.exposed}
        for a in agents
    ]
    return metrics, records
