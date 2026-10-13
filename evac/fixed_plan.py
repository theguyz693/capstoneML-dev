"""
evac/fixed_plan.py
A PRINTABLE evacuation plan made without any live data (Session 7).

Without sensors the building still knows roughly who will be where: the timetable, the bed list,
OPD appointments. This module plans ONCE from that estimate, with the same capacity-booking planner
(CCRP), and turns the result into a plan per room that can be printed on the wall:

    room 301:  60% -> stairs 3 -> East exit,  40% -> stairs 1 -> North exit

  - the estimate is a fresh sample of the scenario's occupancy (different people, start times),
    optionally scaled to model a timetable that is wrong (count_error = -0.3 means 30% fewer
    people than expected actually came ... +0.5 means 50% more);
  - a printed plan does not know where a fire will be, so it is made WITHOUT fire; in a fire,
    anyone whose printed route crosses smoke falls back to the hazard-aware sign route;
  - "split" mode prints the shares per room (a fire warden splits the room); "single" mode prints
    only the most common route per room (simplest signage).

Use:  sc = {**scenario, "route_plan": make_plan(G, scenario), "route_plan_mode": "split"}
      run(G, sc, "fixed_plan", seed)
      write_markdown(G, plan, "docs/plan_floor3.md")
"""

import random
from collections import Counter, defaultdict

from . import config as C
from . import planner
from .routing import route_table


def _scaled(scenario, factor, with_fire=False):
    sc = {k: v for k, v in scenario.items() if k not in ("fires", "route_plan", "compliance", "herding")}
    sc["fires"] = list(scenario.get("fires", [])) if with_fire else []
    if sc.get("spawn") == "groups":
        sc["groups"] = [{**g, "n": max(1, round(g["n"] * factor))} for g in sc["groups"]]
    else:
        sc["n_agents"] = max(1, round(int(sc.get("n_agents", 100)) * factor))
    return sc


def make_plan(G, scenario, samples=5, count_error=0.0, seed_base=1000, with_fire=False):
    """{ 'origin|step_free': [[path, share], ...] } from `samples` timetable-like estimates.
    with_fire=True: a pre-computed plan for the fire's alarm zone (the fire panel knows WHERE the alarm
    is, but there is still no live data about people) - one such plan per alarm zone."""
    from .hazard import apply_hazard
    from .simulator import spawn
    fires = scenario.get("fires", []) if with_fire else []
    if fires:
        G = apply_hazard(G.copy(), [{k: v for k, v in f.items() if k not in ("spread_m_per_s",)} for f in fires])
    routes = defaultdict(Counter)
    tables = {sf: route_table(G, "safe_shortest", bool(fires), sf) for sf in (False, True)}
    for k in range(samples):
        sc = _scaled(scenario, 1.0 + count_error, with_fire)
        sc["_seed"] = seed_base + k
        agents = spawn(G, sc, random.Random(seed_base + k))
        for a in agents:
            a.path = tables[a.step_free].get(a.origin)
        plannable = [a for a in agents if a.path is not None]
        plans = planner.plan_all(G, plannable, bool(fires))
        for a in plannable:
            p = plans.get(a.id, a.path)
            routes[f"{a.origin}|{int(a.step_free)}"][tuple(p)] += 1
    out = {}
    for key, cnt in routes.items():
        tot = sum(cnt.values())
        out[key] = [[list(p), round(c / tot, 4)] for p, c in cnt.most_common()]
    return out


def _safe(G, path, fire_active, step_free):
    from .routing import _allowed
    for u, v in zip(path, path[1:]):
        d = G.edges[u, v]
        if not _allowed(d, fire_active, step_free) or d.get("hazard", 0.0) >= C.HAZARD_EXPOSURE:
            return False
    return True


def assign(G, agents, plan, mode, fire_active):
    """Give each (compliant) person the printed route of their room. Deterministic: people of one
    room are split over the room's routes in proportion to the printed shares."""
    by_room = defaultdict(list)
    for a in agents:
        if a.path is not None and a.compliant:
            by_room[(a.origin, a.step_free)].append(a)
    used = 0
    for (origin, sf), people in by_room.items():
        opts = plan.get(f"{origin}|{int(sf)}")
        if not opts:
            continue
        if mode == "single":
            opts = [[opts[0][0], 1.0]]
        people.sort(key=lambda a: a.id)
        cum, bounds = 0.0, []
        for p, s in opts:
            cum += s
            bounds.append((cum, p))
        for i, a in enumerate(people):
            f = (i + 0.5) / len(people) * cum
            path = next(p for b, p in bounds if f <= b)
            if _safe(G, path, fire_active, a.step_free):
                a.path, a.exit = list(path), path[-1]
                used += 1
    return used


def describe(G, path):
    stairs = [n for n in path if G.nodes[n]["kind"] == "stair"]
    ramp = any(G.nodes[n]["kind"] == "ramp" for n in path)
    lift = any(G.nodes[n]["kind"] == "lift" for n in path)
    via = (f"stairs {stairs[0].split()[-1]}" if stairs else "ramp" if ramp else "lift" if lift else "corridor")
    return f"{via} -> {path[-1].replace('g_', '')}"


def write_markdown(G, plan, path, title):
    rows = []
    for key in sorted(plan, key=lambda k: (G.nodes[k.split('|')[0]]["floor"], k)):
        origin, sf = key.split("|")
        merged = Counter()
        for p, s in plan[key]:
            merged[describe(G, p)] += s
        txt = ", ".join(f"{s:.0%} {r}" for r, s in merged.most_common())
        rows.append(f"| {G.nodes[origin]['floor']} | {origin} | {'step-free' if sf == '1' else 'everyone'} | {txt} |")
    lines = [f"# {title}", "", "Printed evacuation plan made once from the expected occupancy (no live data). "
             "Shares = how a fire warden should split the room.", "",
             "| Floor | Room | Who | Route (share of the room) |", "| --- | --- | --- | --- |"] + rows + [""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
