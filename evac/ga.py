"""
evac/ga.py
Genetic-algorithm baseline: simulation-optimised exit assignment (Session 6B).

Following the simulation-optimisation approach of Abdelghany et al. (2014, EJOR), a genetic algorithm
searches over which exit each ROOM's occupants should use. Every candidate plan is scored by running
the full simulator; people then follow the shortest safe path to their assigned exit (no re-planning).

  genome      one exit per (origin room, subgroup) - each room's occupants are split into 4 subgroups
              so a crowded room can be divided across exits (4 possible exits per gene)
  fitness     last-person-out time (+ a large penalty per person exposed to hazard)
  operators   tournament selection, uniform crossover, per-gene mutation, elitism

Important: the GA optimises against the SAME seed it is scored on (it knows exactly when every person
starts moving), so it is an optimistic offline benchmark, not something deployable in real time.
It costs ~POP x GENS simulations (~4-6 minutes) per scenario, vs < 5 s for the look-ahead planner.

Run:  python -m evac.ga --scenario floor3_full_200 --seed 0
"""

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from evac.building import exits, load_campus     # noqa: E402
from evac.simulator import run, spawn            # noqa: E402

EXPOSURE_PENALTY_S = 300.0


def fitness(G0, scenario, plan, seed):
    m, _ = run(G0.copy(), {**scenario, "exit_plan": plan}, "safe_shortest", seed)
    last = m["evac_time_last_s"] or 1800.0
    return last + EXPOSURE_PENALTY_S * (m["exposed_agents"] or 0), m


def optimise(scenario, seed=0, pop=16, gens=14, p_mut=0.08, G0=None, verbose=True):
    G0 = G0 or load_campus()
    rng = random.Random(1000 + seed)
    origins = sorted({f"{a.origin}#{a.id % 4}" for a in spawn(G0, {**scenario, "_seed": seed}, random.Random(seed))})
    ex = exits(G0)
    # seed the population with the nearest-exit plan so the GA never does worse than its start
    base, _ = run(G0.copy(), scenario, "safe_shortest", seed)
    from evac.routing import route_table
    from evac.hazard import apply_hazard
    G = apply_hazard(G0.copy(), scenario.get("fires", []))
    tab = route_table(G, "safe_shortest", bool(scenario.get("fires")), False)
    nearest = [tab[o.split("#")[0]][-1] if o.split("#")[0] in tab else ex[0] for o in origins]
    popn = [nearest] + [[rng.choice(ex) for _ in origins] for _ in range(pop - 1)]
    scored = []
    t0 = time.time()
    history = []
    for g in range(gens):
        scored = sorted(((fitness(G0, scenario, dict(zip(origins, ind)), seed)[0], ind) for ind in popn),
                        key=lambda x: x[0])
        history.append(scored[0][0])
        if verbose:
            print(f"  gen {g:2d} best fitness {scored[0][0]:.1f}  ({time.time() - t0:.0f}s)")
        elite = [ind for _, ind in scored[:2]]
        nxt = [list(e) for e in elite]
        while len(nxt) < pop:
            a = min(rng.sample(scored, 3))[1]
            b = min(rng.sample(scored, 3))[1]
            child = [x if rng.random() < 0.5 else y for x, y in zip(a, b)]
            child = [rng.choice(ex) if rng.random() < p_mut else c for c in child]
            nxt.append(child)
        popn = nxt
    best_plan = dict(zip(origins, scored[0][1]))
    _, m = fitness(G0, scenario, best_plan, seed)
    m["method"] = "ga"
    m["ga_evaluations"] = pop * gens + 1
    m["ga_runtime_s"] = round(time.time() - t0, 1)
    m["ga_history"] = history
    return m, best_plan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="floor3_full_200")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sc = json.load(open(os.path.join(HERE, "scenarios.json"), encoding="utf-8"))[a.scenario]
    m, plan = optimise(sc, a.seed)
    print({k: m[k] for k in ("evac_time_last_s", "evac_time_mean_s", "exposed_agents", "ga_runtime_s")})


if __name__ == "__main__":
    main()
