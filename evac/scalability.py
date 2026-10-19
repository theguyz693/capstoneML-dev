"""
evac/scalability.py
Phase 2, item 4: is the planner fast enough to run LIVE in bigger buildings with bigger crowds?

The look-ahead planner re-plans every REPLAN_S = 5 s, so one re-plan must finish well inside 5 s on ordinary
hardware. We time, on one CPU core:
  replan      one look-ahead re-plan with EVERYONE standing (the worst case: nobody has left yet)
  ccrp        the one-shot CCRP plan for everyone (Lu, George & Shekhar 2005)
  shortest    the shortest-safe-path table for the whole building (what simple signs use)
  ml_eta      the learned time-to-exit table for every node (ML v3, dashboard/signs)
for (a) the real campus and hospital with growing crowds and (b) generated tower blocks of growing size
(grid floors, rooms on every junction, 4 stairwells, 4 exits) - up to ~8,000 nodes and 6,000 people.

Run:  python -m evac.scalability   -> results/scalability.csv, docs/scalability.md, docs/figures/fig33_scalability.png
"""

import csv
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "scalability.csv")
DOC = os.path.join(ROOT, "docs", "scalability.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig33_scalability.png")


def tower(floors, width, spacing_m=10.0):
    """A generated building: each floor is a width x width grid of corridor junctions with one room on each
    junction, four stairwells at the corners, four exits on the ground floor."""
    from evac import config as C
    from evac.building import build_graph
    U = 1.0 / C.METERS_PER_UNIT
    d = spacing_m * U
    nodes, edges = {}, []
    corners = [(0, 0), (0, width - 1), (width - 1, 0), (width - 1, width - 1)]
    for f in range(floors):
        for x in range(width):
            for y in range(width):
                j = f"f{f}_j{x}_{y}"
                nodes[j] = (x * d, y * d, f)
                r = f"f{f}_room {x}-{y}"
                nodes[r] = (x * d + 0.3 * d, y * d + 0.3 * d, f)
                edges.append((j, r))
                if x:
                    edges.append((j, f"f{f}_j{x - 1}_{y}"))
                if y:
                    edges.append((j, f"f{f}_j{x}_{y - 1}"))
        for k, (x, y) in enumerate(corners):
            s = f"f{f}_stairs {k}"
            nodes[s] = (x * d - 0.2 * d, y * d - 0.2 * d, f)
            edges.append((s, f"f{f}_j{x}_{y}"))
            if f:
                edges.append((s, f"f{f - 1}_stairs {k}"))
    for k, (x, y) in enumerate(corners):
        e = f"exit {k}"
        nodes[e] = (x * d - 0.6 * d, y * d - 0.6 * d, 0)
        edges.append((e, f"f0_j{x}_{y}"))
    return build_graph(nodes, edges)


def _timed(fn, reps=3):
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), out


def measure(G, n_people, label, seed=0, reps=3):
    import math
    from evac import config as C
    from evac import planner
    from evac.learned_planner import EdgeIndex, eta_table
    from evac.routing import route_table
    from evac.simulator import spawn
    rng = random.Random(seed)
    agents = spawn(G, {"n_agents": n_people, "spawn": "building", "_seed": seed}, rng)
    tab = route_table(G, "safe_shortest", False, False)
    agents = [a for a in agents if a.origin in tab]
    for a in agents:
        a.path, a.exit, a.idx, a.start = tab[a.origin], tab[a.origin][-1], 0, 0.0
    t_short, _ = _timed(lambda: route_table(G, "safe_shortest", False, False), reps)
    t_ccrp, _ = _timed(lambda: planner.plan_all(G, agents, False), reps)
    saved = [(a.path, a.exit, a.changes) for a in agents]
    t_replan, _ = _timed(lambda: planner.replan(G, agents, [], agents, 0.0, False, C.HORIZON_S, C.SWITCH_GAIN), 1)
    t_ref = None
    if G.number_of_nodes() * len(agents) <= 2_000_000:   # the plain search (before the A* speed-up), for comparison
        for a, (p, e, ch) in zip(agents, saved):
            a.path, a.exit, a.changes = p, e, ch
        fast = planner.earliest_route
        planner.earliest_route = planner._earliest_route_ref
        try:
            t_ref, _ = _timed(lambda: planner.replan(G, agents, [], agents, 0.0, False, C.HORIZON_S, C.SWITCH_GAIN), 1)
        finally:
            planner.earliest_route = fast
    old = C.DELAY_MODEL
    C.DELAY_MODEL = "v3"
    try:
        idx = EdgeIndex(G)
        t_eta, _ = _timed(lambda: eta_table(G, idx, None, False), reps)
    finally:
        C.DELAY_MODEL = old
    return {"building": label, "nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "people": len(agents),
            "shortest_s": t_short, "ccrp_s": t_ccrp, "replan_s": t_replan, "replan_plain_search_s": t_ref,
            "ml_eta_s": t_eta}


def collect():
    import pandas as pd
    if os.path.exists(OUT):
        return pd.read_csv(OUT)
    from evac.building import load_building
    rows = []
    for b in ("campus", "hospital"):
        G = load_building(b)
        for n in (100, 250, 500, 1000, 2000):
            rows.append(measure(G, n, b))
            print(rows[-1], flush=True)
    for floors, width in ((3, 6), (5, 8), (8, 10), (12, 12), (16, 14), (20, 14)):
        G = tower(floors, width)
        rooms = floors * width * width
        for per_room in ((1, 3) if rooms <= 2000 else (1, 2)):
            rows.append(measure(G, rooms * per_room, f"tower {floors} floors x {width}x{width}"))
            print(rows[-1], flush=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return pd.read_csv(OUT)


def report(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from evac import config as C
    from evac.final_report import BLUE, ORANGE, GREY, INK, MUTED, _style
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    real = D[D["building"].isin(["campus", "hospital"])]
    for b, mk in (("campus", "o"), ("hospital", "s")):
        r = real[real["building"] == b]
        ax[0].plot(r["people"], r["replan_s"], mk + "-", color=BLUE, label=f"{b}: look-ahead re-plan")
        ax[0].plot(r["people"], r["replan_plain_search_s"], mk + ":", color=GREY, label=f"{b}: re-plan before A*")
    tw = D[D["building"].str.startswith("tower")]
    ax[1].scatter(tw["nodes"] * tw["people"], tw["replan_s"], color=BLUE, label="look-ahead re-plan", zorder=3)
    tb = tw.dropna(subset=["replan_plain_search_s"])
    ax[1].scatter(tb["nodes"] * tb["people"], tb["replan_plain_search_s"], facecolors="none", edgecolors=BLUE,
                  label="... before the A* speed-up")
    ax[1].scatter(tw["nodes"] * tw["people"], tw["ml_eta_s"], color=ORANGE, marker="^", label="ML time-to-exit table")
    ax[1].scatter(tw["nodes"] * tw["people"], tw["shortest_s"], color=GREY, marker="s", label="shortest-path table")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    for a in ax:
        a.axhline(C.REPLAN_S, color=ORANGE, ls=":", lw=1.5, label="re-plan interval (5 s)")
        a.legend(fontsize=7, frameon=False)
        _style(a, "both")
    ax[0].set(xlabel="People in the building", ylabel="Compute time (s, one core)")
    ax[0].set_title("Real buildings, growing crowd", loc="left", fontsize=11, fontweight="bold")
    ax[1].set(xlabel="Building size x crowd (nodes x people)", ylabel="Compute time (s)")
    ax[1].set_title("Generated towers up to 20 floors", loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)
    plt.close(fig)

    # simple power-law fit of re-plan time in nodes and people (towers)
    X = np.c_[np.ones(len(tw)), np.log(tw["nodes"]), np.log(tw["people"])]
    coef = np.linalg.lstsq(X, np.log(tw["replan_s"]), rcond=None)[0]
    within = D[D["replan_s"] <= C.REPLAN_S]
    big_ok = within.sort_values("people").iloc[-1] if len(within) else None
    over = D[D["replan_s"] > C.REPLAN_S]
    cam = real[(real["building"] == "campus")]
    both = D.dropna(subset=["replan_plain_search_s"])
    sp = both["replan_plain_search_s"] / both["replan_s"]
    L = ["# Scalability: fast enough to run live? (Phase 2)", "",
         "One look-ahead re-plan has to finish inside the 5 s re-plan interval. Times are on ONE core of the "
         "machine this was run on (median of 3, one run for the full re-plan), with everyone still inside and "
         "standing (the worst case). Generated by `python -m evac.scalability`.", "", "## Findings", "",
         f"1. **Campus (452 nodes)**: a full re-plan takes {cam[cam.people == 250]['replan_s'].iloc[0]:.2f} s for 250 "
         f"people and {cam[cam.people == 2000]['replan_s'].iloc[0]:.2f} s for 2,000 people; one-shot CCRP "
         f"{cam[cam.people == 2000]['ccrp_s'].iloc[0]:.2f} s for 2,000.",
         f"2. **Largest case inside the 5 s budget**: {big_ok['building']} ({int(big_ok['nodes']):,} nodes, "
         f"{int(big_ok['people']):,} people) re-planned in {big_ok['replan_s']:.2f} s." if big_ok is not None else
         "2. No case fit inside 5 s.",
         (f"3. **Over budget**: " + "; ".join(f"{r['building']} with {int(r['people']):,} people ({r['replan_s']:.1f} s)"
                                               for _, r in over.iterrows()) +
          ". Remedies, in order of effort: re-plan only the floors that changed, re-plan less often when the fire is "
          "not growing (the data-freshness study in docs/final_results.md found 80 s enough there, but ~5 s with a "
          "growing fire), or plan floor by floor as two-stage methods do (Wong & Lee 2023)."
          if len(over) else "3. **Every case fits inside the 5 s budget.**"),
         f"4. **Growth**: re-plan time grows roughly as nodes^{coef[1]:.2f} x people^{coef[2]:.2f} (fit on the "
         "generated towers) - each person is one time-dependent shortest-path search over the building.",
         f"5. **Speed-up made in this step**: the planner's route search is now an A* search with a free-flow lower "
         "bound plus cached edge costs (`planner._Fast`). It gives exactly the same evacuation results as before "
         "(checked: identical last-out times and exposure on 12 paired runs and in `tests/test_evac_phase2.py`) and "
         f"is {sp.min():.1f}-{sp.max():.1f}x faster (median {sp.median():.1f}x) where both were timed; whole simulations "
         "run 3-5x faster.",
         "6. The shortest-path table and the ML time-to-exit table (what signs and the dashboard route use) take "
         f"at most {D['shortest_s'].max():.2f} s and {D['ml_eta_s'].max():.2f} s even in the largest tower.", "",
         "![Scalability](figures/fig33_scalability.png)", "",
         "| Building | Nodes | People | Look-ahead re-plan (s) | ... before the A* speed-up (s) | One-shot CCRP (s) | Shortest-path table (s) | ML time-to-exit table (s) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in D.iterrows():
        old = "-" if r["replan_plain_search_s"] != r["replan_plain_search_s"] else f"{r['replan_plain_search_s']:.2f}"
        L.append(f"| {r['building']} | {int(r['nodes']):,} | {int(r['people']):,} | {r['replan_s']:.2f} | {old} | "
                 f"{r['ccrp_s']:.2f} | {r['shortest_s']:.3f} | {r['ml_eta_s']:.3f} |")
    L += ["", "Pure Python on one core; the planner is not parallelised or compiled, so these are upper bounds.", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[5:12]))


def main():
    report(collect())


if __name__ == "__main__":
    main()
