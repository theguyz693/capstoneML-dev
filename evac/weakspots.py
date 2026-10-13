"""
evac/weakspots.py
Which single failure hurts the building most?  (N-1 contingency / criticality analysis)

For each candidate element - every stair flight, every exit door, and the most-used corridor
segments - we block it, re-run the same evacuation (same people, same seeds), and measure how
much later the last person gets out. Ranking those deltas gives the building's critical points:
where a blocked stair or a locked door does the most damage, i.e. where to put extra signage,
wider doors or a fire warden.

Run:  python -m evac.weakspots                       -> docs/weakspots.md + docs/figures/fig5_weakspots.png
      python -m evac.weakspots --scenario floor3_full_200 --method ccrp --seeds 3
"""

import argparse
import json
import os
import statistics
import sys

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from evac.building import load_campus, exits       # noqa: E402
from evac.simulator import run                     # noqa: E402

DOC = os.path.join(ROOT, "docs", "weakspots.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig5_weakspots.png")


def _label(G, edges, kind):
    if kind == "stair_shaft":
        return f"Stairwell {edges[0][0].split()[-1]} (all floors)"
    if kind == "ramp":
        return "Ramp (all floors)"
    u, v = edges[0]
    if kind == "exit_door":
        return f"Exit door: {u if G.nodes[u]['kind'] == 'exit' else v}"
    return f"Corridor {u} - {v}"


def candidates(G, n_corridors=12):
    """Elements that can fail as a unit: whole stair shafts (a smoke-logged or locked stairwell),
    exit doors, and the corridor segments most used on routes to exits."""
    shafts = {}
    for u, v, d in G.edges(data=True):
        if d["etype"] == "stair":
            shafts.setdefault(u.split()[-1], []).append((u, v))
    out = [("stair_shaft", e) for _, e in sorted(shafts.items())]
    ramp = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "ramp"]
    if ramp:                                   # hospital: the floor-to-floor ramp fails as a unit
        out.append(("ramp", ramp))
    out += [("exit_door", [(u, v)]) for u, v, d in G.edges(data=True) if d["etype"] == "exit_door"]
    ex = exits(G)
    H = nx.Graph((u, v, {"w": d["length_m"]}) for u, v, d in G.edges(data=True)
                 if d["etype"] in ("corridor", "stair", "exit_door"))
    bc = nx.edge_betweenness_centrality_subset(H, sources=[n for n in H if G.nodes[n]["kind"] == "junction"],
                                               targets=[e for e in ex if e in H], weight="w", normalized=False)
    corr = sorted(((e, s) for e, s in bc.items() if G.edges[e]["etype"] == "corridor"), key=lambda x: -x[1])
    out += [("corridor", [e]) for e, _ in corr[:n_corridors]]
    return out


def analyse(scenario, method="ccrp", seeds=(0, 1, 2), G0=None):
    G0 = G0 if G0 is not None else load_campus()
    base = [run(G0.copy(), scenario, method, s)[0] for s in seeds]
    b_last = statistics.mean(m["evac_time_last_s"] for m in base)
    b_mean = statistics.mean(m["evac_time_mean_s"] for m in base)
    rows = []
    for kind, edges in candidates(G0):
        G = G0.copy()
        G.remove_edges_from(edges)
        ms = [run(G.copy(), scenario, method, s)[0] for s in seeds]
        trapped = statistics.mean(m["stranded"] + m["timed_out"] + m["needs_assistance"] for m in ms) - \
            statistics.mean(m["needs_assistance"] for m in base)
        last = [m["evac_time_last_s"] for m in ms if m["evac_time_last_s"] is not None]
        mean = [m["evac_time_mean_s"] for m in ms if m["evac_time_mean_s"] is not None]
        rows.append({
            "element": _label(G0, edges, kind), "edges": [list(e) for e in edges], "type": kind,
            "last_out_delta_s": round(statistics.mean(last) - b_last, 1) if last else None,
            "mean_delta_s": round(statistics.mean(mean) - b_mean, 1) if mean else None,
            "people_cut_off": round(max(0.0, trapped), 1),
        })
    rows.sort(key=lambda r: (-r["people_cut_off"], -(r["last_out_delta_s"] or 0)))
    return {"baseline_last_out_s": round(b_last, 1), "baseline_mean_s": round(b_mean, 1), "rows": rows}


def report(res, scen_name, method):
    rows = res["rows"]
    lines = [f"# Weak-spot analysis: `{scen_name}`, method `{method}`", "",
             f"Baseline: last person out at {res['baseline_last_out_s']} s (mean {res['baseline_mean_s']} s). "
             "Each row blocks ONE element and re-runs the same evacuation.", "",
             "| Rank | Blocked element | Last out (+s) | Average person (+s) | People cut off |",
             "| --- | --- | --- | --- | --- |"]
    for i, r in enumerate(rows[:20], 1):
        fmt = lambda x: "-" if x is None else f"{x:+.1f}"
        lines.append(f"| {i} | {r['element']} | {fmt(r['last_out_delta_s'])} | {fmt(r['mean_delta_s'])} | {r['people_cut_off']:.0f} |")
    lines += ["", "People cut off = no remaining route to any exit, newly needing assistance, or still inside at the 30-minute cap.",
              "", "![Weak spots](figures/fig5_weakspots.png)", ""]
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(ROOT, "docs", "weakspots.json"), "w", encoding="utf-8") as f:
        json.dump({"scenario": scen_name, "method": method, **res}, f, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = [r for r in rows if r["last_out_delta_s"] is not None][:12][::-1]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        cols = {"stair_shaft": "#2a78d6", "exit_door": "#eb6834", "corridor": "#a3a29a"}
        ax.barh(range(len(top)), [r["last_out_delta_s"] for r in top], height=0.6,
                color=[cols[r["type"]] for r in top])
        for i, r in enumerate(top):
            ax.text(r["last_out_delta_s"] + 1, i, f"+{r['last_out_delta_s']:.0f} s", va="center", fontsize=8.5)
        ax.set_yticks(range(len(top)), [r["element"] for r in top], fontsize=8.5)
        ax.set_xlabel("Extra time until the last person is out (s)")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=k.replace("_", " ")) for k, c in cols.items()],
                  frameon=False, loc="lower right", fontsize=8.5)
        ax.set_title("Single failures that slow evacuation the most", loc="left", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        fig.savefig(FIG, dpi=160)
    except ImportError:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="baseline_150_no_fire")
    ap.add_argument("--method", default="ccrp")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    scen = json.load(open(os.path.join(HERE, "scenarios.json"), encoding="utf-8"))
    res = analyse(scen[a.scenario], a.method, tuple(range(a.seeds)))
    report(res, a.scenario, a.method)
    for r in res["rows"][:10]:
        print(r)
    print(f"\nWrote {DOC}")


if __name__ == "__main__":
    main()
