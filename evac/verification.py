"""
evac/verification.py
Simulator sanity tests, in the spirit of ISO 20414 / RiMEA verification tests.

These check that the simulator does what its assumptions say, on tiny test buildings:
  A. Free walking   - one person, 50 m corridor: time ~= distance / free speed
  B. Door flow      - 100 people through one door: flow through the door ~= 1.3 persons/s
                      per metre of width, and it grows linearly with door width.
                      (Downstream, the Weidmann curve caps corridor flow at ~1.14 p/s/m (at 1.75 persons/m^2),
                      so a corridor of the same width as the door can also be the bottleneck.)
  C. Stairs         - the same crowd takes longer down 2 flights than along a flat
                      corridor of equal length (slower speed, lower stair flow)
  D. Speed-density  - measured walking speeds on a crowded corridor follow the
                      Weidmann fundamental diagram used by the model

Verification shows the code implements the model correctly. It is NOT validation
against real people - that needs real-world data (see report limitations).

Run:  python -m evac.verification          -> docs/verification/ (PNG + summary.md)
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from evac import config as C                              # noqa: E402
from evac.building import build_graph                     # noqa: E402
from evac.simulator import run, weidmann_speed_factor     # noqa: E402

U = 1.0 / C.METERS_PER_UNIT   # graph units per metre
OUT_DIR = os.path.join(os.path.dirname(HERE), "docs", "verification")
BASE = {"spawn": "single", "origin": "room", "pre_movement_max_s": 0.0,
        "speed_spread": 0.0, "fires": []}


def test_free_walk():
    nodes = {"room": (0, 0, 0), "j1": (1 * U, 0, 0), "j2": (51 * U, 0, 0), "test exit": (52 * U, 0, 0)}
    G = build_graph(nodes, [("room", "j1"), ("j1", "j2"), ("j2", "test exit")])
    m, _ = run(G, {**BASE, "n_agents": 1}, "nearest", seed=0)
    expected = sum(d["length_m"] / d["free_speed"] for _, _, d in G.edges(data=True))
    measured = m["evac_time_last_s"]
    return {"test": "A. Free walking (52 m)", "expected": round(expected, 1), "measured": measured,
            "unit": "s", "pass": abs(measured - expected) <= 3 * C.DT + 0.5}


def door_flow(width, n=100):
    G = build_graph({"room": (0, 0, 0), "test exit": (2 * U, 0, 0)}, [("room", "test exit")])
    d = G.edges["room", "test exit"]
    d["width_m"], d["flow_cap"], d["jam_cap"] = width, C.SPECIFIC_FLOW["exit_door"] * width, 10_000
    _, recs = run(G, {**BASE, "n_agents": n}, "nearest", seed=0)
    # measured AT the door: the times people pass through it
    ts = sorted(r["t_left"] for r in recs if r["t_left"] is not None)
    return (len(ts) - 1) / (ts[-1] - ts[0])


def test_door_flow():
    widths = [0.8, 1.0, 1.2, 1.6, 2.0, 2.4]
    rows = [(w, round(door_flow(w), 2), round(C.SPECIFIC_FLOW["exit_door"] * w, 2)) for w in widths]
    ok = all(abs(m - e) / e < 0.10 for _, m, e in rows)
    one = [r for r in rows if r[0] == 1.0][0]
    return {"test": "B. Door flow, 1.0 m door", "expected": one[2], "measured": one[1],
            "unit": "persons/s", "pass": ok, "series": rows}


def _three_floor(stairs: bool):
    if stairs:
        nodes = {"room": (0, 0, 2), "f2_j": (1 * U, 0, 2), "f2_stairs a": (2 * U, 0, 2),
                 "f1_stairs a": (2 * U, 0, 1), "stairs a": (2 * U, 0, 0), "j0": (3 * U, 0, 0),
                 "test exit": (4 * U, 0, 0)}
        edges = [("room", "f2_j"), ("f2_j", "f2_stairs a"), ("f2_stairs a", "f1_stairs a"),
                 ("f1_stairs a", "stairs a"), ("stairs a", "j0"), ("j0", "test exit")]
    else:  # flat corridor of the same walking length as two flights
        L = 2 * C.STAIR_FLIGHT_LENGTH_M
        nodes = {"room": (0, 0, 0), "j_a": (1 * U, 0, 0), "j_b": (2 * U, 0, 0),
                 "j_c": ((2 + L) * U, 0, 0), "j0": ((3 + L) * U, 0, 0), "test exit": ((4 + L) * U, 0, 0)}
        edges = [("room", "j_a"), ("j_a", "j_b"), ("j_b", "j_c"), ("j_c", "j0"), ("j0", "test exit")]
    return build_graph(nodes, edges)


def test_stairs(n=60):
    res = {}
    for kind in ("corridor", "stairs"):
        G = _three_floor(kind == "stairs")
        m, _ = run(G, {**BASE, "n_agents": n}, "nearest", seed=0)
        res[kind] = m["evac_time_last_s"]
    return {"test": f"C. {n} people: 2 stair flights vs flat corridor", "expected": "stairs slower",
            "measured": f"{res['stairs']} s vs {res['corridor']} s", "unit": "",
            "pass": res["stairs"] > res["corridor"], "series": res}


def test_fundamental_diagram(n=400):
    nodes = {"room": (0, 0, 0), "j1": (1 * U, 0, 0), "j2": (31 * U, 0, 0), "test exit": (32 * U, 0, 0)}
    G = build_graph(nodes, [("room", "j1"), ("j1", "j2"), ("j2", "test exit")])
    for a, b in (("room", "j1"), ("j2", "test exit")):           # very wide doors so the corridor fills
        e = G.edges[a, b]
        e["width_m"], e["flow_cap"], e["jam_cap"] = 12.0, 30.0, 10_000
    m, _ = run(G, {**BASE, "n_agents": n, "trace_edge": ("j1", "j2")}, "nearest", seed=0)
    pts = [(dens, v) for _, dens, v in m["edge_trace"] if dens > 0]
    worst = max(abs(v - C.FREE_WALK_SPEED * weidmann_speed_factor(dens)) for dens, v in pts)
    return {"test": "D. Speed vs density on a crowded corridor", "expected": "on Weidmann curve",
            "measured": f"max deviation {worst:.3f} m/s, peak {max(p[0] for p in pts):.2f} p/m^2",
            "unit": "", "pass": worst < 0.005, "series": pts}


def plot(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping figure")
        return None
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    rows = results["B"]["series"]
    ax[0].plot([r[0] for r in rows], [r[2] for r in rows], "--", color="#888", label="Model target (1.3 p/s/m)")
    ax[0].plot([r[0] for r in rows], [r[1] for r in rows], "o", color="#1A56DB", label="Simulated")
    ax[0].set(xlabel="Door width (m)", ylabel="Flow (persons/s)", title="B. Door flow grows linearly with width")
    ax[0].legend(frameon=False)

    s = results["C"]["series"]
    ax[1].bar(["Flat corridor", "2 stair flights"], [s["corridor"], s["stairs"]], color=["#9CA3AF", "#C81E1E"])
    ax[1].set(ylabel="Time for 60 people to exit (s)", title="C. Stairs are a slower bottleneck")

    pts = results["D"]["series"]
    xs = [i / 50 for i in range(1, int(C.JAM_DENSITY * 50))]
    ax[2].plot(xs, [C.FREE_WALK_SPEED * weidmann_speed_factor(x) for x in xs], color="#888",
               label="Weidmann (1993) curve")
    ax[2].scatter([p[0] for p in pts], [p[1] for p in pts], s=8, color="#1A56DB", label="Simulated samples")
    ax[2].set(xlabel="Density (persons/m²)", ylabel="Speed (m/s)", title="D. Speed falls with crowd density")
    ax[2].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {"A": test_free_walk(), "B": test_door_flow(), "C": test_stairs(), "D": test_fundamental_diagram()}
    fig = plot(results, os.path.join(OUT_DIR, "verification.png"))
    lines = ["# Simulator verification", "",
             "Checks that the simulator implements its stated assumptions (see `evac/config.py`).",
             "This is verification, not validation against real evacuations.", "",
             "| Test | Expected | Measured | Result |", "| --- | --- | --- | --- |"]
    for r in results.values():
        lines.append(f"| {r['test']} | {r['expected']} {r['unit']} | {r['measured']} {r['unit']} | "
                     f"{'PASS' if r['pass'] else 'FAIL'} |")
    if fig:
        lines += ["", "![Verification](verification.png)"]
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for r in results.values():
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['test']}: expected {r['expected']} {r['unit']}, "
              f"measured {r['measured']} {r['unit']}")
    print(f"\nWrote {OUT_DIR}")
    return results


if __name__ == "__main__":
    main()
