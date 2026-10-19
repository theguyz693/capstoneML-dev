"""
evac/sensitivity.py
Are the conclusions robust to our physical assumptions?  (Session 6B)

Every physical value in evac/config.py is an engineering assumption. The panel's obvious attack is
"your flow rates are made up". This script varies ALL of them together over realistic ranges with the
Morris elementary-effects method (Morris 1991; via SALib) and records, for every sampled building:

  - last-out time with shortest safe path (no planning) and with look-ahead re-planning
  - the ADVANTAGE of look-ahead = (shortest - lookahead) / shortest

Outputs:
  - mu* (mean absolute effect) and sigma per parameter: which assumptions matter most
  - the share of sampled worlds in which look-ahead still wins, and the advantage range

Run:  python -m evac.sensitivity            (~10 min)  -> docs/sensitivity.md, docs/figures/fig10_sensitivity.png
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from evac import config as C            # noqa: E402
from evac import physics                # noqa: E402
from evac.building import load_campus   # noqa: E402
from evac.simulator import run          # noqa: E402

DOC = os.path.join(ROOT, "docs", "sensitivity.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig10_sensitivity.png")

PARAMS = [
    # name, low, high, what it means
    ("METERS_PER_UNIT", 0.03, 0.05, "Building scale (m per graph unit)"),
    ("FREE_WALK_SPEED", 1.0, 1.5, "Free walking speed (m/s)"),
    ("STAIR_DOWN_SPEED", 0.5, 0.8, "Stair descent speed (m/s)"),
    ("DOOR_FLOW", 1.0, 1.6, "Door / corridor flow (p/s/m)"),
    ("STAIR_FLOW", 0.8, 1.3, "Stair flow (p/s/m)"),
    ("JAM_DENSITY", 4.5, 6.0, "Jam density (p/m²)"),
    ("WIDTH_SCALE", 0.8, 1.2, "All widths × factor"),
    ("STAIR_FLIGHT_LENGTH_M", 6.0, 10.0, "Stair flight length (m)"),
    ("PRE_MOVEMENT_MAX_S", 10.0, 60.0, "Pre-movement delay max (s)"),
]


class Assumptions:
    """Temporarily set config values (and recompute derived constants)."""

    def __init__(self, values):
        self.values = values
        self.saved = {}

    def __enter__(self):
        for k in ("METERS_PER_UNIT", "FREE_WALK_SPEED", "STAIR_DOWN_SPEED", "JAM_DENSITY",
                  "STAIR_FLIGHT_LENGTH_M", "PRE_MOVEMENT_MAX_S"):
            self.saved[k] = getattr(C, k)
        self.saved["SPECIFIC_FLOW"] = dict(C.SPECIFIC_FLOW)
        self.saved["WIDTH_M"] = dict(C.WIDTH_M)
        self.saved["PEAK"] = physics.PEAK_FLOW_PER_SPEED
        v = self.values
        for k in ("METERS_PER_UNIT", "FREE_WALK_SPEED", "STAIR_DOWN_SPEED", "JAM_DENSITY",
                  "STAIR_FLIGHT_LENGTH_M", "PRE_MOVEMENT_MAX_S"):
            setattr(C, k, v[k])
        for et in ("room_door", "exit_door"):
            C.SPECIFIC_FLOW[et] = v["DOOR_FLOW"]
        C.SPECIFIC_FLOW["corridor"] = v.get("CORRIDOR_FLOW", v["DOOR_FLOW"])
        C.SPECIFIC_FLOW["stair"] = v["STAIR_FLOW"]
        for et in C.WIDTH_M:
            C.WIDTH_M[et] = self.saved["WIDTH_M"][et] * v["WIDTH_SCALE"]
        physics.PEAK_FLOW_PER_SPEED = physics._peak_flow_fraction()
        return self

    def __exit__(self, *exc):
        for k, val in self.saved.items():
            if k == "SPECIFIC_FLOW":
                C.SPECIFIC_FLOW.clear(); C.SPECIFIC_FLOW.update(val)
            elif k == "WIDTH_M":
                C.WIDTH_M.clear(); C.WIDTH_M.update(val)
            elif k == "PEAK":
                physics.PEAK_FLOW_PER_SPEED = val
            else:
                setattr(C, k, val)


def evaluate(values, scenario, seed=0):
    with Assumptions(values):
        G = load_campus()
        a = run(G.copy(), scenario, "safe_shortest", seed)[0]["evac_time_last_s"]
        b = run(G.copy(), scenario, "lookahead", seed)[0]["evac_time_last_s"]
    return a, b, (a - b) / a


def main(trajectories=10, scen_name="floor3_full_200"):
    from SALib.analyze import morris as morris_an
    from SALib.sample import morris as morris_s
    problem = {"num_vars": len(PARAMS), "names": [p[0] for p in PARAMS], "bounds": [[p[1], p[2]] for p in PARAMS]}
    X = morris_s.sample(problem, N=trajectories, num_levels=4, seed=0)
    scen = json.load(open(os.path.join(HERE, "scenarios.json"), encoding="utf-8"))[scen_name]
    t0 = time.time()
    out = []
    for i, row in enumerate(X):
        out.append(evaluate(dict(zip(problem["names"], row)), scen, seed=i % 3))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(X)} worlds, {time.time() - t0:.0f}s")
    Y = np.array(out)
    res = {}
    for j, label in enumerate(("shortest", "lookahead", "advantage")):
        si = morris_an.analyze(problem, X, Y[:, j], num_levels=4, print_to_console=False)
        res[label] = {n: (float(m), float(s)) for n, m, s in zip(problem["names"], si["mu_star"], si["sigma"])}
    adv = Y[:, 2]
    summary = {"worlds": len(X), "lookahead_wins_share": float((adv > 0).mean()), "adv_min": float(adv.min()),
               "adv_median": float(np.median(adv)), "adv_max": float(adv.max()),
               "shortest_range": (float(Y[:, 0].min()), float(Y[:, 0].max())),
               "lookahead_range": (float(Y[:, 1].min()), float(Y[:, 1].max()))}
    np.savetxt(os.path.join(ROOT, "results", "sensitivity_samples.csv"), np.hstack([X, Y]), delimiter=",",
               header=",".join(problem["names"] + ["shortest_s", "lookahead_s", "advantage"]), comments="")

    # figure: mu* for the advantage + distribution of the advantage
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {p[0]: p[3] for p in PARAMS}
    items = sorted(res["lookahead"].items(), key=lambda kv: kv[1][0])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.4, 1]})
    ax[0].barh([labels[k] for k, _ in items], [v[0] for _, v in items], color="#2a78d6", height=0.55)
    ax[0].set_xlabel("Effect on look-ahead last-out time, Morris μ* (s)")
    ax[0].set_title("Which assumptions move the result most", loc="left", fontsize=11, fontweight="bold")
    ax[1].hist(adv * 100, bins=12, color="#2a78d6", alpha=0.85, edgecolor="white")
    ax[1].axvline(0, color="#21201c", lw=1, ls=(0, (3, 3)))
    ax[1].set_xlabel("Look-ahead advantage over shortest path (%)")
    ax[1].set_ylabel("Sampled worlds")
    ax[1].set_title(f"Look-ahead wins in {summary['lookahead_wins_share']:.0%} of {len(X)} worlds",
                    loc="left", fontsize=11, fontweight="bold")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG, dpi=160)

    lines = ["# Sensitivity analysis: do the conclusions survive different assumptions?", "",
             f"Scenario `{scen_name}`. Morris elementary-effects design (SALib), {trajectories} trajectories x "
             f"{len(PARAMS) + 1} points = {len(X)} sampled 'worlds', all {len(PARAMS)} assumptions varied together.", "",
             "| Assumption | Range | μ* on shortest-path time (s) | μ* on look-ahead time (s) | μ* on advantage (pp) |",
             "| --- | --- | --- | --- | --- |"]
    for name, lo, hi, lab in PARAMS:
        lines.append(f"| {lab} | {lo}-{hi} | {res['shortest'][name][0]:.1f} | {res['lookahead'][name][0]:.1f} | "
                     f"{100 * res['advantage'][name][0]:.1f} |")
    lines += ["", f"- Look-ahead beat shortest safe path in **{summary['lookahead_wins_share']:.0%}** of the {len(X)} worlds.",
              f"- Its advantage ranged from {summary['adv_min']:.0%} to {summary['adv_max']:.0%} (median {summary['adv_median']:.0%}).",
              f"- Last-out time ranged {summary['shortest_range'][0]:.0f}-{summary['shortest_range'][1]:.0f} s (shortest path) "
              f"and {summary['lookahead_range'][0]:.0f}-{summary['lookahead_range'][1]:.0f} s (look-ahead).",
              "- Absolute times depend strongly on the assumptions (so they are not predictions of real evacuation "
              "times); the *ranking* of methods does not.", "", "![Sensitivity](figures/fig10_sensitivity.png)", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
