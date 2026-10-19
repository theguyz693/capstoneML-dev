"""
evac/safety_tradeoff.py
Phase 2, item 1: the SAFETY-vs-SPEED trade-off, made visible.

Most evacuation-routing papers report only evacuation time; a few report exposure, almost none show how one is
traded for the other. Our capacity planner has one dial, config.SAFETY_WEIGHT, that scales how strongly it avoids
smoky corridors (0 = only the hard rule "never walk into fire", 1 = default, >1 = avoid smoke at almost any time
cost). Sweeping it traces a curve of (time, people exposed to smoke). The published methods and our other
baselines are single points on the same axes (same 30 seeds, results/final_A.csv, lit_A.csv, hosp_A.csv,
lit_H.csv). A method is DOMINATED if some setting of look-ahead is both at least as fast and at least as safe.

Run:  python -m evac.safety_tradeoff   -> results/safety_curve.csv, docs/safety_tradeoff.md, docs/figures/fig30_safety_speed.png
"""

import csv
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "safety_curve.csv")
DOC = os.path.join(ROOT, "docs", "safety_tradeoff.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig30_safety_speed.png")
WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SEEDS = list(range(30))
# scenarios where smoke and speed can actually conflict (people exposed > 0 for some method)
CAMPUS = ["floor3_200_spreading_fire_stairs1", "building_250_spreading_fire_j19", "crowd_250_fire_j19", "fire_j19_150"]
HOSP = ["hosp_day_fire_lobby", "hosp_day_spreading_f2"]
OTHERS = ["nearest", "safe_shortest", "reactive", "quickest", "casper", "ccrp"]
NAMES = {"nearest": "Nearest exit", "safe_shortest": "Shortest safe path", "reactive": "Reactive re-route",
         "quickest": "Quickest path (2012)", "casper": "CASPER (2014)", "ccrp": "CCRP (2005)"}


def _job(args):
    building, scenario, w, seed = args
    if building == "campus":
        from evac import final_experiments as F
        if F._G is None:
            F._init()
        r = F._job(("S", "safety", scenario, {}, "lookahead", seed, {"SAFETY_WEIGHT": w}))
        r["metric_s"] = r["evac_time_last_s"]
    else:
        from evac import hospital_experiments as H
        if H._G is None:
            H._init()
        r = H._job(("S", "safety", scenario, {}, "lookahead", seed, {}, {"SAFETY_WEIGHT": w}))
        r["metric_s"] = r["ambulatory_last_s"]
    return {"building": building, "scenario": scenario, "weight": w, "seed": seed, "metric_s": r["metric_s"],
            "mean_s": r["evac_time_mean_s"], "exposed": r["exposed_agents"]}


def sweep(workers=2):
    import pandas as pd
    if os.path.exists(OUT):
        return pd.read_csv(OUT)
    jobs = [("campus", s, w, sd) for s in CAMPUS for w in WEIGHTS for sd in SEEDS]
    jobs += [("hospital", s, w, sd) for s in HOSP for w in WEIGHTS for sd in SEEDS]
    rows, t0 = [], time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} runs {time.time() - t0:.0f}s", flush=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return pd.read_csv(OUT)


def method_points():
    import pandas as pd
    pts = []
    for f, b, col in (("final_A.csv", "campus", "evac_time_last_s"), ("lit_A.csv", "campus", "evac_time_last_s"),
                      ("hosp_A.csv", "hospital", "ambulatory_last_s"), ("lit_H.csv", "hospital", "ambulatory_last_s")):
        p = os.path.join(RES, f)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d["method"].isin(OTHERS) & d["seed"].isin(SEEDS) & d["scenario"].isin(CAMPUS + HOSP)]
        g = d.groupby(["scenario", "method"]).agg(metric_s=(col, "mean"), exposed=("exposed_agents", "mean"))
        g["building"] = b
        pts.append(g.reset_index())
    return pd.concat(pts, ignore_index=True)


def report(S, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evac.final_report import BLUE, GREY, INK, MUTED, TITLE, _style
    from evac.hospital_report import TITLE as HTITLE
    title = dict(TITLE)
    title.update({k: "Hospital: " + v for k, v in HTITLE.items()})
    curve = S.groupby(["scenario", "weight"]).agg(metric_s=("metric_s", "mean"), exposed=("exposed", "mean")).reset_index()
    scen = CAMPUS + HOSP
    marks = {"nearest": "v", "safe_shortest": "s", "reactive": "D", "quickest": "^", "casper": "P", "ccrp": "X"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    lines, dominated = [], []
    for ax, s in zip(axes.flat, scen):
        c = curve[curve["scenario"] == s].sort_values("weight")
        ax.plot(c["metric_s"], c["exposed"], "-o", color=BLUE, lw=2, ms=4, label="Look-ahead, safety dial 0 -> 8", zorder=3)
        for _, r in c.iterrows():
            if r["weight"] in (0.0, 1.0, 8.0):
                ax.annotate(f"{r['weight']:g}", (r["metric_s"], r["exposed"]), textcoords="offset points",
                            xytext=(5, 4), fontsize=7, color=BLUE)
        pts = P[P["scenario"] == s]
        for _, r in pts.iterrows():
            ax.plot(r["metric_s"], r["exposed"], marks[r["method"]], color=GREY if r["method"] != "ccrp" else INK,
                    ms=7, label=NAMES[r["method"]], zorder=4)
            dom = c[(c["metric_s"] <= r["metric_s"] * 1.01) & (c["exposed"] <= r["exposed"] + 0.5)]
            if len(dom):
                b = dom.iloc[0]
                dominated.append((s, r["method"], b["weight"]))
        ax.set_title(title.get(s, s), loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Last person out (s)" if s in CAMPUS else "Last walking person out (s)", fontsize=8, color=MUTED)
        ax.set_ylabel("People exposed to smoke", fontsize=8, color=MUTED)
        _style(ax, "both")
    h, l = axes.flat[0].get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    for ax in axes.flat[1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(ll, hh)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG, dpi=170)
    plt.close(fig)

    # findings
    find = []
    for s in scen:
        c = curve[curve["scenario"] == s].set_index("weight")
        z, d, hi = c.loc[0.0], c.loc[1.0], c.loc[8.0]
        find.append((s, z, d, hi))
    n_pts = len(P)
    dom_methods = sorted({(s, m) for s, m, _ in dominated})
    by_method = {}
    for s, m in dom_methods:
        by_method.setdefault(m, []).append(s)
    L = ["# Safety vs speed: the trade-off made visible (Phase 2)", "",
         "`config.SAFETY_WEIGHT` is one dial for the capacity planner: 0 = only the hard rule (never walk into "
         "fire), 1 = default, 8 = avoid smoke at almost any time cost. Sweeping it gives a curve of (time, people "
         "exposed). Other methods are single points from the main experiments (same 30 seeds). "
         "Generated by `python -m evac.safety_tradeoff`.", "", "## Findings", ""]
    k = 1
    for s, z, d, hi in find:
        L.append(f"{k}. **{title.get(s, s)}**: dial 0 -> {z['metric_s']:.0f} s / {z['exposed']:.1f} exposed; "
                 f"default (1) -> {d['metric_s']:.0f} s / {d['exposed']:.1f}; dial 8 -> {hi['metric_s']:.0f} s / "
                 f"{hi['exposed']:.1f}.")
        k += 1
    L.append(f"{k}. **Dominated methods** (some dial setting of look-ahead is at least as fast, within 1%, AND at least "
             f"as safe): {len(dom_methods)} of {n_pts} method-scenario points. " +
             "; ".join(f"{NAMES[m]} in {len(v)} of {len(scen)} scenarios" for m, v in sorted(by_method.items())) + ".")
    k += 1
    safer = []
    for _, r in P.iterrows():
        c = curve[curve["scenario"] == r["scenario"]]
        if r["exposed"] < c["exposed"].min() - 0.5:
            safer.append(f"{NAMES[r['method']]} in {title.get(r['scenario'], r['scenario'])} "
                         f"({r['metric_s']:.0f} s / {r['exposed']:.0f} exposed vs best look-ahead {c['exposed'].min():.0f})")
    if safer:
        L.append(f"{k}. **Where look-ahead is NOT the safest at any setting:** " + "; ".join(safer) + ". Likely reason: "
                 "quickest path re-decides for each person at every junction with the latest smoke, while look-ahead "
                 "plans every 5 s as if the fire stayed where it is now. This is the gap the fire-forecast planner (Phase 3) "
                 "targets.")
        k += 1
    L.append(f"{k}. Reading it: a method's single number hides where it sits on this curve. Reporting time alone "
             "rewards walking people through smoke (e.g. nearest exit); reporting the curve lets a fire engineer "
             "pick the trade-off on purpose.")
    L += ["", "![Safety vs speed](figures/fig30_safety_speed.png)", "",
          "## The curve (mean of 30 seeds)", "", "| Scenario | " + " | ".join(f"dial {w:g}" for w in WEIGHTS) + " |",
          "| --- |" + " --- |" * len(WEIGHTS)]
    for s in scen:
        c = curve[curve["scenario"] == s].set_index("weight")
        L.append(f"| {title.get(s, s)} | " + " | ".join(f"{c.loc[w, 'metric_s']:.0f} s / {c.loc[w, 'exposed']:.0f}"
                                                         for w in WEIGHTS) + " |")
    L += ["", "Each cell: time / people exposed. Campus = last person out; hospital = last walking person out.", "",
          "## Other methods (same seeds)", "", "| Scenario | " + " | ".join(NAMES[m] for m in OTHERS) + " |",
          "| --- |" + " --- |" * len(OTHERS)]
    for s in scen:
        row = []
        for m in OTHERS:
            r = P[(P["scenario"] == s) & (P["method"] == m)]
            row.append(f"{r['metric_s'].iloc[0]:.0f} s / {r['exposed'].iloc[0]:.0f}" if len(r) else "-")
        L.append(f"| {title.get(s, s)} | " + " | ".join(row) + " |")
    L += ["", "How the dial works: the planner's cost of walking an edge is walking time x (1 + w x (smoke multiplier "
          "- 1)) + w x 300 s if the edge is smoky (hazard >= 0.3). Edges on fire (hazard >= 0.92) are never used, "
          "whatever w. Exposure = people who crossed an edge with hazard >= 0.3 (plus people defending in place "
          "reached by smoke).", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[5:5 + len(find) + 2]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    S = sweep(a.workers)
    report(S, method_points())


if __name__ == "__main__":
    main()
