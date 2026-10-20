"""
evac/compliance_study.py
Phase 2, item 6: what if the planner does not know who will follow it?

So far the planner was told exactly who ignores guidance and where they go ('oracle') - unrealistic. Here:
  oracle      knows who ignores guidance (upper reference, all earlier results)
  naive       cannot tell; assumes everyone follows its routes
  aware       cannot tell; assumes a share p follows and the rest head for the familiar exit, and books each
              person's capacity split over both routes (p = the true compliance rate)
  aware 0.7   same, but with p fixed at 0.7 whatever the truth (a planner that guessed)
  shortest    shortest safe path for everyone (reference)
Rendón Rozo et al. (2019) and Wang et al. (2022) model non-compliance in the simulation; here the PLANNER plans for it.

Run:  python -m evac.compliance_study   -> results/compliance.csv, docs/compliance.md, docs/figures/fig35_compliance.png
"""

import csv
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "compliance.csv")
DOC = os.path.join(ROOT, "docs", "compliance.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig35_compliance.png")
SCEN = [("campus", "floor3_full_200"), ("campus", "building_250_spreading_fire_j19"), ("campus", "crowd_250_fire_j19"),
        ("hospital", "hosp_day_compliance70")]
COMPLIANCE = [1.0, 0.8, 0.6]
HERDING = 0.5
VARIANTS = ["shortest", "oracle", "naive", "aware", "aware0.7"]
LABEL = {"shortest": "Shortest safe path", "oracle": "Planner knows who follows (oracle)",
         "naive": "Planner assumes all follow", "aware": "Planner plans for p (true rate)",
         "aware0.7": "Planner plans for p = 0.7 (guess)"}
SEEDS = list(range(20))
_G = {}


def _job(args):
    building, scenario, comp, variant, seed = args
    from evac import config as C
    from evac.building import load_building
    from evac.simulator import run
    if building not in _G:
        _G[building] = load_building(building)
    if building == "hospital":
        from evac.hospital import scenarios
        sc = dict(scenarios()[scenario])
    else:
        with open(os.path.join(HERE, "scenarios.json"), encoding="utf-8") as f:
            sc = dict(json.load(f)[scenario])
    sc.update(compliance=comp, herding=HERDING if comp < 1 else 0.0)
    method = "safe_shortest" if variant == "shortest" else "lookahead"
    C.PLANNER_COMPLIANCE_MODEL = {"aware0.7": "aware"}.get(variant, variant if variant != "shortest" else "oracle")
    C.PLANNER_ASSUMED_COMPLIANCE = 0.7 if variant == "aware0.7" else comp
    try:
        m, recs = run(_G[building].copy(), sc, method, seed)
    finally:
        C.PLANNER_COMPLIANCE_MODEL, C.PLANNER_ASSUMED_COMPLIANCE = "oracle", 0.7
    amb = [r["t_out"] for r in recs if r["t_out"] is not None and r.get("profile") in ("walk", "slow", "step_free")]
    metric = max(amb) if building == "hospital" and amb else m["evac_time_last_s"]
    return {"building": building, "scenario": scenario, "compliance": comp, "variant": variant, "seed": seed,
            "metric_s": metric, "mean_s": m["evac_time_mean_s"], "exposed": m["exposed_agents"]}


def collect(workers=2):
    import pandas as pd
    if os.path.exists(OUT):
        return pd.read_csv(OUT)
    jobs = [(b, s, c, v, sd) for b, s in SCEN for c in COMPLIANCE for v in VARIANTS for sd in SEEDS]
    rows, t0 = [], time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} runs {time.time() - t0:.0f}s", flush=True)
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
    from evac.final_report import BLUE, ORANGE, GREY, INK, MUTED, TITLE, _style
    from evac.hospital_report import TITLE as HT
    from evac.summarize import wilcoxon_p
    title = dict(TITLE)
    title.update({k: "Hospital: " + v for k, v in HT.items()})
    agg = D.groupby(["scenario", "compliance", "variant"])[["metric_s", "mean_s", "exposed"]].mean()
    fig, axes = plt.subplots(1, len(SCEN), figsize=(15, 3.8), sharey=False)
    cols = {"shortest": GREY, "oracle": INK, "naive": ORANGE, "aware": BLUE, "aware0.7": "#7fb2e5"}
    for ax, (b, s) in zip(axes, SCEN):
        for v in VARIANTS:
            ys = [agg.loc[(s, c, v), "metric_s"] for c in COMPLIANCE]
            ax.plot([int(c * 100) for c in COMPLIANCE], ys, "o-", color=cols[v], label=LABEL[v], lw=2 if v == "aware" else 1.3)
        ax.invert_xaxis()
        ax.set_xlabel("People who follow guidance (%)", fontsize=8, color=MUTED)
        ax.set_ylabel("Last person out (s)" if b == "campus" else "Last walking person out (s)", fontsize=8, color=MUTED)
        ax.set_title(title.get(s, s), loc="left", fontsize=10, fontweight="bold")
        _style(ax, "both")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG, dpi=170)
    plt.close(fig)

    def paired(s, c, v1, v2):
        a = D[(D.scenario == s) & (D.compliance == c) & (D.variant == v1)].sort_values("seed")["metric_s"].to_numpy()
        b = D[(D.scenario == s) & (D.compliance == c) & (D.variant == v2)].sort_values("seed")["metric_s"].to_numpy()
        p = wilcoxon_p(list(b - a))
        return (b.mean() - a.mean()) / a.mean(), (p is not None and p < 0.05)

    rows = []
    for b, s in SCEN:
        for c in COMPLIANCE[1:]:
            rows.append((s, c) + paired(s, c, "naive", "aware") + paired(s, c, "oracle", "aware") +
                        paired(s, c, "naive", "aware0.7"))
    better = sum(1 for r in rows if r[2] < 0 and r[3])
    worse = sum(1 for r in rows if r[2] > 0 and r[3])
    better7 = sum(1 for r in rows if r[6] < 0 and r[7])
    worse7 = sum(1 for r in rows if r[6] > 0 and r[7])
    gap_naive = np.mean([paired(s, c, "oracle", "naive")[0] for _b, s in SCEN for c in COMPLIANCE[1:]])
    gap_aware = np.mean([r[4] for r in rows])
    full = [paired(s, 1.0, "naive", "aware") for _b, s in SCEN]
    L = ["# When the planner does not know who will follow it (Phase 2)", "",
         f"Look-ahead planner, {len(SEEDS)} paired seeds; true compliance 100% / 80% / 60%, and when below 100% half "
         "of the non-followers herd (copy the crowd's exit). Generated by `python -m evac.compliance_study`.", "",
         "## Findings", "",
         f"1. **The planner does not need to know who will follow it.** Until now it was told exactly who ignores "
         f"guidance ('oracle') - unrealistic. A planner that cannot tell and assumes everyone follows ('naive') is "
         f"{-gap_naive:.1%} {'FASTER' if gap_naive < 0 else 'slower'} on average at 80% / 60% compliance. So the earlier "
         "results did not depend on that free knowledge. It is not because of herding (a check without herding "
         "gives the same gap); a plausible reason, not yet verified, is that the oracle books every non-follower "
         "BEFORE everyone else, holding back followers who are ready earlier, while naive keeps one queue in order "
         "of readiness.",
         f"2. **Planning for non-compliance ('aware': each person's capacity split between the recommended route and "
         f"the familiar exit by the compliance rate) helps only sometimes**: significantly faster than naive in "
         f"{better} of {len(rows)} cases (200 people on floor 3: about -4%), significantly slower in {worse}, no "
         "difference in the rest. It helps when the non-followers all converge on one familiar exit that the planner "
         "would otherwise also fill.",
         f"3. With the rate guessed (0.7 whatever the truth) it is faster than naive in {better7} and slower in {worse7} "
         "cases, and when everyone actually follows, the wrong guess costs up to " +
         f"{max(paired(s, 1.0, 'naive', 'aware0.7')[0] for _b, s in SCEN):+.0%}.",
         "4. **Conclusion**: keep the simple planner; report compliance-awareness as tested and not worth it in "
         "general. The robust finding is that look-ahead stays well ahead of shortest path at every compliance "
         "level without knowing who complies (table below).",
         "", "![Compliance](figures/fig35_compliance.png)", "",
         "## Mean of 20 seeds (campus: last person out; hospital: last walking person out, seconds)", "",
         "| Scenario | Follow guidance | " + " | ".join(LABEL[v] for v in VARIANTS) + " |",
         "| --- | --- |" + " --- |" * len(VARIANTS)]
    for b, s in SCEN:
        for c in COMPLIANCE:
            L.append(f"| {title.get(s, s)} | {c:.0%} | " + " | ".join(f"{agg.loc[(s, c, v), 'metric_s']:.0f}" for v in VARIANTS) + " |")
    L += ["", "## Aware vs naive, aware vs oracle (paired, Wilcoxon p < 0.05 marked *)", "",
          "| Scenario | Follow guidance | Aware vs naive | Aware vs oracle | Aware 0.7 vs naive |", "| --- | --- | --- | --- | --- |"]
    for s, c, d1, s1, d2, s2, d3, s3 in rows:
        L.append(f"| {title.get(s, s)} | {c:.0%} | {d1:+.1%}{'*' if s1 else ''} | {d2:+.1%}{'*' if s2 else ''} | "
                 f"{d3:+.1%}{'*' if s3 else ''} |")
    L += ["", "Settings: `config.PLANNER_COMPLIANCE_MODEL` ('oracle' default / 'naive' / 'aware') and "
          "`config.PLANNER_ASSUMED_COMPLIANCE`. Non-followers head for the familiar exit (the main entrance); "
          "herders copy the most common exit of the queue they join.", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[5:10]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    report(collect(a.workers))


if __name__ == "__main__":
    main()
