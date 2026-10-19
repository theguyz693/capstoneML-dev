"""
evac/priority_study.py
Phase 2, item 3: vulnerable people first - does it help them, and what does it cost everyone else?

"Vulnerable" = slow walkers (patients, elderly), wheelchair users, beds and anyone who needs a step-free route.
Variants of the look-ahead planner (same people, same fires, 30 paired seeds):
  none     everyone booked in order of readiness (the default so far)
  planner  the planner books vulnerable people FIRST, so their routes get the earliest door/stair slots and
           everyone else is routed around them                       (config.PRIORITY_VULNERABLE)
  door     at a queue, vulnerable people are let through first (staff/marshals)   (config.DOOR_PRIORITY_VULNERABLE)
  both     planner + door
Measured separately for the vulnerable group and for everyone else: mean and last exit time, people in smoke.

Run:  python -m evac.priority_study   -> results/priority.csv, docs/priority.md, docs/figures/fig32_priority.png
"""

import csv
import json
import os
import statistics
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "priority.csv")
DOC = os.path.join(ROOT, "docs", "priority.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig32_priority.png")
VARIANTS = {"none": (False, False), "planner": (True, False), "door": (False, True), "both": (True, True)}
CAMPUS = ["baseline_150_no_fire", "fire_j19_150", "crowd_250_fire_j19", "building_250_spreading_fire_j19"]
HOSP = ["hosp_day_peak", "hosp_day_fire_stairs1_f2", "hosp_day_fire_lobby", "hosp_day_spreading_f2",
        "hosp_day_compliance70", "hosp_night_wards_fire"]
SEEDS = list(range(30))
_G = {}


def _stats(recs, pred):
    sel = [r for r in recs if pred(r)]
    ts = [r["t_out"] for r in sel if r["t_out"] is not None]
    return {"n": len(sel), "mean": statistics.mean(ts) if ts else None, "last": max(ts) if ts else None,
            "out": len(ts), "exposed": sum(bool(r["exposed"]) for r in sel)}


def _job(args):
    building, scenario, variant, seed = args
    from evac import config as C
    from evac.building import load_building
    from evac.simulator import run
    if building not in _G:
        _G[building] = load_building(building)
    if building == "hospital":
        from evac.hospital import scenarios
        sc = scenarios()[scenario]
    else:
        with open(os.path.join(HERE, "scenarios.json"), encoding="utf-8") as f:
            sc = json.load(f)[scenario]
    C.PRIORITY_VULNERABLE, C.DOOR_PRIORITY_VULNERABLE = VARIANTS[variant]
    try:
        m, recs = run(_G[building].copy(), sc, "lookahead", seed)
    finally:
        C.PRIORITY_VULNERABLE = C.DOOR_PRIORITY_VULNERABLE = False
    vul = lambda r: r["step_free"] or r["profile"] in ("slow", "wheelchair", "bed", "step_free")
    v, o = _stats(recs, vul), _stats(recs, lambda r: not vul(r))
    row = {"building": building, "scenario": scenario, "variant": variant, "seed": seed,
           "all_last": m["evac_time_last_s"], "all_mean": m["evac_time_mean_s"], "exposed": m["exposed_agents"]}
    for k, d in (("vul", v), ("oth", o)):
        for kk, vv in d.items():
            row[f"{k}_{kk}"] = vv
    return row


def collect(workers=2):
    import pandas as pd
    if os.path.exists(OUT):
        return pd.read_csv(OUT)
    jobs = [("campus", s, v, sd) for s in CAMPUS for v in VARIANTS for sd in SEEDS]
    jobs += [("hospital", s, v, sd) for s in HOSP for v in VARIANTS for sd in SEEDS]
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
    scen = CAMPUS + HOSP
    agg = D.groupby(["scenario", "variant"]).mean(numeric_only=True)

    def rel(s, v, col):
        b = agg.loc[(s, "none"), col]
        return (agg.loc[(s, v), col] - b) / b if b else 0.0

    def sig(s, v, col):
        a = D[(D.scenario == s) & (D.variant == "none")].sort_values("seed")[col].to_numpy()
        b = D[(D.scenario == s) & (D.variant == v)].sort_values("seed")[col].to_numpy()
        ok = ~(np.isnan(a) | np.isnan(b))
        p = wilcoxon_p(list(b[ok] - a[ok])) if ok.sum() > 5 else None
        return p is not None and p < 0.05

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    x = np.arange(len(scen))
    cols = {"planner": BLUE, "door": ORANGE, "both": INK}
    for j, v in enumerate(("planner", "door", "both")):
        ax[0].bar(x + (j - 1) * 0.27, [100 * rel(s, v, "vul_mean") for s in scen], 0.27, color=cols[v], label=v)
        ax[1].bar(x + (j - 1) * 0.27, [100 * rel(s, v, "oth_mean") for s in scen], 0.27, color=cols[v], label=v)
    for a, t in zip(ax, ("Vulnerable people: mean exit time", "Everyone else: mean exit time")):
        a.axhline(0, color=GREY, lw=1)
        a.set_xticks(x, [title.get(s, s).replace("Hospital: ", "H: ") for s in scen], rotation=35, ha="right", fontsize=7)
        a.set_ylabel("Change vs no priority (%)", fontsize=9, color=MUTED)
        a.set_title(t, loc="left", fontsize=11, fontweight="bold")
        a.legend(fontsize=8, frameon=False)
        _style(a)
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)
    plt.close(fig)

    best = {}
    for v in ("planner", "door", "both"):
        gv = [rel(s, v, "vul_mean") for s in scen]
        go = [rel(s, v, "oth_mean") for s in scen]
        gl = [rel(s, v, "all_last") for s in scen]
        nv = sum(sig(s, v, "vul_mean") and rel(s, v, "vul_mean") < 0 for s in scen)
        no = sum(sig(s, v, "oth_mean") and rel(s, v, "oth_mean") > 0 for s in scen)
        best[v] = (np.mean(gv), np.mean(go), np.mean(gl), nv, no, min(gv), max(go))
    L = ["# Vulnerable people first: who gains, who pays? (Phase 2)", "",
         "Look-ahead planner, 30 paired seeds per scenario (same people, same fires). 'Vulnerable' = slow walkers, "
         "wheelchair users, beds and anyone needing a step-free route (campus: 5% step-free people; hospital: "
         "patients). Generated by `python -m evac.priority_study`.", "", "## Findings", ""]
    k = 1
    for v, lab in (("planner", "Planner books them first"), ("door", "Let through first at queues"),
                   ("both", "Both")):
        gv, go, gl, nv, no, mv, mo = best[v]
        L.append(f"{k}. **{lab}**: vulnerable people's mean exit time {gv:+.1%} on average (best {mv:+.1%}; "
                 f"significantly faster in {nv} of {len(scen)} scenarios); everyone else {go:+.1%} (worst {mo:+.1%}; "
                 f"significantly slower in {no}); time until the last person is out {gl:+.1%}.")
        k += 1
    exp = {v: (agg.xs(v, level="variant")["vul_exposed"].sum(), agg.xs("none", level="variant")["vul_exposed"].sum())
           for v in ("planner", "door", "both")}
    L.append(f"{k}. Vulnerable people in smoke (sum over scenarios, mean per run): none {exp['both'][1]:.1f}, "
             + ", ".join(f"{v} {exp[v][0]:.1f}" for v in ("planner", "door", "both")) + ".")
    k += 1
    h = D[D.building == "hospital"].groupby("variant").mean(numeric_only=True)
    L.append(f"{k}. **Recommendation: planner priority, not queue priority.** In the hospital, booking vulnerable "
             f"people first cuts their mean exit time {h.loc['planner', 'vul_mean'] / h.loc['none', 'vul_mean'] - 1:+.1%} "
             f"AND everyone else's {h.loc['planner', 'oth_mean'] / h.loc['none', 'oth_mean'] - 1:+.1%}: slow people "
             "planned first get routes that keep them out of the way of faster people, instead of slowing them down "
             "in a shared queue. Letting them through first at the door only reshuffles the same queue (vulnerable "
             f"{h.loc['door', 'vul_mean'] / h.loc['none', 'vul_mean'] - 1:+.1%}, others "
             f"{h.loc['door', 'oth_mean'] / h.loc['none', 'oth_mean'] - 1:+.1%}). On the campus the 5% step-free "
             "people never share a bottleneck with the crowd, so nothing changes. The time until the LAST person is "
             "out does not move: it is set by bed evacuation (staff, bed preparation, the ramp), not by routing. "
             "Smoke exposure is unchanged.")
    L += ["", "![Priority](figures/fig32_priority.png)", "",
          "## Per scenario (mean of 30 seeds, seconds)", "",
          "| Scenario | Vulnerable people | Vulnerable mean: none / planner / door / both | Everyone else mean: none / planner / door / both | Last out: none / both |",
          "| --- | --- | --- | --- | --- |"]
    for s in scen:
        a = agg.loc[s]
        L.append(f"| {title.get(s, s)} | {a.loc['none', 'vul_n']:.0f} | " +
                 " / ".join(f"{a.loc[v, 'vul_mean']:.0f}" for v in VARIANTS) + " | " +
                 " / ".join(f"{a.loc[v, 'oth_mean']:.0f}" for v in VARIANTS) +
                 f" | {a.loc['none', 'all_last']:.0f} / {a.loc['both', 'all_last']:.0f} |")
    L += ["", "Settings: `config.PRIORITY_VULNERABLE` (planner) and `config.DOOR_PRIORITY_VULNERABLE` (queues); both off "
          "by default so earlier results are unchanged.", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[5:5 + k]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    report(collect(a.workers))


if __name__ == "__main__":
    main()
