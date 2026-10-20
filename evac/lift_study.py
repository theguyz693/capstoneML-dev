"""
evac/lift_study.py
Phase 2, item 7: evacuation lifts in the hospital.

Until now every lift was out of service in a fire, so wheelchair users and beds above the ground floor had one
way down: the ramp (the hospital's single point of failure, docs/hospital_results.md). Here the two bed lifts
can be FIRE-PROTECTED EVACUATION LIFTS (config.EVAC_LIFTS): usable in a fire by step-free people only, modelled
as a car (one group of up to LIFT_CAR_CAPACITY person-equivalents per round trip; a bed = 4).

Variants (hospital fire scenarios, paired seeds):
  no lifts                   lifts out of service in a fire (as before)
  evacuation lifts           the bed lifts work as evacuation lifts; the planner decides who uses them
  ramp blocked, no lifts     the ramp is out of service (e.g. smoke-logged) and lifts are off
  ramp blocked, evac lifts   the ramp is out of service, evacuation lifts work

Run:  python -m evac.lift_study   -> results/lifts.csv, docs/lifts.md, docs/figures/fig36_lifts.png
"""

import csv
import os
import statistics
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "lifts.csv")
DOC = os.path.join(ROOT, "docs", "lifts.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig36_lifts.png")
SCEN = ["hosp_day_fire_stairs1_f2", "hosp_day_fire_lobby", "hosp_day_spreading_f2", "hosp_night_wards_fire"]
VARIANTS = {"no lifts": (False, False), "evacuation lifts": (True, False),
            "ramp blocked, no lifts": (False, True), "ramp blocked, evac lifts": (True, True)}
METHODS = ["safe_shortest", "ccrp", "lookahead"]
SEEDS = list(range(10))
_G = None


def _job(args):
    scenario, variant, method, seed = args
    global _G
    from evac import config as C
    from evac.building import load_building
    from evac.hospital import scenarios
    from evac.simulator import run
    if _G is None:
        _G = load_building("hospital")
    lifts, ramp_blocked = VARIANTS[variant]
    G = _G.copy()
    if ramp_blocked:
        G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "ramp"])
    C.EVAC_LIFTS = lifts
    try:
        m, recs = run(G, scenarios()[scenario], method, seed)
    finally:
        C.EVAC_LIFTS = False

    def grp(p):
        ts = [r["t_out"] for r in recs if r["profile"] == p and r["t_out"] is not None]
        return statistics.mean(ts) if ts else None, len(ts), sum(r["profile"] == p for r in recs)
    bed_mean, bed_out, bed_n = grp("bed")
    wc_mean, wc_out, wc_n = grp("wheelchair")
    return {"scenario": scenario, "variant": variant, "method": method, "seed": seed,
            "everyone_out_s": m["evac_time_last_s"], "evacuated": m["evacuated"], "n": m["n_agents"],
            "not_out": m["n_agents"] - m["evacuated"], "exposed": m["exposed_agents"],
            "bed_mean_s": bed_mean, "bed_out": bed_out, "bed_n": bed_n,
            "wc_mean_s": wc_mean, "wc_out": wc_out, "wc_n": wc_n,
            "lift_users": sum(r["via"] == "lift" for r in recs)}


def collect(workers=2):
    import pandas as pd
    if os.path.exists(OUT):
        return pd.read_csv(OUT)
    jobs = [(s, v, m, sd) for s in SCEN for v in VARIANTS for m in METHODS for sd in SEEDS]
    rows, t0 = [], time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 60 == 0:
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
    from evac.final_report import BLUE, ORANGE, GREY, INK, MUTED, _style
    from evac.hospital_report import TITLE as HT
    agg = D.groupby(["scenario", "variant", "method"]).mean(numeric_only=True)
    V = list(VARIANTS)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(SCEN))
    cols = {"no lifts": GREY, "evacuation lifts": BLUE, "ramp blocked, no lifts": ORANGE,
            "ramp blocked, evac lifts": INK}
    for j, v in enumerate(V):
        ax[0].bar(x + (j - 1.5) * 0.2, [agg.loc[(s, v, "lookahead"), "not_out"] for s in SCEN], 0.2, color=cols[v], label=v)
        ax[1].bar(x + (j - 1.5) * 0.2, [agg.loc[(s, v, "lookahead"), "bed_mean_s"] for s in SCEN], 0.2, color=cols[v], label=v)
    for a, t, yl in ((ax[0], "People who cannot get out", "People (mean per run)"),
                     (ax[1], "Beds: mean time out", "Seconds")):
        a.set_xticks(x, [HT.get(s, s) for s in SCEN], rotation=15, ha="right", fontsize=8)
        a.set_ylabel(yl, fontsize=9, color=MUTED)
        a.set_title(t + " (look-ahead)", loc="left", fontsize=11, fontweight="bold")
        a.legend(fontsize=7, frameon=False, loc="upper left")
        _style(a)
    ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.3)
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.3)
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)
    plt.close(fig)

    def mean_over(v, m, col):
        return float(np.mean([agg.loc[(s, v, m), col] for s in SCEN]))
    L = ["# Evacuation lifts in the hospital (Phase 2)", "",
         "The two bed lifts as fire-protected evacuation lifts for wheelchair users and beds (car: one group of up "
         "to 4 person-equivalents per round trip, ~1 m/s, 30 s doors/loading; ASSUMPTIONS - see config.py). 4 hospital "
         f"fire scenarios x {len(SEEDS)} paired seeds. Generated by `python -m evac.lift_study`.", "", "## Findings", ""]
    rb_no = mean_over("ramp blocked, no lifts", "lookahead", "not_out")
    rb_ev = mean_over("ramp blocked, evac lifts", "lookahead", "not_out")
    L.append(f"1. **Lifts as a back-up save lives.** With the ramp out of service and no lifts, on average "
             f"{rb_no:.0f} people (all wheelchair users and beds above the ground floor) cannot get out. With evacuation "
             f"lifts: {rb_ev:.1f} (a few, only in the two spreading-fire scenarios, are cut off by the fire or still "
             "queueing at the simulation's 30-minute limit). Everyone-out time then: " + ", ".join(
                 f"{HT[s]} {agg.loc[(s, 'ramp blocked, evac lifts', 'lookahead'), 'everyone_out_s']:.0f} s" for s in SCEN) + ".")
    k = 2
    for m in METHODS:
        a = mean_over("no lifts", m, "everyone_out_s")
        b = mean_over("evacuation lifts", m, "everyone_out_s")
        ba = mean_over("no lifts", m, "bed_mean_s")
        bb = mean_over("evacuation lifts", m, "bed_mean_s")
        L.append(f"{k}. **With the ramp working, {m}**: everyone-out {a:.0f} s without lifts vs {b:.0f} s with them "
                 f"({b / a - 1:+.0%}); beds' mean {ba:.0f} -> {bb:.0f} s; "
                 f"{mean_over('evacuation lifts', m, 'lift_users'):.0f} people use the lifts.")
        k += 1
    L.append(f"{k}. **Reading it**: a lift car moves one bed (or two wheelchairs) per round trip of about a minute and a "
             "half; the 2.1 m ramp keeps moving beds continuously. So when the ramp works, sending people to the "
             "lifts makes them wait, and our planners, which book each floor-to-floor lift link separately, "
             "underestimate that wait - they send too many. Evacuation lifts are a back-up for when the ramp is lost, "
             "not a faster route in a 4-storey hospital with a ramp. Recommendation: keep lifts for people who have "
             "no safe ramp route.")
    L += ["", "![Lifts](figures/fig36_lifts.png)", "",
          "## Look-ahead, mean of seeds", "",
          "| Scenario | Variant | Everyone out (s) | Not out | Beds mean (s) | Wheelchairs mean (s) | Lift users | Exposed |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in SCEN:
        for v in V:
            r = agg.loc[(s, v, "lookahead")]
            f = lambda x: "-" if x != x else f"{x:.0f}"
            L.append(f"| {HT[s]} | {v} | {f(r['everyone_out_s'])} | {r['not_out']:.1f} | {f(r['bed_mean_s'])} | "
                     f"{f(r['wc_mean_s'])} | {r['lift_users']:.0f} | {r['exposed']:.1f} |")
    L += ["", "Settings: `config.EVAC_LIFTS`, `EVAC_LIFT_NAMES`, `LIFT_CAR_CAPACITY`, `LIFT_SPEED`, `LIFT_DOOR_LOAD_S`, "
          "`LIFT_LOAD_WINDOW_S`. Lift evacuation literature: Kinsey, Galea & Lawrence (2012); Ronchi & Nilsson (2013).", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[5:5 + k + 1]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    report(collect(a.workers))


if __name__ == "__main__":
    main()
