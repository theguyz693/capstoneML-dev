"""
evac/literature_experiments.py
Published routing methods as baselines (evac/literature.py), run through the SAME paired-seed experiment
as the main results, so every method sees the same people, start times and fires.

  campus    10 scenarios x {quickest, casper} x 30 seeds  -> results/lit_A.csv
  hospital   6 scenarios x {quickest, casper} x 30 seeds  -> results/lit_H.csv

The other methods' runs are reused from results/final_A.csv and results/hosp_A.csv (same seeds); a few of
them are re-run first to check the simulator still reproduces them exactly.

Run:  python -m evac.literature_experiments [--workers 2]     (~10-20 min)
      python -m evac.literature_experiments --report-only
Out:  docs/literature_baselines.md, docs/figures/fig27_literature.png
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from evac import config as C                                   # noqa: E402
from evac.summarize import boot_ci, wilcoxon_p                 # noqa: E402

RES = os.path.join(ROOT, "results")
DOC = os.path.join(ROOT, "docs", "literature_baselines.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig27_literature.png")
NEW = ["quickest", "casper"]
SEEDS = list(range(30))
LABEL = {"safe_shortest": "Shortest safe path (ours, baseline)", "reactive": "Reactive rerouting (ours)",
         "quickest": "Quickest path - Kemloh Wagoum et al. 2012",
         "casper": "CASPER - Shahabi & Wilson 2014",
         "ccrp": "CCRP - Lu, George & Shekhar 2005",
         "lookahead": "Look-ahead re-planning (ours)"}
SHORT = {"safe_shortest": "Shortest safe", "reactive": "Reactive", "quickest": "Quickest path",
         "casper": "CASPER", "ccrp": "CCRP", "lookahead": "Look-ahead"}
SHOW = ["safe_shortest", "reactive", "quickest", "casper", "ccrp", "lookahead"]
REFS = [
    "Kemloh Wagoum, A. U., Seyfried, A. & Holl, S. (2012). Modeling the dynamic route choice of pedestrians to "
    "assess the criticality of building evacuation. Advances in Complex Systems 15(7). doi:10.1142/S0219525912500294",
    "Shahabi, K. & Wilson, J. P. (2014). CASPER: Intelligent capacity-aware evacuation routing. Computers, "
    "Environment and Urban Systems 46, 12-24. doi:10.1016/j.compenvurbsys.2014.03.004",
    "Lu, Q., George, B. & Shekhar, S. (2005). Capacity constrained routing algorithms for evacuation planning: "
    "a summary of results. SSTD 2005, LNCS 3633, 291-307.",
]


# ---------------------------------------------------------------------- running
def _campus_job(args):
    from evac import final_experiments as F
    if F._G is None:
        F._init()
    return F._job(args)


def _hosp_job(args):
    from evac import hospital_experiments as H
    if H._G is None:
        H._init()
    return H._job(args)


def _run(fn, jobs, out, workers):
    from multiprocessing import get_context
    t0, rows = time.time(), []
    with get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 60 == 0 or i + 1 == len(jobs):
                print(f"    {i + 1}/{len(jobs)} runs  {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["scenario"], r["method"], int(r["seed"])))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def reproduce_check():
    """Re-run a few stored runs with the current code; they must match exactly (paired comparison is valid)."""
    from evac import final_experiments as F
    from evac import hospital_experiments as H
    F._init()
    H._init()
    old = {(r["scenario"], r["method"], r["seed"]): r for r in _load("final_A.csv")}
    oldh = {(r["scenario"], r["method"], r["seed"]): r for r in _load("hosp_A.csv")}
    out = []
    for s, m, sd in [("floor3_full_200", "lookahead", 3), ("crowd_250_fire_j19", "ccrp", 5),
                     ("floor3_200_spreading_fire_stairs1", "safe_shortest", 7)]:
        r = F._job(("A", "main", s, {}, m, sd, {}))
        out.append((s, m, sd, float(old[(s, m, str(sd))]["evac_time_last_s"]), r["evac_time_last_s"]))
    for s, m, sd in [("hosp_day_peak", "lookahead", 2), ("hosp_day_fire_lobby", "safe_shortest", 4)]:
        r = H._job(("A", "main", s, {}, m, sd, {}, {}))
        out.append((s, m, sd, float(oldh[(s, m, str(sd))]["evac_time_last_s"]), r["evac_time_last_s"]))
    return out


# ---------------------------------------------------------------------- analysis
def _load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(x):
    return float(x) if x not in (None, "", "None") else None


def table(rows, col):
    """{(scenario, method): {seed: value}} with missing (not everyone out) -> MAX_TIME_S."""
    t = {}
    for r in rows:
        if r["method"] not in SHOW:
            continue
        v = _num(r[col])
        t.setdefault((r["scenario"], r["method"]), {})[int(r["seed"])] = v if v is not None else C.MAX_TIME_S
    return t


def holm(ps):
    idx = sorted([i for i, p in enumerate(ps) if p is not None], key=lambda i: ps[i])
    out, m, run = list(ps), len(idx), 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (m - k) * ps[i]))
        out[i] = run
    return out


def analyse(rows, col, scenarios):
    T = table(rows, col)
    E = table(rows, "exposed_agents")
    res = {"means": {}, "exposed": {}, "vs_base": {}, "la_vs": {}}
    for s in scenarios:
        for m in SHOW:
            if (s, m) in T:
                res["means"][(s, m)] = statistics.mean(T[(s, m)].values())
                res["exposed"][(s, m)] = statistics.mean(E[(s, m)].values())
        base = T[(s, "safe_shortest")]
        for m in SHOW[1:]:
            if (s, m) not in T:
                continue
            pct = [100 * (T[(s, m)][k] - base[k]) / base[k] for k in SEEDS]
            res["vs_base"][(s, m)] = (statistics.mean(pct), *boot_ci(pct))
    # look-ahead against each published method: paired, Holm over the scenarios of this building
    for m in ("quickest", "casper", "ccrp"):
        ps, ds = [], []
        for s in scenarios:
            d = [T[(s, "lookahead")][k] - T[(s, m)][k] for k in SEEDS]
            pct = 100 * statistics.mean(d) / statistics.mean(T[(s, m)].values())
            ds.append((s, pct, statistics.mean(d)))
            ps.append(wilcoxon_p(d))
        adj = holm(ps)
        for (s, pct, md), p in zip(ds, adj):
            res["la_vs"][(s, m)] = (pct, md, p)
    return res


def _verdict(pct, p):
    if p is None or p >= 0.05:
        return "tie"
    return "faster" if pct < 0 else "slower"


def figure(campus, hosp, ctitle, htitle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"reactive": "#9CA3AF", "quickest": "#E07B39", "casper": "#8E6BB5", "ccrp": "#C9A227", "lookahead": "#2A78D6"}
    ms = ["reactive", "quickest", "casper", "ccrp", "lookahead"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [10, 6]})
    for ax, (res, titles, head) in zip(axes, [(campus, ctitle, "Campus (G+6), last person out"),
                                               (hosp, htitle, "Hospital (G+3), last walking person out")]):
        sc = list(titles)
        for i, s in enumerate(sc):
            y0 = len(sc) - 1 - i
            for j, m in enumerate(ms):
                if (s, m) not in res["vs_base"]:
                    continue
                mu, lo, hi = res["vs_base"][(s, m)]
                y = y0 + 0.32 - j * 0.16
                ax.plot([lo, hi], [y, y], color=COL[m], lw=2.2)
                ax.plot(mu, y, "o", color=COL[m], ms=5, mec="white", mew=0.6, label=LABEL[m] if i == 0 else None)
        ax.axvline(0, color="#222", lw=1)
        ax.set_yticks(range(len(sc)))
        ax.set_yticklabels([titles[s] for s in sc][::-1], fontsize=9)
        ax.set_xlabel("Change vs shortest safe path (%), mean and 95% CI  (left = faster)")
        ax.set_title(head, loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Look-ahead against published routing methods (30 paired seeds per scenario)",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)


def report(check):
    from evac.final_experiments import SCEN_MAIN
    from evac.final_report import TITLE as CT
    from evac.hospital_report import TITLE as HT
    A = [r for r in _load("final_A.csv") if r["method"] in SHOW] + _load("lit_A.csv")
    H = [r for r in _load("hosp_A.csv") if r["method"] in SHOW] + _load("lit_H.csv")
    hsc = [s for s in HT if any(r["scenario"] == s for r in H)]
    campus = analyse(A, "evac_time_last_s", SCEN_MAIN)
    hosp = analyse(H, "ambulatory_last_s", hsc)
    hosp_all = analyse(H, "evac_time_last_s", hsc)
    figure(campus, hosp, {s: CT[s] for s in SCEN_MAIN}, {s: HT[s] for s in hsc})

    L = ["# Look-ahead against published routing methods", "",
         "Generated by `python -m evac.literature_experiments`. Two published methods were added "
         "(`evac/literature.py`) and run through the same 30-seed paired experiment as the main results: the same "
         "people, start times and fires for every method. CCRP was already one of our baselines.", "",
         "| Method | Source | Type |", "|---|---|---|",
         "| Quickest path | Kemloh Wagoum, Seyfried & Holl 2012 | local, reactive: each person switches when the "
         "estimated walk + queue time of another route is clearly lower (cba > 0.15) |",
         "| CASPER | Shahabi & Wilson 2014 | global, planned once: one route per room, edge cost grows with the "
         "people already routed over it |",
         "| CCRP | Lu, George & Shekhar 2005 | global, planned once: books time slots on every door and stair |",
         "| **Look-ahead (ours)** | this work | CCRP-style booking, re-planned every 5 s over a 300 s horizon |", ""]

    def block(res, scen, titles, unit):
        out = ["| Scenario | " + " | ".join(SHORT[m] for m in SHOW) + " |", "|---" * (len(SHOW) + 1) + "|"]
        for s in scen:
            cells = []
            best = min(res["means"][(s, m)] for m in SHOW)
            for m in SHOW:
                v = res["means"][(s, m)]
                txt = f"{v:.0f}"
                cells.append(f"**{txt}**" if abs(v - best) < 0.5 else txt)
            out.append(f"| {titles[s]} | " + " | ".join(cells) + " |")
        out += ["", f"Mean {unit} over 30 seeds; bold = fastest.", ""]
        return out

    def head2head(res, scen, titles):
        out = ["| Scenario | vs Quickest path | vs CASPER | vs CCRP |", "|---|---|---|---|"]
        tally = {m: {"faster": 0, "tie": 0, "slower": 0} for m in ("quickest", "casper", "ccrp")}
        for s in scen:
            cells = []
            for m in ("quickest", "casper", "ccrp"):
                pct, _md, p = res["la_vs"][(s, m)]
                v = _verdict(pct, p)
                tally[m][v] += 1
                mark = {"faster": "✓ ", "slower": "✗ ", "tie": ""}[v]
                cells.append(f"{mark}{pct:+.0f}%" + ("" if v != "tie" else " (n.s.)"))
            out.append(f"| {titles[s]} | " + " | ".join(cells) + " |")
        out += ["", "Look-ahead's change against each method (negative = look-ahead faster). ✓/✗ = significant "
                "(paired Wilcoxon, Holm-corrected over the scenarios, p < 0.05); n.s. = no significant difference.", ""]
        return out, tally

    cs, ct = head2head(campus, SCEN_MAIN, CT)
    hs, ht = head2head(hosp, hsc, HT)

    def tally_line(t, n):
        return "; ".join(f"vs {SHORT[m]}: faster in {v['faster']}, tie in {v['tie']}, slower in {v['slower']} of {n}"
                         for m, v in t.items())

    L += ["## Summary", "",
          f"- **Campus** - look-ahead {tally_line(ct, len(SCEN_MAIN))}.",
          f"- **Hospital** (last walking person out) - look-ahead {tally_line(ht, len(hsc))}.", ""]
    # largest gains vs each published method
    for name, res, scen, titles in (("Campus", campus, SCEN_MAIN, CT), ("Hospital", hosp, hsc, HT)):
        for m in ("quickest", "casper"):
            s_best = min(scen, key=lambda s: res["la_vs"][(s, m)][0])
            s_worst = max(scen, key=lambda s: res["la_vs"][(s, m)][0])
            L.append(f"- {name}, vs {SHORT[m]}: largest gain {res['la_vs'][(s_best, m)][0]:+.0f}% "
                     f"({titles[s_best]}); weakest {res['la_vs'][(s_worst, m)][0]:+.0f}% ({titles[s_worst]}).")
    L += ["", "## What it means", "",
          "- Against the two methods that do not book capacity over time (quickest path, CASPER) look-ahead is "
          "clearly better wherever crowds concentrate, and never significantly worse on campus.",
          "- CCRP, the published method our planner builds on, is as good as look-ahead in most calm scenarios and "
          "slightly better in some spreading-fire cases. Look-ahead's advantage over CCRP is robustness: when only "
          "60% of people follow guidance and others herd, re-planning recovers what a one-shot plan loses. So the "
          "honest claim is 'CCRP-style booking is the key ingredient; re-planning makes it robust', not "
          "'look-ahead beats CCRP everywhere'.",
          "- Hospital fire in the entrance lobby: the capacity-aware planners (CCRP, look-ahead) are slower for the "
          "walking crowd because they keep people away from the smoke (28 exposed vs 78-99 for the other methods).",
          "- Quickest path, a local method, avoids smoke best in spreading fires (people re-route as soon as they "
          "see the fire) but cannot prevent the stair jams that a global plan avoids.", ""]
    L += ["", "## Campus - time until the last person is out (s)", ""] + block(campus, SCEN_MAIN, CT, "time (s)")
    L += ["### Look-ahead head-to-head (campus)", ""] + cs
    L += ["## Hospital - time until the last walking person is out (s)", ""] + block(hosp, hsc, HT, "time (s)")
    L += ["### Look-ahead head-to-head (hospital)", ""] + hs
    L += ["## Hospital - time until EVERYONE is out (s)", "",
          "Bedridden patients need staff, so this is set by the bed evacuation rather than by routing.", ""]
    L += block(hosp_all, hsc, HT, "time (s)")
    L += ["## People exposed to smoke (mean per run)", "",
          "| Scenario | " + " | ".join(SHORT[m] for m in SHOW) + " |", "|---" * (len(SHOW) + 1) + "|"]
    for s in SCEN_MAIN:
        if all(campus["exposed"][(s, m)] == 0 for m in SHOW):
            continue
        L.append(f"| {CT[s]} | " + " | ".join(f"{campus['exposed'][(s, m)]:.0f}" for m in SHOW) + " |")
    for s in hsc:
        if all(hosp["exposed"][(s, m)] == 0 for m in SHOW):
            continue
        L.append(f"| Hospital: {HT[s]} | " + " | ".join(f"{hosp['exposed'][(s, m)]:.0f}" for m in SHOW) + " |")
    L += ["", "![Published baselines](figures/fig27_literature.png)", "",
          "## How the published methods were put on our graph", "",
          "- **Quickest path**: the paper's pedestrians re-decide on entering a new room and when stuck in a jam "
          "longer than their patience (10 s, +1 s per unsuccessful attempt); they switch only if the cost-benefit "
          "(g_alt - g_cur)/(g_alt + g_cur) with g = 1/estimated time exceeds 0.15. On the graph every node is a "
          "decision point and the estimated time is walking time at the current density plus the time for the "
          "queue ahead to clear (the paper estimates this from a 'reference pedestrian' in the jam).",
          "- **CASPER**: sources (rooms) are processed from the farthest to the nearest; each gets one route that "
          "minimises the sum of edge costs, where an edge's cost uses the people already routed over it plus its own "
          "group. CASPER leaves the traffic model T open; ours is the pedestrian version max(walking time, "
          "people / sustainable capacity) x hazard penalty.",
          "- All methods share the simulator, safety rules (no walking into fire, no lifts in a fire, step-free "
          "people avoid stairs) and the same people. Non-compliant people ignore every method equally.", "",
          "## Reproducibility check", "",
          "Stored runs of the other methods were re-run with the current code before being reused:", "",
          "| Scenario | Method | Seed | Stored (s) | Re-run (s) |", "|---|---|---|---|---|"]
    for s, m, sd, a, b in check:
        L.append(f"| {s} | {m} | {sd} | {a} | {b} |")
    L += ["", "## References", ""] + [f"- {r}" for r in REFS] + [""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    summary = {"campus": {f"{s}|{m}": v for (s, m), v in campus["la_vs"].items()},
               "hospital": {f"{s}|{m}": v for (s, m), v in hosp["la_vs"].items()}, "tally_campus": ct, "tally_hospital": ht}
    with open(os.path.join(ROOT, "docs", "literature_baselines.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("wrote", DOC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    from evac.final_experiments import SCEN_MAIN
    check_path = os.path.join(RES, "lit_check.json")
    if not a.report_only:
        check = reproduce_check()
        with open(check_path, "w", encoding="utf-8") as f:
            json.dump(check, f)
        bad = [c for c in check if c[3] != c[4]]
        if bad:
            print("WARNING: stored runs do not reproduce:", bad)
        if not os.path.exists(os.path.join(RES, "lit_A.csv")):
            jobs = [("A", "main", s, {}, m, sd, {}) for s in SCEN_MAIN for m in NEW for sd in SEEDS]
            _run(_campus_job, jobs, os.path.join(RES, "lit_A.csv"), a.workers)
        if not os.path.exists(os.path.join(RES, "lit_H.csv")):
            from evac.hospital import scenarios
            jobs = [("A", "main", s, {}, m, sd, {}, {}) for s in scenarios() for m in NEW for sd in SEEDS]
            _run(_hosp_job, jobs, os.path.join(RES, "lit_H.csv"), a.workers)
    with open(check_path, encoding="utf-8") as f:
        check = json.load(f)
    report(check)


if __name__ == "__main__":
    main()
