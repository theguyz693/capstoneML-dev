"""
evac/final_experiments.py
All final experiments for the report (Session 6C). Resumable: each part writes results/final_<part>.csv
and is skipped if that file already exists (delete it to re-run).

  A  main      10 scenarios x 6 methods x 30 paired seeds (+ fairness group statistics per run)
  B  mlloop    predictor-in-the-loop: formula vs ML vs cautious ML, 4 scenarios x 30 seeds
  C  ga        genetic-algorithm exit assignment (offline benchmark), 3 scenarios x 2 seeds, pop 24 x 30 generations
  D  optimum   theoretical lower bound, 10 scenarios x 3 seeds
  E  tipping   crowd size 25-300 on one floor: when does coordination start to matter?
  F  (fairness - computed from A)
  G  fresh     how fresh must crowd data be? re-planning interval 2-80 s
  H  ablation  remove one planner ingredient at a time

Run:  python -m evac.final_experiments [--parts A B ...] [--workers 2]
Then: python -m evac.final_report      (tables + figures -> docs/final_results.md)
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

RES = os.path.join(ROOT, "results")
SCEN_MAIN = ["baseline_150_no_fire", "fire_j19_150", "fire_j6_east_exit_150", "lecture_hall_f1_120",
             "crowd_250_fire_j19", "floor3_full_200", "floor3_full_200_fire_stairs1",
             "floor3_200_spreading_fire_stairs1", "building_250_spreading_fire_j19", "floor3_200_compliance60_herding"]
METHODS_MAIN = ["nearest", "safe_shortest", "reactive", "ccrp", "lookahead", "lookahead_ml"]
SEEDS = list(range(30))

_G = None
_SC = None


def _init():
    global _G, _SC
    from evac.building import load_campus
    _G = load_campus()
    with open(os.path.join(HERE, "scenarios.json"), encoding="utf-8") as f:
        _SC = json.load(f)


def _job(args):
    """args = (part, tag, scenario_name, scenario_overrides, method, seed, config_overrides)"""
    import statistics
    from evac import config as C
    from evac.simulator import run
    part, tag, name, over, method, seed, cfg = args
    saved = {k: getattr(C, k) for k in cfg}
    for k, v in cfg.items():
        setattr(C, k, v)
    try:
        m, recs = run(_G.copy(), {**_SC[name], **over}, method, seed)
    finally:
        for k, v in saved.items():
            setattr(C, k, v)
    row = {"part": part, "tag": tag, "scenario": name, "method": method, "seed": seed}
    for k in ("n_agents", "evacuated", "evac_time_last_s", "evac_time_mean_s", "evac_time_p95_s", "exposed_agents",
              "needs_assistance", "stranded", "timed_out", "max_queue", "route_changes", "forced_detours", "compute_s"):
        row[k] = m.get(k)
    # fairness: who pays? (evacuated people only)
    def grp(pred):
        ts = [r["t_out"] for r in recs if r["t_out"] is not None and pred(r)]
        return (round(statistics.mean(ts), 1) if ts else None), (round(max(ts), 1) if ts else None)
    fl = {r["agent"]: _G.nodes[r["origin"]]["floor"] for r in recs}
    row["sf_mean_s"], row["sf_max_s"] = grp(lambda r: r["step_free"])
    row["upper_mean_s"], row["upper_max_s"] = grp(lambda r: fl[r["agent"]] >= 4)
    row["lower_mean_s"], row["lower_max_s"] = grp(lambda r: fl[r["agent"]] <= 2)
    return row


def _run_jobs(jobs, out, workers):
    import csv
    t0 = time.time()
    rows = []
    with Pool(workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"    {i + 1}/{len(jobs)} runs  {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["tag"], r["scenario"], r["method"], r["seed"]))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def part_A():
    return [("A", "main", s, {}, m, sd, {}) for s in SCEN_MAIN for m in METHODS_MAIN for sd in SEEDS]


def part_B():
    sc = ["floor3_full_200", "lecture_hall_f1_120", "floor3_200_compliance60_herding", "floor3_200_spreading_fire_stairs1"]
    return [("B", "mlloop", s, {}, m, sd, {}) for s in sc for m in ("pred_formula", "pred_ml", "pred_ml_upper") for sd in SEEDS]


def part_E():
    jobs = []
    for n in (25, 50, 100, 150, 200, 300):
        for m in ("safe_shortest", "reactive", "ccrp", "lookahead"):
            for sd in range(10):
                jobs.append(("E", f"n={n}", "floor3_full_200", {"n_agents": n}, m, sd, {}))
    return jobs


def part_G():
    jobs = []
    for s in ("floor3_200_compliance60_herding", "floor3_200_spreading_fire_stairs1"):
        for r in (2, 5, 10, 20, 40, 80):
            for sd in range(10):
                jobs.append(("G", f"replan={r}", s, {}, "lookahead", sd, {"REPLAN_S": float(r)}))
    return jobs


def part_H():
    variants = [("full", {}, {}, "lookahead"),
                ("no capacity booking", {"horizon_s": 0.001}, {}, "lookahead"),
                ("no switching threshold", {"switch_gain": 0.0}, {}, "lookahead"),
                ("no hazard detour penalty", {}, {"HAZARD_DETOUR_PENALTY_S": 0.0}, "lookahead"),
                ("nominal (not sustainable) capacity", {}, {"PLANNER_USE_SUSTAINABLE_CAP": False}, "lookahead"),
                ("ignores non-compliant people", {}, {"PLANNER_BOOK_NONCOMPLIANT": False}, "lookahead"),
                ("no re-planning (one-shot)", {}, {}, "ccrp")]
    jobs = []
    for s in ("floor3_full_200", "floor3_200_compliance60_herding", "floor3_200_spreading_fire_stairs1"):
        for tag, over, cfg, m in variants:
            for sd in range(10):
                jobs.append(("H", tag, s, over, m, sd, cfg))
    return jobs


def part_D(workers):
    jobs = [("D", "optimum", s, {}, "optimum", sd, {}) for s in SCEN_MAIN for sd in range(3)]
    _run_jobs(jobs, os.path.join(RES, "final_D.csv"), workers)


def part_C():
    import csv
    from evac.ga import optimise
    _init()
    rows = []
    for s in ("lecture_hall_f1_120", "floor3_full_200", "crowd_250_fire_j19"):
        for sd in (0, 1):
            t0 = time.time()
            m, _ = optimise(_SC[s], sd, pop=24, gens=30, G0=_G, verbose=False)
            rows.append({"part": "C", "tag": "ga", "scenario": s, "method": "ga", "seed": sd,
                         "evac_time_last_s": m["evac_time_last_s"], "evac_time_mean_s": m["evac_time_mean_s"],
                         "exposed_agents": m["exposed_agents"], "route_changes": 0,
                         "ga_evaluations": m["ga_evaluations"], "compute_s": round(time.time() - t0, 1)})
            print(f"    GA {s} seed {sd}: {m['evac_time_last_s']} s ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(RES, "final_C.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=["A", "B", "E", "G", "H", "D", "C"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    os.makedirs(RES, exist_ok=True)
    simple = {"A": part_A, "B": part_B, "E": part_E, "G": part_G, "H": part_H}
    for p in a.parts:
        out = os.path.join(RES, f"final_{p}.csv")
        if os.path.exists(out):
            print(f"part {p}: exists, skipping ({out})")
            continue
        print(f"part {p} ...", flush=True)
        if p in simple:
            _run_jobs(simple[p](), out, a.workers)
        elif p == "D":
            part_D(a.workers)
        elif p == "C":
            part_C()


if __name__ == "__main__":
    main()
