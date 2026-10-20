"""
evac/reproduce_all.py
Phase 2, item 8: ONE command that regenerates every result, table and figure of the project.

    python -m evac.reproduce_all                    run every step (re-uses finished results in results/)
    python -m evac.reproduce_all --fresh            move old results to results/_previous/<time>/ first, run all
    python -m evac.reproduce_all --only eta jams    just these steps
    python -m evac.reproduce_all --from safety      this step and everything after it
    python -m evac.reproduce_all --list             show the steps

Every simulation is seeded, so the same code gives the same numbers. Each step writes its raw runs to results/
and its report to docs/ (see docs/REPRODUCE.md). Times are for a 2-core laptop.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

# (key, module + args, cached result files (relative to results/), minutes on 2 cores, what it produces)
STEPS = [
    ("calibration", ["evac.calibration"], ["calibration_sim_vs_real.csv"], 1,
     "simulator calibrated on Juelich crowd experiments -> docs/calibration.md"),
    ("ml", ["evac.ml_v3"], ["delay_v3_campus.csv", "delay_v3_hospital.csv", "delay_v3_synthetic.csv",
                            "delay_v3_real_juelich.csv", "ml_v3_eval.json", "ml_v3_loop.csv"], 70,
     "crossing-time dataset, ML v3, real-data test -> docs/ml_v3.md, models/delay_lgbm_v3*"),
    ("main", ["evac.final_experiments"], ["final_A.csv", "final_B.csv", "final_C.csv", "final_D.csv", "final_E.csv",
                                          "final_G.csv", "final_H.csv"], 40,
     "campus: 6 methods x 10 scenarios x 30 seeds, ML in the loop, GA, lower bound, tipping point, freshness, ablations"),
    ("main_report", ["evac.final_report"], [], 1, "docs/final_results.md + figures 12-17"),
    ("hospital", ["evac.hospital_experiments"], ["hosp_A.csv", "hosp_D.csv", "hosp_S.csv", "hosp_W.json"], 15,
     "hospital: methods, lower bound, assumptions, weak spots"),
    ("hospital_report", ["evac.hospital_report"], [], 1, "docs/hospital_results.md + figures 18-21"),
    ("literature", ["evac.literature_experiments"], ["lit_A.csv", "lit_H.csv", "lit_check.json"], 12,
     "published baselines: quickest path, CASPER -> docs/literature_baselines.md"),
    ("validation", ["evac.validation"], [], 6, "door flow, stairs (NIST), IMO tests -> docs/validation.md"),
    ("safety", ["evac.safety_tradeoff"], ["safety_curve.csv"], 25, "safety-vs-speed curve -> docs/safety_tradeoff.md"),
    ("eta", ["evac.eta_study"], ["eta_study.csv"], 6, "time-to-exit on signs -> docs/eta_study.md"),
    ("priority", ["evac.priority_study"], ["priority.csv"], 15, "vulnerable people first -> docs/priority.md"),
    ("scalability", ["evac.scalability"], ["scalability.csv"], 20, "compute time vs building size -> docs/scalability.md"),
    ("jams", ["evac.jam_study"], ["jams.csv"], 8, "secondary jams -> docs/jam_study.md"),
    ("compliance", ["evac.compliance_study"], ["compliance.csv"], 15, "unknown compliance -> docs/compliance.md"),
    ("lifts", ["evac.lift_study"], ["lifts.csv"], 10, "evacuation lifts -> docs/lifts.md"),
    ("pack", ["evac.building_pack", "export"], [], 1, "building pack: buildings/campus.json, hospital.json"),
]
TESTS = sorted(f for f in os.listdir(os.path.join(ROOT, "tests")) if f.startswith("test_evac") and f.endswith(".py"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--fresh", action="store_true", help="move old results of the chosen steps away first")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    keys = [s[0] for s in STEPS]
    if a.list:
        for k, mod, _out, mins, what in STEPS:
            print(f"{k:16s} ~{mins:3d} min  python -m {' '.join(mod):34s} {what}")
        print(f"{'tests':16s}   ~5 min  {len(TESTS)} test files in tests/")
        print(f"total ~{sum(s[3] for s in STEPS) + 5} min on 2 cores (much less when results/ already has the runs)")
        return
    chosen = [s for s in STEPS if (not a.only or s[0] in a.only)]
    if a.start:
        if a.start not in keys:
            sys.exit(f"unknown step {a.start}; steps: {', '.join(keys)}")
        chosen = [s for s in chosen if keys.index(s[0]) >= keys.index(a.start)]
    os.makedirs(RES, exist_ok=True)
    if a.fresh:
        dest = os.path.join(RES, "_previous", time.strftime("%Y%m%d-%H%M%S"))
        for _k, _m, outs, _min, _w in chosen:
            for o in outs:
                p = os.path.join(RES, o)
                if os.path.exists(p):
                    os.makedirs(dest, exist_ok=True)
                    shutil.move(p, os.path.join(dest, o))
        print(f"old results moved to {dest}" if os.path.isdir(dest) else "no old results to move")
    log = open(os.path.join(RES, "reproduce_log.txt"), "a", encoding="utf-8")
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv[1:])}\n")
    failed = []
    t_all = time.time()
    for k, mod, _outs, mins, what in chosen:
        t0 = time.time()
        print(f"\n[{k}] python -m {' '.join(mod)}   (~{mins} min)  {what}", flush=True)
        r = subprocess.run([sys.executable, "-m", *mod], cwd=ROOT)
        dt = time.time() - t0
        status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
        log.write(f"{k:16s} {status:20s} {dt / 60:6.1f} min\n")
        log.flush()
        print(f"[{k}] {status} in {dt / 60:.1f} min", flush=True)
        if r.returncode:
            failed.append(k)
    if not a.no_tests and not a.only and not a.start:
        for t in TESTS:
            r = subprocess.run([sys.executable, os.path.join("tests", t)], cwd=ROOT, capture_output=True, text=True)
            ok = r.returncode == 0
            log.write(f"test {t:36s} {'ok' if ok else 'FAILED'}\n")
            print(f"[test] {t}: {'ok' if ok else 'FAILED'}", flush=True)
            if not ok:
                failed.append(t)
                print(r.stdout[-800:], r.stderr[-800:])
    log.write(f"total {(time.time() - t_all) / 60:.1f} min; failed: {failed or 'none'}\n")
    log.close()
    print(f"\nDone in {(time.time() - t_all) / 60:.1f} min. " + (f"FAILED: {', '.join(failed)}" if failed else "Everything reproduced."))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
