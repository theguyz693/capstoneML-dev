"""
evac/hospital_experiments.py
Session 6 experiments repeated on the SECOND building (oncology hospital), plus hospital-specific
questions. Resumable: each part writes results/hosp_<part>.csv / .json (delete to re-run).

  A  main       6 hospital scenarios x 7 methods x 30 paired seeds, outcomes per occupant profile
  D  optimum    lower bound, 6 scenarios x 3 seeds
  W  weakspots  block one stairwell / the ramp / an exit door / a key corridor (with and without fire)
  S  hospital assumptions: bed size, bed speed, bed preparation time, ramp speed, share of bedridden

Run:  python -m evac.hospital_experiments [--parts A D W S] [--workers 2]
Then: python -m evac.hospital_report      -> docs/hospital_results.md + figures
"""

import argparse
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

METHODS = ["nearest", "safe_shortest", "reactive", "ccrp", "lookahead", "lookahead_ml", "pred_ml"]
PROFILES = ("walk", "slow", "wheelchair", "bed")
SEEDS = list(range(30))
_G = _SC = None


def _init():
    global _G, _SC
    from evac.hospital import load_hospital, scenarios
    _G, _SC = load_hospital(), scenarios()


class Profiles:
    """Temporarily change config.PROFILES / config values (restored afterwards)."""

    def __init__(self, prof=None, cfg=None):
        self.prof = prof or {}
        self.cfg = {k: v for k, v in (cfg or {}).items() if not k.startswith("_")}

    def __enter__(self):
        from evac import config as C
        self.saved_p = {k: dict(v) for k, v in C.PROFILES.items()}
        self.saved_c = {k: getattr(C, k) for k in self.cfg}
        for name, upd in self.prof.items():
            C.PROFILES[name].update(upd)
        for k, v in self.cfg.items():
            setattr(C, k, v)

    def __exit__(self, *e):
        from evac import config as C
        for k, v in self.saved_p.items():
            C.PROFILES[k] = v
        for k, v in self.saved_c.items():
            setattr(C, k, v)


def _row(part, tag, name, method, seed, m, recs):
    row = {"part": part, "tag": tag, "scenario": name, "method": method, "seed": seed}
    for k in ("n_agents", "evacuated", "evac_time_last_s", "evac_time_mean_s", "evac_time_p95_s", "exposed_agents",
              "needs_assistance", "stranded", "timed_out", "max_queue", "route_changes", "compute_s"):
        row[k] = m.get(k)
    for p in PROFILES:
        ts = [r["t_out"] for r in recs if r.get("profile") == p and r["t_out"] is not None]
        row[f"{p}_n"] = sum(r.get("profile") == p for r in recs)
        row[f"{p}_mean_s"] = round(statistics.mean(ts), 1) if ts else None
        row[f"{p}_max_s"] = round(max(ts), 1) if ts else None
        row[f"{p}_exposed"] = sum(1 for r in recs if r.get("profile") == p and r["exposed"])
    amb = [r["t_out"] for r in recs if r.get("profile") in ("walk", "slow") and r["t_out"] is not None]
    row["ambulatory_last_s"] = max(amb) if amb else None
    return row


def _job(args):
    part, tag, name, over, method, seed, prof, cfg = args
    from evac.simulator import run
    with Profiles(prof, cfg):
        G = _G.copy()
        if cfg.get("_rebuild"):                       # ramp speed etc. live on the graph: rebuild it
            from evac.hospital import load_hospital
            G = load_hospital()
        m, recs = run(G, {**_SC[name], **over}, method, seed)
    return _row(part, tag, name, method, seed, m, recs)


def _run(jobs, out, workers):
    t0, rows = time.time(), []
    with Pool(workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                print(f"    {i + 1}/{len(jobs)} runs  {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["tag"], r["scenario"], r["method"], r["seed"]))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def part_A():
    _init()
    return [("A", "main", s, {}, m, sd, {}, {}) for s in _SC for m in METHODS for sd in SEEDS]


def part_D():
    _init()
    return [("D", "optimum", s, {}, "optimum", sd, {}, {}) for s in _SC for sd in range(3)]


def part_S():
    """One assumption at a time (others at default), fire scenario where the ramp is the only step-free way."""
    variants = [("default", {}, {})]
    for v in (2.0, 6.0):
        variants.append((f"bed size {v:g} person-eq.", {"bed": {"size": v}}, {}))
    for v in (0.3, 0.8):
        variants.append((f"bed speed x{v:g}", {"bed": {"speed": v}}, {}))
    for v in ((30, 120), (120, 480)):
        variants.append((f"bed preparation {v[0]}-{v[1]} s", {"bed": {"pre_s": v}}, {}))
    for v in (0.6, 1.3):
        variants.append((f"ramp speed {v:g} m/s", {}, {"RAMP_SPEED": v, "_rebuild": True}))
    jobs = []
    for tag, prof, cfg in variants:
        for m in ("safe_shortest", "lookahead"):
            for sd in range(10):
                jobs.append(("S", tag, "hosp_day_fire_stairs1_f2", {}, m, sd, prof, cfg))
    return jobs


def part_W():
    from evac.hospital import load_hospital, scenarios
    from evac.weakspots import analyse
    G, S = load_hospital(), scenarios()
    out = {}
    for name in ("hosp_day_peak", "hosp_day_fire_stairs1_f2"):
        t0 = time.time()
        out[name] = analyse(S[name], method="lookahead", seeds=(0, 1, 2), G0=G)
        print(f"    weak spots {name}: {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(RES, "hosp_W.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=["A", "D", "W", "S"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    os.makedirs(RES, exist_ok=True)
    for p in a.parts:
        out = os.path.join(RES, f"hosp_{p}.json" if p == "W" else f"hosp_{p}.csv")
        if os.path.exists(out):
            print(f"part {p}: exists, skipping ({out})")
            continue
        print(f"part {p} ...", flush=True)
        if p == "W":
            part_W()
        else:
            _run({"A": part_A, "D": part_D, "S": part_S}[p](), out, a.workers)


if __name__ == "__main__":
    main()
