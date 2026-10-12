"""
evac/runner.py
Headless experiment runner: scenarios x methods x seeds -> one CSV row per run.

Usage (from the repo root):
  python -m evac.runner                                  # all scenarios, nearest + safe_shortest, 5 seeds
  python -m evac.runner --scenarios fire_j19_150 --methods nearest safe_shortest ml --seeds 30
  python -m evac.runner --out results/my_run.csv

Seeds are paired: seed k gives the SAME people in the SAME rooms for every method,
so methods can be compared run by run.
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # repo root, for campus_data / ml_routing_model

from evac.building import load_campus           # noqa: E402
from evac.simulator import run                 # noqa: E402

DEFAULT_SCEN = os.path.join(HERE, "scenarios.json")


def load_scenarios(path=DEFAULT_SCEN):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_matrix(scenarios, names, methods, seeds, base_graph=None, verbose=True):
    G0 = base_graph or load_campus()
    rows = []
    for name in names:
        sc = scenarios[name]
        for seed in seeds:
            for m in methods:
                metrics, _ = run(G0.copy(), sc, m, seed)
                row = {"scenario": name, **metrics}
                row["exits_used"] = json.dumps(row["exits_used"])
                rows.append(row)
                if verbose:
                    print(f"{name:28s} {m:14s} seed={seed:<3d} last_out={metrics['evac_time_last_s']}s "
                          f"exposed={metrics['exposed_agents']} max_queue={metrics['max_queue']} "
                          f"({metrics['compute_s']}s)")
    return rows


def write_csv(rows, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run evacuation experiments headless.")
    ap.add_argument("--scenarios", nargs="*", help="scenario names (default: all)")
    ap.add_argument("--scenario-file", default=DEFAULT_SCEN)
    ap.add_argument("--methods", nargs="*", default=["nearest", "safe_shortest"])
    ap.add_argument("--seeds", type=int, default=5, help="number of paired seeds (0..N-1)")
    ap.add_argument("--out", default=os.path.join("results", "runs.csv"))
    args = ap.parse_args(argv)

    scen = load_scenarios(args.scenario_file)
    names = args.scenarios or list(scen)
    rows = run_matrix(scen, names, args.methods, list(range(args.seeds)))
    write_csv(rows, args.out)
    print(f"\nWrote {len(rows)} runs to {args.out}")


if __name__ == "__main__":
    main()
