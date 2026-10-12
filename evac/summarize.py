"""
evac/summarize.py
Turns runner CSV output into the tables that go in the report.

For each scenario x method: mean of each metric with a 95% bootstrap confidence interval.
For each scenario: paired comparison of every method against a reference method
(same seed = same people in the same rooms), as the median per-seed difference with a
95% bootstrap CI and a Wilcoxon signed-rank p-value (if scipy is installed).

Run:  python -m evac.summarize results/runs.csv --reference safe_shortest
"""

import argparse
import csv
import os
import random
import statistics
from collections import defaultdict

METRICS = ["evac_time_last_s", "evac_time_mean_s", "evac_time_p95_s", "exposed_agents",
           "max_queue", "route_changes", "needs_assistance"]


def boot_ci(values, stat=statistics.mean, n=2000, seed=0):
    if len(values) < 2:
        return (values[0], values[0]) if values else (None, None)
    rng = random.Random(seed)
    sims = sorted(stat([rng.choice(values) for _ in values]) for _ in range(n))
    return sims[int(0.025 * n)], sims[int(0.975 * n) - 1]


def wilcoxon_p(diffs):
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None
    if all(d == 0 for d in diffs) or len(diffs) < 6:
        return None
    return float(wilcoxon(diffs).pvalue)


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for m in METRICS + ["seed"]:
            if m in r and r[m] not in ("", "None"):
                r[m] = float(r[m])
            elif m in r:
                r[m] = None
    return rows


def summarize(rows, reference):
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["method"])].append(r)
    lines = ["# Results summary", "", "Mean [95% bootstrap CI] over seeds.", ""]
    scenarios = sorted({r["scenario"] for r in rows})
    for sc in scenarios:
        methods = sorted({m for s, m in by if s == sc})
        lines += [f"## {sc}", "", "| Method | n | Last out (s) | Mean time (s) | Exposed agents | Max queue | Route changes |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for m in methods:
            rs = by[(sc, m)]
            cells = []
            for met in ["evac_time_last_s", "evac_time_mean_s", "exposed_agents", "max_queue", "route_changes"]:
                vals = [r[met] for r in rs if r.get(met) is not None]
                if not vals:
                    cells.append("-")
                    continue
                lo, hi = boot_ci(vals)
                cells.append(f"{statistics.mean(vals):.1f} [{lo:.1f}, {hi:.1f}]")
            lines.append(f"| {m} | {len(rs)} | " + " | ".join(cells) + " |")
        if (sc, reference) in by:
            ref = {r["seed"]: r for r in by[(sc, reference)]}
            lines += ["", f"Paired difference vs `{reference}` in last-out time (negative = faster):", "",
                      "| Method | Median diff (s) | 95% CI | Wilcoxon p |", "| --- | --- | --- | --- |"]
            for m in methods:
                if m == reference:
                    continue
                diffs = [r["evac_time_last_s"] - ref[r["seed"]]["evac_time_last_s"]
                         for r in by[(sc, m)] if r["seed"] in ref and r["evac_time_last_s"] is not None
                         and ref[r["seed"]]["evac_time_last_s"] is not None]
                if not diffs:
                    continue
                lo, hi = boot_ci(diffs, stat=statistics.median)
                p = wilcoxon_p(diffs)
                lines.append(f"| {m} | {statistics.median(diffs):+.1f} | [{lo:+.1f}, {hi:+.1f}] | "
                             f"{'-' if p is None else f'{p:.3g}'} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=os.path.join("results", "runs.csv"))
    ap.add_argument("--reference", default="safe_shortest")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = summarize(load(a.csv), a.reference)
    out = a.out or os.path.splitext(a.csv)[0] + "_summary.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
