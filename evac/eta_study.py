"""
evac/eta_study.py
Phase 2, item 2: can a sign honestly tell people how long it will take to get out?

During look-ahead evacuations (campus + hospital, 20 seeds) we stop every 20 s and, for everyone who is inside
and on the move, record two predictions of their remaining time to exit:
  ml    sum of the learned crossing times (ML v3) along their route, crowd frozen as it is now
        (what /api/research/route shows today)
  plan  the capacity planner's own bookings: everyone's routes booked on every door and stair in order, so the
        crowd that reaches a stair BEFORE this person is counted (a forecast, not a snapshot)
...then compare with when the person really got out.

A sign shows a RANGE. Ranges are calibrated with conformal prediction (split by seed: seeds 0-9 calibrate,
seeds 10-19 test) on the log of actual/predicted, so "90 %" is checked on evacuations never used to set it.

Run:  python -m evac.eta_study   -> results/eta_study.csv, docs/eta_study.md, docs/figures/fig31_eta.png,
                                     evac/eta_calibration.json (used by /api/research/route and /signs)
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
OUT = os.path.join(RES, "eta_study.csv")
DOC = os.path.join(ROOT, "docs", "eta_study.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig31_eta.png")
CAL = os.path.join(HERE, "eta_calibration.json")
CAMPUS = ["floor3_full_200", "lecture_hall_f1_120", "crowd_250_fire_j19", "building_250_spreading_fire_j19"]
HOSP = ["hosp_day_peak", "hosp_day_fire_lobby", "hosp_day_spreading_f2"]
SEEDS = list(range(20))
CAL_SEEDS = set(range(10))
PROBES = list(range(20, 481, 30))
_G = {}


def _job(args):
    building, scenario, seed = args
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
    m, _ = run(_G[building].copy(), {**sc, "eta_probe_at": PROBES}, "lookahead", seed)
    return [{"building": building, "scenario": scenario, "seed": seed, **r} for r in m["eta_rows"]]


def collect(workers=2):
    import pandas as pd
    if os.path.exists(OUT):
        return pd.read_csv(OUT)
    jobs = [("campus", s, sd) for s in CAMPUS for sd in SEEDS] + [("hospital", s, sd) for s in HOSP for sd in SEEDS]
    rows, t0 = [], time.time()
    with Pool(workers) as pool:
        for i, rs in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            rows += rs
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(jobs)} evacuations, {len(rows)} predictions, {time.time() - t0:.0f}s", flush=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return pd.read_csv(OUT)


def conformal(df, col, alpha=0.10):
    """Multiplicative bias factor (median of actual/pred) and a symmetric log-scale margin q so that
    [pred*b*e^-q, pred*b*e^q] holds the actual time for (1 - alpha) of calibration predictions."""
    import numpy as np
    r = np.log(np.maximum(df["actual_s"], 1.0) / np.maximum(df[col], 1.0))
    b = float(np.median(r))
    q = float(np.quantile(np.abs(r - b), 1 - alpha))
    return b, q


BINS = [0, 30, 60, 120, 240, 1e9]


def mondrian(cal, col, alpha=0.10):
    """Conformal calibration per band of predicted time (short estimates are relatively less certain)."""
    import numpy as np
    out = []
    for lo, hi in zip(BINS, BINS[1:]):
        g = cal[(cal[col] > lo) & (cal[col] <= hi)]
        if len(g) < 50:
            g = cal
        b, q = conformal(g, col, alpha)
        out.append({"from_s": lo, "to_s": hi, "bias_factor": float(np.exp(b)), "range_factor": float(np.exp(q)), "n": int(len(g))})
    return out


def apply_bands(bands, pred):
    import numpy as np
    pred = np.maximum(np.asarray(pred, float), 1.0)
    b = np.ones_like(pred)
    f = np.ones_like(pred)
    for band in bands:
        m = (pred > band["from_s"]) & (pred <= band["to_s"])
        b[m], f[m] = band["bias_factor"], band["range_factor"]
    c = pred * b
    return c, c / f, c * f


def evaluate(D):
    import numpy as np
    D = D.dropna(subset=["actual_s", "eta_ml", "eta_plan"])
    D = D[D["actual_s"] > 0]
    cal, test = D[D["seed"].isin(CAL_SEEDS)], D[~D["seed"].isin(CAL_SEEDS)]
    out = {"rows": int(len(D)), "test_rows": int(len(test))}
    for col in ("eta_ml", "eta_plan"):
        b, q = conformal(cal, col)
        p = test[col].to_numpy()
        y = test["actual_s"].to_numpy()
        pc = np.maximum(p, 1.0) * np.exp(b)
        lo, hi = pc * np.exp(-q), pc * np.exp(q)
        out[col] = {"bias_factor": float(np.exp(b)), "log_margin": q,
                    "MAE raw": float(np.mean(np.abs(p - y))), "mean error raw": float(np.mean(p - y)),
                    "MAE corrected": float(np.mean(np.abs(pc - y))),
                    "coverage": float(np.mean((y >= lo) & (y <= hi))),
                    "mean width s": float(np.mean(hi - lo)), "range factor": float(np.exp(q))}
        bands = mondrian(cal, col)
        c2, lo2, hi2 = apply_bands(bands, p)
        out[col].update({"bands": bands, "MAE banded": float(np.mean(np.abs(c2 - y))),
                         "coverage banded": float(np.mean((y >= lo2) & (y <= hi2))),
                         "mean width banded s": float(np.mean(hi2 - lo2))})
        if col == "eta_ml":                                   # the model's own (real-data widened) range
            ok = test.dropna(subset=["eta_ml_lo", "eta_ml_hi"])
            out[col]["own_range_coverage"] = float(np.mean((ok["actual_s"] >= ok["eta_ml_lo"]) &
                                                                          (ok["actual_s"] <= ok["eta_ml_hi"])))
    # by situation (test seeds)
    groups = {"all": np.ones(len(test), bool), "campus": (test["building"] == "campus").to_numpy(),
              "hospital": (test["building"] == "hospital").to_numpy(),
              "queue where they stand (>= 5)": (test["queue_here"] >= 5).to_numpy(),
              "more than 2 min to go": (test["actual_s"] > 120).to_numpy(),
              "fire scenario": test["scenario"].str.contains("fire").to_numpy()}
    by = []
    for g, m in groups.items():
        r = {"group": g, "n": int(m.sum())}
        for col in ("eta_ml", "eta_plan"):
            b, q = out[col]["bias_factor"], out[col]["log_margin"]
            p = np.maximum(test[col].to_numpy()[m], 1.0) * b
            y = test["actual_s"].to_numpy()[m]
            r[f"{col} MAE"] = float(np.mean(np.abs(p - y)))
            r[f"{col} coverage"] = float(np.mean((y >= p * np.exp(-q)) & (y <= p * np.exp(q))))
        by.append(r)
    out["by_group"] = by
    return out, test


def report(ev, test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from evac.final_report import BLUE, ORANGE, GREY, MUTED, _style
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    y = test["actual_s"].to_numpy()
    for a, col, c, lab in ((ax[0], "eta_ml", ORANGE, "ML crossing times (crowd as now)"),
                           (ax[1], "eta_plan", BLUE, "Planner bookings (crowd forecast)")):
        p = np.maximum(test[col].to_numpy(), 1.0) * ev[col]["bias_factor"]
        k = np.random.default_rng(0).choice(len(y), min(4000, len(y)), replace=False)
        a.scatter(y[k], p[k], s=3, color=c, alpha=0.4)
        lim = float(np.percentile(np.r_[y, p], 99))
        xs = np.linspace(1, lim, 50)
        a.plot(xs, xs, color=GREY, lw=1)
        f = ev[col]["range factor"]
        a.fill_between(xs, xs / f, xs * f, color=GREY, alpha=0.12, label="90% range shown on the sign")
        a.set(xlim=(0, lim), ylim=(0, lim), xlabel="Actual time to exit (s)", ylabel="Predicted (s)")
        a.set_title(f"{lab}\nMAE {ev[col]['MAE corrected']:.0f} s, range covers {ev[col]['coverage']:.0%}",
                    loc="left", fontsize=10, fontweight="bold")
        a.legend(fontsize=7, frameon=False, loc="upper left")
        _style(a, "both")
    B = ev["by_group"]
    x = np.arange(len(B))
    ax[2].bar(x - 0.2, [b["eta_ml MAE"] for b in B], 0.4, color=ORANGE, label="ML (crowd as now)")
    ax[2].bar(x + 0.2, [b["eta_plan MAE"] for b in B], 0.4, color=BLUE, label="Planner (crowd forecast)")
    ax[2].set_xticks(x, [b["group"].replace(" (>= 5)", "").replace("more than ", ">") for b in B], fontsize=7, rotation=20)
    ax[2].set_ylabel("Error (s), test evacuations", fontsize=9, color=MUTED)
    ax[2].set_title("Where each is better", loc="left", fontsize=10, fontweight="bold")
    ax[2].legend(fontsize=7, frameon=False)
    _style(ax[2])
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)
    plt.close(fig)

    ml, pl = ev["eta_ml"], ev["eta_plan"]
    qg = next(b for b in ev["by_group"] if b["group"].startswith("queue"))
    best = "eta_plan" if pl["MAE corrected"] <= ml["MAE corrected"] else "eta_ml"
    L = ["# Time-to-exit on signs: how good is the estimate? (Phase 2)", "",
         "Every 30 s during look-ahead evacuations (4 campus + 3 hospital scenarios, 20 seeds) we recorded, for "
         "everyone inside and on the move, two predictions of their remaining time, then compared them with when "
         "they really got out. Ranges are calibrated on seeds 0-9 and tested on seeds 10-19 (conformal prediction). "
         "Generated by `python -m evac.eta_study`.", "", "## Findings", "",
         f"1. **{ev['rows']:,} predictions** ({ev['test_rows']:,} on test evacuations).",
         f"2. **Raw estimates are optimistic**: the ML sum is off by {ml['mean error raw']:+.0f} s on average and the "
         f"planner's bookings by {pl['mean error raw']:+.0f} s (people take longer than predicted: queues still "
         f"forming, stragglers, non-followers). A single correction factor fixes most of it: x{ml['bias_factor']:.2f} "
         f"(ML), x{pl['bias_factor']:.2f} (planner).",
         f"3. **After correction**: average error {ml['MAE corrected']:.0f} s (ML) vs {pl['MAE corrected']:.0f} s "
         f"(planner). A 90% range is x/÷ {ml['range factor']:.2f} (ML) or x/÷ {pl['range factor']:.2f} (planner); "
         f"on the test evacuations it held the real time for {ml['coverage']:.0%} / {pl['coverage']:.0%} of people.",
         f"4. **Ranges that fit the situation**: short estimates are relatively less certain than long ones, so the "
         f"range is calibrated per band of predicted time (Mondrian conformal). ML: x/÷ "
         + ", ".join(f"{b['range_factor']:.2f}" for b in ml["bands"]) + " for estimates up to 30 s, 1, 2, 4 min and "
         f"longer; coverage {ml['coverage banded']:.0%}, average range width {ml['mean width banded s']:.0f} s "
         f"(one-size range: {ml['mean width s']:.0f} s).",
         f"5. The ML model's own range (built from crossing-level uncertainty, widened on real data) covered only "
         f"{ml['own_range_coverage']:.0%} of whole-route times: errors along a route add up, so "
         "route-level ranges must be calibrated at route level - which is what the sign now uses.",
         f"6. **Signs use the {'planner forecast' if best == 'eta_plan' else 'ML sum'}** (lower error) with its "
         "range for its band of predicted time (`evac/eta_calibration.json`), on the dashboard route and the kiosk "
         "signs. The planner's booking forecast is kept for analysis: it is worse exactly where it matters, at a "
         f"queue (error {qg['eta_plan MAE']:.0f} s vs {qg['eta_ml MAE']:.0f} s) - its 4-second slots over-simplify how a queue drains.", "",
         "![Time to exit](figures/fig31_eta.png)", "",
         "## By situation (test evacuations, after correction)", "",
         "| Situation | Predictions | ML error (s) | ML range covers | Planner error (s) | Planner range covers |",
         "| --- | --- | --- | --- | --- | --- |"]
    for b in ev["by_group"]:
        L.append(f"| {b['group']} | {b['n']:,} | {b['eta_ml MAE']:.0f} | {b['eta_ml coverage']:.0%} | "
                 f"{b['eta_plan MAE']:.0f} | {b['eta_plan coverage']:.0%} |")
    L += ["", "Honest limits: these are simulated people, so the check is that the estimate is consistent with the "
          "simulator, not with a real building. The real-data check of the crossing times is in `docs/ml_v3.md`.", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    with open(CAL, "w", encoding="utf-8") as f:
        json.dump({"best": best, "eta_ml": {k: ml[k] for k in ("bias_factor", "range factor", "coverage", "bands",
                                                                  "coverage banded")},
                   "eta_plan": {k: pl[k] for k in ("bias_factor", "range factor", "coverage", "bands", "coverage banded")},
                   "source": "evac/eta_study.py (look-ahead runs, conformal split by seed)"}, f, indent=1)
    print("\n".join(L[5:13]))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    D = collect(a.workers)
    ev, test = evaluate(D)
    report(ev, test)


if __name__ == "__main__":
    main()
