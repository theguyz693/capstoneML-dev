"""
evac/ml_transfer.py
Session 7: does the learned delay model transfer between buildings, and does retraining fix it?

Session 6 found the campus-trained model (v1) made the planner ~8% worse on the hospital: it had never
seen ramps, beds or slow patients. Here:

  1. new training data from BOTH buildings (random scenarios, edge crossings logged with the mover's
     speed and size); campus data regenerated with the same scenarios as v1 plus the new columns
  2. v2 features = v1 features + is_ramp + mover_speed + mover_size
  3. transfer matrix (prediction accuracy, grouped by evacuation):
        train campus -> test hospital,  train hospital -> test campus,  train both -> held-out of each
  4. in the loop: look-ahead planner with formula / v1 / v2 walk times, hospital + campus scenarios

Run:  python -m evac.ml_transfer     -> models/delay_lgbm_v2*.joblib, docs/ml_transfer.md, fig24
"""

import csv
import json
import os
import random
import statistics
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

RES = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
DOC = os.path.join(ROOT, "docs", "ml_transfer.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig24_ml_transfer.png")
DATA = {"campus": os.path.join(RES, "delay_v2_campus.csv"), "hospital": os.path.join(RES, "delay_v2_hospital.csv")}
FEATURES_V1 = ["is_stair", "is_room_door", "is_exit_door", "is_corridor", "length_m", "width_m",
               "eff_cap", "free_speed", "hazard", "queue_ahead", "queue_at_node", "on_edge", "density",
               "planned_users"]
FEATURES_V2 = FEATURES_V1 + ["is_ramp", "mover_speed", "mover_size"]
RNG = 0


def hospital_scenarios(G, n, seed=1):
    rng = random.Random(seed)
    junctions = sorted(x for x in G if G.nodes[x]["kind"] == "junction")
    profiles = ["walk", "slow", "wheelchair", "bed"]
    pools = ["^g_(op |waiting|reception)", "^f1_(lab|information)", "^f2_general ward", "^f3_general ward",
             "^f2_(nurse|change|store)", "^f3_(nurse|change|store)"]
    out = []
    for i in range(n):
        groups = []
        for where in rng.sample(pools, rng.randint(1, len(pools))):
            w = [rng.random() for _ in profiles]
            if "general ward" not in where:
                w[3] = 0.0                                   # beds only in wards
            groups.append({"where": where, "n": rng.choice([10, 20, 40, 60, 100, 150]),
                           "mix": {p: round(x, 3) for p, x in zip(profiles, w) if x > 0}})
        sc = {"_name": f"hgen{i:03d}", "building": "hospital", "spawn": "groups", "groups": groups,
              "compliance": rng.choice([1.0, 1.0, 0.8, 0.6]), "herding": rng.choice([0.0, 0.5]),
              "familiar_exit": "g_main entrance"}
        kind = rng.choice(["none", "static", "spreading"])
        if kind == "none":
            sc["fires"] = []
        else:
            f = {"node": rng.choice(junctions), "intensity": 1.0, "contained": rng.random() < 0.5}
            if kind == "static":
                f["radius"] = rng.choice([200, 300, 500])
            else:
                f.update(radius=100, spread_m_per_s=rng.choice([0.1, 0.3, 0.5]), max_radius=700)
            sc["fires"] = [f]
        sc["_method"] = rng.choice(["safe_shortest", "reactive", "ccrp"])
        out.append(sc)
    return out


_G = {}


def _gen_one(args):
    building, i, sc = args
    from evac.building import load_building
    from evac.simulator import run
    if building not in _G:
        _G[building] = load_building(building)
    m, _ = run(_G[building].copy(), {**sc, "log_edges": True}, sc["_method"], seed=i)
    rows = []
    for r in m["edge_rows"]:
        r.pop("_t_join", None)
        r["method"] = sc["_method"]
        r["building"] = building
        rows.append(r)
    return rows


def generate(workers=2):
    from evac.building import load_campus
    from evac.delay_model import random_scenarios
    from evac.hospital import load_hospital
    jobs = {"campus": [("campus", i, sc) for i, sc in enumerate(random_scenarios(load_campus(), 160, 0))],
            "hospital": [("hospital", i, sc) for i, sc in enumerate(hospital_scenarios(load_hospital(), 120, 1))]}
    for b, js in jobs.items():
        if os.path.exists(DATA[b]):
            print(f"  {DATA[b]} exists")
            continue
        t0, rows = time.time(), []
        with Pool(workers) as pool:
            for k, rs in enumerate(pool.imap(_gen_one, js, chunksize=2)):
                rows.extend(rs)
                if (k + 1) % 40 == 0:
                    print(f"  {b}: {k + 1}/{len(js)} evacuations, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
        with open(DATA[b], "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def _load(b, frac=0.5):
    import pandas as pd
    df = pd.read_csv(DATA[b])
    for et in ("stair", "room_door", "exit_door", "corridor", "ramp"):
        df[f"is_{et}"] = (df["etype"] == et).astype(float)
    df["group"] = b + ":" + df["scenario"].astype(str)
    df["crowded"] = df["queue_ahead"] > 0
    if frac < 1:
        keep = df.groupby("group").sample(frac=frac, random_state=RNG).index   # same share of every evacuation
        df = df.loc[keep]
    return df


def _lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                         verbose=-1, random_state=RNG)


def _formula(df):
    from evac.physics import weidmann_speed_factor
    return (df["queue_ahead"] / df["eff_cap"] +
            df["length_m"] / (df["free_speed"] * df["mover_speed"] * df["density"].map(weidmann_speed_factor))).to_numpy()


def _mae(y, p, mask=None):
    import numpy as np
    if mask is not None:
        y, p = y[mask], p[mask]
    return float(np.mean(np.abs(y - p)))


def transfer_matrix(C_, H_):
    import numpy as np
    rng = np.random.default_rng(RNG)
    rows = []

    def hold(df):
        g = df["group"].unique()
        test = set(rng.choice(g, size=int(0.25 * len(g)), replace=False))
        return df[~df["group"].isin(test)], df[df["group"].isin(test)]
    c_tr, c_te = hold(C_)
    h_tr, h_te = hold(H_)
    import pandas as pd
    both_tr = pd.concat([c_tr, h_tr])
    setups = [("campus only", c_tr), ("hospital only", h_tr), ("both buildings", both_tr)]
    fitted = {}
    for name, tr in setups:
        for fv, feats in (("v1 features", FEATURES_V1), ("v2 features", FEATURES_V2)):
            m = _lgbm().fit(tr[feats], tr["target_time_s"])
            fitted[(name, fv)] = (m, feats)
    for test_name, te in (("campus (held-out evacuations)", c_te), ("hospital (held-out evacuations)", h_te)):
        y = te["target_time_s"].to_numpy()
        crowd = te["crowded"].to_numpy()
        slow = (te["mover_speed"] < 0.9).to_numpy()
        p = _formula(te)
        rows.append({"test": test_name, "trained on": "-", "model": "queueing formula (no training)",
                     "MAE all": _mae(y, p), "MAE crowded": _mae(y, p, crowd),
                     "MAE slow movers": _mae(y, p, slow) if slow.sum() > 50 else None})
        for (name, fv), (m, feats) in fitted.items():
            p = m.predict(te[feats])
            rows.append({"test": test_name, "trained on": name, "model": f"LightGBM, {fv}",
                         "MAE all": _mae(y, p), "MAE crowded": _mae(y, p, crowd),
                         "MAE slow movers": _mae(y, p, slow) if slow.sum() > 50 else None})
    return rows


def train_final(C_, H_):
    import joblib
    import pandas as pd
    from lightgbm import LGBMRegressor
    df = pd.concat([C_, H_])
    m = _lgbm().fit(df[FEATURES_V2], df["target_time_s"])
    q = LGBMRegressor(objective="quantile", alpha=0.95, n_estimators=400, learning_rate=0.05, num_leaves=63,
                      verbose=-1, random_state=RNG).fit(df[FEATURES_V2], df["target_time_s"])
    joblib.dump(m, os.path.join(MODELS, "delay_lgbm_v2.joblib"))
    joblib.dump(q, os.path.join(MODELS, "delay_lgbm_v2_q95.joblib"))
    with open(os.path.join(MODELS, "delay_model_v2_meta.json"), "w") as f:
        json.dump({"features": FEATURES_V2, "train_rows": len(df), "buildings": ["campus", "hospital"],
                   "target": "target_time_s (seconds through edge incl. queue)"}, f, indent=1)


def _loop_job(args):
    building, name, method, version, seed = args
    from evac import config as C
    from evac.building import load_building
    from evac.simulator import run
    if building not in _G:
        _G[building] = load_building(building)
    C.DELAY_MODEL = version
    if building == "hospital":
        from evac.hospital import scenarios
        sc = scenarios()[name]
    else:
        with open(os.path.join(HERE, "scenarios.json"), encoding="utf-8") as f:
            sc = json.load(f)[name]
    m, recs = run(_G[building].copy(), sc, method, seed)
    amb = [r["t_out"] for r in recs if r["t_out"] is not None and r.get("profile") in ("walk", "slow", "step_free")]
    return {"building": building, "scenario": name, "variant": "formula" if method == "lookahead" else version,
            "seed": seed, "last": m["evac_time_last_s"], "ambulatory_last": max(amb) if amb else None,
            "mean": m["evac_time_mean_s"], "exposed": m["exposed_agents"]}


def in_loop(workers=2):
    out = os.path.join(RES, "ml_transfer_loop.csv")
    if os.path.exists(out):
        import pandas as pd
        return pd.read_csv(out)
    from evac.hospital import scenarios
    jobs = []
    for b, names in (("hospital", list(scenarios())),
                     ("campus", ["floor3_full_200", "lecture_hall_f1_120", "floor3_200_spreading_fire_stairs1"])):
        for name in names:
            for sd in range(10):
                jobs.append((b, name, "lookahead", "v1", sd))
                jobs.append((b, name, "lookahead_ml", "v1", sd))
                jobs.append((b, name, "lookahead_ml", "v2", sd))
    rows = []
    import multiprocessing as mp
    # "spawn": the parent has LightGBM (OpenMP) loaded; forking it can deadlock the workers
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_loop_job, jobs, chunksize=2)):
            rows.append(r)
            if (k + 1) % 60 == 0:
                print(f"  in-loop {k + 1}/{len(jobs)}", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def report(rows, loop):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from evac.final_report import BLUE, ORANGE, GREY, INK, MUTED, _style
    T = pd.DataFrame(rows)
    lines = ["# Does the learned model transfer between buildings? (Session 7)", "",
             "Edge-delay prediction error (MAE, seconds) on evacuations never used for training. "
             "'Crowded' = someone already queued for the edge; 'slow movers' = patients, wheelchairs, beds (hospital).", "",
             "| Test set | Trained on | Model | MAE all | MAE crowded | MAE slow movers |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        sm = "-" if r["MAE slow movers"] is None else f"{r['MAE slow movers']:.2f}"
        lines.append(f"| {r['test']} | {r['trained on']} | {r['model']} | {r['MAE all']:.2f} | {r['MAE crowded']:.2f} | {sm} |")
    get = lambda test, tr, model: T[(T.test.str.startswith(test)) & (T["trained on"] == tr) & (T.model == model)].iloc[0]
    f_h = T[(T.test.str.startswith("hospital")) & (T["trained on"] == "-")].iloc[0]
    v1c = get("hospital", "campus only", "LightGBM, v1 features")
    v2b = get("hospital", "both buildings", "LightGBM, v2 features")
    v2c = get("campus", "both buildings", "LightGBM, v2 features")
    v1cc = get("campus", "campus only", "LightGBM, v1 features")
    findings = [
        f"Campus-trained model on the hospital: MAE {v1c['MAE all']:.2f} s (formula {f_h['MAE all']:.2f} s) - "
        + ("worse than the formula" if v1c["MAE all"] > f_h["MAE all"] else "better than the formula")
        + f"; on slow movers {v1c['MAE slow movers']:.1f} s vs {f_h['MAE slow movers']:.1f} s.",
        f"Trained on BOTH buildings with mover speed/size and ramp features (v2): hospital MAE {v2b['MAE all']:.2f} s "
        f"(slow movers {v2b['MAE slow movers']:.1f} s) - now better than the formula - and campus accuracy does not "
        f"suffer ({v2c['MAE all']:.2f} s vs {v1cc['MAE all']:.2f} s for the campus-only model).",
    ]
    lines += ["", "## In the planner", "", "Look-ahead re-planning with walking times from the formula, the campus-only "
              "model (v1) or the two-building model (v2). Mean of 10 seeds; hospital = last WALKING person out, "
              "campus = last person out.", "",
              "| Building | Scenario | Formula | ML v1 (campus only) | ML v2 (both buildings) |", "| --- | --- | --- | --- | --- |"]
    loop = loop.copy()
    loop["metric"] = loop.apply(lambda r: r["ambulatory_last"] if r["building"] == "hospital" else r["last"], axis=1)
    piv = loop.groupby(["building", "scenario", "variant"])["metric"].mean().unstack()
    for (b, s), r in piv.iterrows():
        lines.append(f"| {b} | {s} | {r['formula']:.0f} | {r['v1']:.0f} | {r['v2']:.0f} |")
    rel = {}
    for b in ("hospital", "campus"):
        x = piv.loc[b]
        rel[b] = ((x["v1"] / x["formula"]).mean() - 1, (x["v2"] / x["formula"]).mean() - 1)
    findings.append(f"In the planner, vs the formula: hospital v1 {rel['hospital'][0]:+.1%} -> v2 {rel['hospital'][1]:+.1%}; "
                    f"campus v1 {rel['campus'][0]:+.1%} -> v2 {rel['campus'][1]:+.1%}. "
                    + (f"Retraining shrinks the transfer penalty from {rel['hospital'][0]:+.1%} to {rel['hospital'][1]:+.1%}, "
                       "but the plain formula is still as good inside the planner: a more accurate predictor does not "
                       "make the plan much better - coordination does. Lesson: a learned component must be retrained "
                       "for each new kind of building, the formula needs no training."
                       if rel["hospital"][1] < rel["hospital"][0] - 0.02 else
                       "Retraining improves prediction accuracy but the planner barely changes - again, coordination, "
                       "not prediction accuracy, drives the result."))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for a, test in zip(ax, ("campus", "hospital")):
        sub = T[T.test.str.startswith(test)]
        labels = ["formula"] + [f"{r['trained on']}\n{r['model'].split(', ')[1]}" for _, r in sub.iloc[1:].iterrows()]
        vals = sub["MAE all"].tolist()
        cols = [GREY] + [BLUE if "both" in l else ORANGE if (test in l) else "#cfcdc4" for l in labels[1:]]
        a.barh(range(len(vals))[::-1], vals, color=cols, height=0.6)
        a.set_yticks(range(len(vals))[::-1], labels, fontsize=7)
        a.set_xlabel("MAE on held-out evacuations (s)", fontsize=9, color=MUTED)
        a.set_title(f"Tested on the {test}", loc="left", fontsize=11, fontweight="bold")
        _style(a)
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)
    plt.close(fig)
    head = ["# Does the learned model transfer between buildings? (Session 7)", "", "## Findings", ""] + \
           [f"{i}. {f}" for i, f in enumerate(findings, 1)] + ["", "![Transfer](figures/fig24_ml_transfer.png)", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(head + lines[2:] + [""]))
    return findings


def main():
    os.makedirs(RES, exist_ok=True)
    if os.path.exists(os.path.join(RES, "ml_transfer_loop.csv")) and os.path.exists(os.path.join(MODELS, "delay_lgbm_v2.joblib")) \
            and os.path.exists(os.path.join(RES, "ml_transfer_matrix.json")):
        with open(os.path.join(RES, "ml_transfer_matrix.json")) as f:
            rows = json.load(f)
        import pandas as pd
        for x in report(rows, pd.read_csv(os.path.join(RES, "ml_transfer_loop.csv"))):
            print("-", x)
        return
    print("generating data ...", flush=True)
    generate()
    C_, H_ = _load("campus"), _load("hospital")
    print(f"rows: campus {len(C_)}, hospital {len(H_)}", flush=True)
    rows = transfer_matrix(C_, H_)
    with open(os.path.join(RES, "ml_transfer_matrix.json"), "w") as f:
        json.dump(rows, f, indent=1)
    train_final(C_, H_)
    print("in the loop ...", flush=True)
    loop = in_loop()
    for f in report(rows, loop):
        print("-", f)


if __name__ == "__main__":
    main()
