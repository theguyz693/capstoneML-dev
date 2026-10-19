"""
evac/ml_v3.py
Improving ML, steps 3-6: ONE learned model of "how long does it take to get through this corridor / door /
stair right now", trained on the CALIBRATED simulator and tested on REAL crowd data.

Why a new model (mentor's critique of the old ones):
  v1  Random Forest (train_model.py) learned a hand-written cost formula from random numbers - it can only
      copy the formula. Dropped from all claims.
  v2  LightGBM on simulator crossings (Session 7) - learned from a simulator that had never been compared
      with real people (its doors passed about half the measured flow).
  v3  (this file) the same kind of model, but
        - the simulator is first calibrated on Juelich experiments (evac/calibration.py) and smoke now
          slows people down (Frantzich & Nilsson 2003),
        - the training data covers both buildings AND random simple geometries (corridors and doors of
          many widths and lengths), so the model learns crowd physics rather than one floor plan,
        - it is tested on REAL crossings measured in the Juelich experiments, not only on simulated ones,
        - it gives a range (5 % / 95 % quantiles), used for time-to-exit estimates on signs.

Dataset rows: one per person crossing one edge. Features = what the system can see when the person reaches
the edge (edge type, length, width, capacity, free speed, smoke level, people queued ahead, people at the
node, people on the edge, density, people whose current route uses the edge, the person's speed and size).
Target = seconds from arriving at the edge to leaving it (queue wait + walk).

Run:  python -m evac.ml_v3            (all steps; cached in results/)
      python -m evac.ml_v3 --generate (only make the dataset)
-> models/delay_lgbm_v3*.joblib, results/delay_v3_*.csv, docs/ml_v3.md, docs/figures/fig29_ml_v3.png
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

RES = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
DOC = os.path.join(ROOT, "docs", "ml_v3.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig29_ml_v3.png")
DATA = {b: os.path.join(RES, f"delay_v3_{b}.csv") for b in ("campus", "hospital", "synthetic")}
REAL = os.path.join(RES, "delay_v3_real_juelich.csv")
FEATURES = ["is_stair", "is_room_door", "is_exit_door", "is_corridor", "is_ramp", "length_m", "width_m",
            "eff_cap", "free_speed", "hazard", "queue_ahead", "queue_at_node", "on_edge", "density",
            "planned_users", "mover_speed", "mover_size"]
ETYPES = ("stair", "room_door", "exit_door", "corridor", "ramp")
# v3 model inputs = FEATURES + physics-derived inputs (evac/learned_planner.add_physics); target = log(time / formula)
SEED = 0
METHODS = ["safe_shortest", "reactive", "ccrp", "lookahead", "nearest"]


# ================================================================== step 3: dataset from the calibrated simulator
def synthetic_scenarios(n, seed=7):
    """Random simple geometries: two rooms, a corridor network, two exits, every width and length random."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        geo = {"L_room": rng.uniform(2, 12), "L_cor": rng.uniform(3, 25), "L_cor2": rng.uniform(3, 25),
               "L_exit": rng.uniform(2, 10),
               "w_door_a": rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6]),
               "w_door_b": rng.choice([0.7, 0.9, 1.0, 1.2, 1.6, 2.0]),
               "w_cor": rng.uniform(1.0, 5.5), "w_cor2": rng.uniform(1.0, 5.5),
               "w_exit_e": rng.choice([0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8, 2.4]),
               "w_exit_w": rng.choice([0.7, 0.9, 1.0, 1.2, 1.8, 2.4])}
        sc = {"_name": f"syn{i:03d}", "_geo": geo, "spawn": "building",
              "n_agents": rng.choice([20, 40, 80, 120, 200, 300]),
              "pre_movement_max_s": rng.choice([0.0, 10.0, 30.0, 60.0]),
              "compliance": rng.choice([1.0, 1.0, 0.7]), "herding": rng.choice([0.0, 0.5]),
              "familiar_exit": "exit east", "step_free_frac": 0.0}
        sc["groups"] = None
        kind = rng.choice(["none", "none", "static", "spreading"])
        if kind == "none":
            sc["fires"] = []
        else:
            f = {"node": rng.choice(["j1", "j2", "j3"]), "intensity": rng.choice([0.6, 0.8, 1.0])}
            if kind == "static":
                f["radius"] = rng.choice([150, 300, 500])
            else:
                f.update(radius=50, spread_m_per_s=rng.choice([0.2, 0.5]), max_radius=700)
            sc["fires"] = [f]
        mode = rng.choice(["walk", "walk", "spread", "groups"])
        if mode == "spread":      # wider spread of walking speeds
            sc["speed_spread"] = 0.35
        elif mode == "groups":    # patients, wheelchairs and beds too, like the hospital
            n = sc["n_agents"]
            mix = {"walk": rng.random(), "slow": rng.random(), "wheelchair": rng.random() * 0.5, "bed": rng.random() * 0.3}
            sc.update(spawn="groups", groups=[{"where": "^room a$", "n": max(1, n // 2), "mix": mix},
                                              {"where": "^room b$", "n": max(1, n - n // 2), "mix": mix}])
        sc["_method"] = rng.choice(METHODS)
        out.append(sc)
    return out


def synthetic_graph(geo):
    from evac import config as C
    from evac.building import build_graph, set_edge_width
    U = 1.0 / C.METERS_PER_UNIT
    # j1 -- j2 -- exit east;  j1 -- j3 -- exit west;  room a -> j1, room b -> j2
    nodes = {"j1": (0, 0, 0), "j2": (geo["L_cor"] * U, 0, 0), "j3": (-geo["L_cor2"] * U, 0, 0),
             "room a": (0, geo["L_room"] * U, 0), "room b": (geo["L_cor"] * U, geo["L_room"] * U, 0),
             "exit east": ((geo["L_cor"] + geo["L_exit"]) * U, 0, 0),
             "exit west": (-(geo["L_cor2"] + geo["L_exit"]) * U, 0, 0)}
    edges = [("room a", "j1"), ("room b", "j2"), ("j1", "j2"), ("j1", "j3"), ("j2", "exit east"),
             ("j3", "exit west")]
    G = build_graph(nodes, edges, kinds={"j1": "junction", "j2": "junction", "j3": "junction"})
    set_edge_width(G.edges["room a", "j1"], geo["w_door_a"])
    set_edge_width(G.edges["room b", "j2"], geo["w_door_b"])
    set_edge_width(G.edges["j1", "j2"], round(geo["w_cor"], 2))
    set_edge_width(G.edges["j1", "j3"], round(geo["w_cor2"], 2))
    set_edge_width(G.edges["j2", "exit east"], geo["w_exit_e"])
    set_edge_width(G.edges["j3", "exit west"], geo["w_exit_w"])
    return G


_G = {}


def _gen_one(args):
    building, i, sc = args
    from evac.simulator import run
    if building == "synthetic":
        G = synthetic_graph(sc["_geo"])
        sc = {k: v for k, v in sc.items() if k != "_geo" and not (k == "groups" and v is None)}
    else:
        from evac.building import load_building
        if building not in _G:
            _G[building] = load_building(building)
        G = _G[building].copy()
    m, _ = run(G, {**sc, "log_edges": True}, sc["_method"], seed=1000 + i)
    rows = []
    for r in m["edge_rows"]:
        r.pop("_t_join", None)
        r["method"] = sc["_method"]
        r["building"] = building
        r["scenario"] = f"{building}:{sc['_name']}"
        rows.append(r)
    return rows


def generate(workers=2):
    from multiprocessing import Pool
    from evac.building import load_building
    from evac.delay_model import random_scenarios
    from evac.ml_transfer import hospital_scenarios
    rng = random.Random(11)
    camp = random_scenarios(load_building("campus"), 200, 0)
    hosp = hospital_scenarios(load_building("hospital"), 150, 1)
    for sc in camp + hosp:          # the planners too, so the data contains the states they create
        sc["_method"] = rng.choice(METHODS)
    jobs = {"campus": [("campus", i, sc) for i, sc in enumerate(camp)],
            "hospital": [("hospital", i, sc) for i, sc in enumerate(hosp)],
            "synthetic": [("synthetic", i, sc) for i, sc in enumerate(synthetic_scenarios(600))]}
    for b, js in jobs.items():
        if os.path.exists(DATA[b]):
            print(f"  {DATA[b]} exists", flush=True)
            continue
        t0, rows = time.time(), []
        with Pool(workers) as pool:
            for k, rs in enumerate(pool.imap_unordered(_gen_one, js, chunksize=2)):
                rows.extend(rs)
                if (k + 1) % 50 == 0:
                    print(f"  {b}: {k + 1}/{len(js)} evacuations, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
        cols = list(rows[0].keys())
        with open(DATA[b], "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


# ================================================================== step 5 data: REAL crossings (Juelich)
def _edge_features(etype, L, w, params):
    """Static features of a test edge under a parameter set ('juelich' or 'sfpe')."""
    from evac import config as C
    from evac import physics
    C.use_params(params)
    dw = physics.density_width(etype, w)
    v0 = C.FREE_WALK_SPEED
    eff = min(C.SPECIFIC_FLOW[etype] * w, physics.sustainable_flow(dw, v0))
    jam = max(2, int(C.JAM_DENSITY * L * dw))
    return {"free_speed": v0, "eff_cap": eff, "dens_width": dw, "jam_cap": jam}


REAL_SETUPS = [
    # file, unit, fps, etype, width, axis, join line, leave line, kind of test
    *[(f"uo-{e}-180-180", "cm", 16, "corridor", 1.8, "y", 2.0, -2.0, "corridor 1.8 m") for e in
      ("050", "060", "070", "100", "145", "180")],
    *[(f"traj_UNI_CORR_500_{k}", "m", 25, "corridor", 5.0, "x", 2.0, -2.0, "corridor 5 m") for k in ("01", "03", "08")],
    ("uo-180-180-070", "cm", 16, "exit_door", 0.70, "y", -1.0, -4.0, "bottleneck"),
    ("uo-180-180-095", "cm", 16, "exit_door", 0.95, "y", -1.0, -4.0, "bottleneck"),
    ("uo-180-180-120", "cm", 16, "exit_door", 1.20, "y", -1.0, -4.0, "bottleneck"),
    ("040_c_56_h-", "m", 25, "exit_door", 0.50, "y", 2.5, -0.5, "bottleneck"),
]


def real_rows():
    """Each person who walks from the join line to the leave line becomes one row in the SAME format as the
    simulator rows (people ahead in the section, density, people still to come), target = measured time."""
    import numpy as np
    from evac.calibration import load
    if os.path.exists(REAL):
        import pandas as pd
        return pd.read_csv(REAL)
    out = []
    for name, unit, fps, et, w, ax, j_line, l_line, kind in REAL_SETUPS:
        d = load(name, unit, fps)
        L = abs(j_line - l_line)
        tj, tl = {}, {}
        for pid, g in d.groupby("id"):
            g = g.sort_values("frame")
            pos = g[ax].to_numpy()
            fr = g["frame"].to_numpy()
            a = np.where(pos < j_line)[0]
            b = np.where(pos < l_line)[0]
            if len(a) and len(b) and pos[0] > j_line:
                tj[pid], tl[pid] = fr[a[0]], fr[b[0]]
        ids = sorted(tj, key=lambda p: tj[p])
        n_all = d["id"].nunique()
        stat = {p: _edge_features(et, L, w, p) for p in ("juelich", "sfpe")}
        for pid in ids:
            f0 = tj[pid]
            ahead = sum(1 for q in ids if q != pid and tj[q] <= f0 < tl[q])
            done = sum(1 for q in ids if tl[q] <= f0)
            row = {"run": name, "test": kind, "etype": et, "length_m": L, "width_m": w, "hazard": 0.0,
                   "mover_speed": 1.0, "mover_size": 1.0, "people_ahead": ahead,
                   "planned_users": max(0, n_all - done - 1), "target_time_s": (tl[pid] - f0) / fps}
            for p, s in stat.items():
                on = min(ahead, s["jam_cap"])
                row[f"{p}:on_edge"] = on
                row[f"{p}:queue_ahead"] = ahead - on
                row[f"{p}:density"] = on / (L * s["dens_width"])
                row[f"{p}:eff_cap"] = s["eff_cap"]
                row[f"{p}:free_speed"] = s["free_speed"]
            out.append(row)
    from evac import config as C
    C.use_params("juelich")
    import pandas as pd
    df = pd.DataFrame(out)
    df.to_csv(REAL, index=False)
    return df


def real_X(df, params):
    import pandas as pd
    X = pd.DataFrame(index=df.index)
    for et in ETYPES:
        X[f"is_{et}"] = (df["etype"] == et).astype(float)
    for c in ("length_m", "width_m", "hazard", "planned_users", "mover_speed", "mover_size"):
        X[c] = df[c]
    for c in ("eff_cap", "free_speed", "on_edge", "queue_ahead", "density"):
        X[c] = df[f"{params}:{c}"]
    X["queue_at_node"] = X["queue_ahead"]
    return X


# ================================================================== step 4: train + evaluate
def load_sim():
    import pandas as pd
    parts = []
    for b, p in DATA.items():
        df = pd.read_csv(p)
        df["building"] = b
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    for et in ETYPES:
        df[f"is_{et}"] = (df["etype"] == et).astype(float)
    df["group"] = df["scenario"].astype(str)
    df["crowded"] = df["queue_ahead"] > 0
    df["smoky"] = df["hazard"] >= 0.3
    return with_physics(df)


def formula(X, params="juelich"):
    """The textbook queueing formula the planner used: people queued ahead / capacity + length / speed at the
    current density (and smoke slowdown). params selects the speed-density curve."""
    from evac import config as C
    from evac.learned_planner import formula_time
    C.use_params(params)
    out = formula_time(X)
    C.use_params("juelich")
    return out


def lgbm(objective="regression", alpha=None):
    from lightgbm import LGBMRegressor
    kw = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=30, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=SEED, n_jobs=2)
    if objective == "quantile":
        kw.update(objective="quantile", alpha=alpha, n_estimators=400)
    return LGBMRegressor(**kw)


def _mae(y, p):
    import numpy as np
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def _hyb_fit(D, objective="regression", alpha=None, feats=None):
    """Hybrid fit: target = log(measured time / formula time), inputs = raw + physics features."""
    import numpy as np
    from evac.learned_planner import V3_EXTRA
    feats = feats or FEATURES + V3_EXTRA
    f = np.maximum(D["t_formula"].to_numpy(), 0.3)
    r = np.log(np.maximum(D["target_time_s"].to_numpy(), 0.1) / f)
    return lgbm(objective, alpha).fit(D[feats], r)


def _hyb_pred(m, D):
    import numpy as np
    return np.maximum(D["t_formula"].to_numpy(), 0.3) * np.exp(m.predict(D[list(m.feature_name_)]))


def with_physics(df):
    from evac.learned_planner import add_physics
    from evac import config as C
    C.use_params("juelich")
    return add_physics(df)


def cross_validate(df, folds=5):
    """Whole evacuations are held out (no rows of a test evacuation are seen in training)."""
    import numpy as np
    from sklearn.model_selection import GroupKFold
    y = df["target_time_s"].to_numpy()
    p_ml, p_lo, p_hi = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(df, y, df["group"])):
        A, B = df.iloc[tr], df.iloc[te]
        p_ml[te] = _hyb_pred(_hyb_fit(A), B)
        p_lo[te] = _hyb_pred(_hyb_fit(A, "quantile", 0.05), B)
        p_hi[te] = _hyb_pred(_hyb_fit(A, "quantile", 0.95), B)
        print(f"  fold {k + 1}/{folds}", flush=True)
    p_f = df["t_formula"].to_numpy()
    rows = []
    subsets = [("all crossings", np.ones(len(df), bool)), ("someone queued ahead", df["crowded"].to_numpy()),
               ("in smoke (hazard >= 0.3)", df["smoky"].to_numpy()),
               ("slow movers (patients, wheelchairs, beds)", (df["mover_speed"] < 0.8).to_numpy())]
    for et in ("room_door", "exit_door", "corridor", "stair", "ramp"):
        subsets.append((f"edge type: {et}", (df["etype"] == et).to_numpy()))
    for b in ("campus", "hospital", "synthetic"):
        subsets.append((f"building: {b}", (df["building"] == b).to_numpy()))
    for name, m in subsets:
        if m.sum() < 50:
            continue
        cover = float(np.mean((y[m] >= p_lo[m]) & (y[m] <= p_hi[m])))
        rows.append({"subset": name, "rows": int(m.sum()), "mean_time_s": float(y[m].mean()),
                     "MAE formula": _mae(y[m], p_f[m]), "MAE ML": _mae(y[m], p_ml[m]),
                     "90% range covers": cover})
    return rows


def ablation(df):
    """Which design choice makes the model work on REAL data? One grouped 75/25 split of the simulated data;
    every variant is also scored on the real Juelich crossings."""
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit
    from evac.learned_planner import V3_EXTRA
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(df, groups=df["group"]))
    A, B = df.iloc[tr], df.iloc[te]
    R = real_rows()
    XR = with_physics(real_X(R, "juelich"))
    yr = R["target_time_s"].to_numpy()
    out = []

    def score(name, pb, pr):
        r = {"variant": name, "MAE simulated (held out)": _mae(B["target_time_s"], pb), "MAE real (all)": _mae(yr, pr)}
        for t in sorted(R["test"].unique()):
            k = (R["test"] == t).to_numpy()
            r[f"MAE real: {t}"] = _mae(yr[k], pr[k])
        out.append(r)
    score("physics formula only (calibrated)", B["t_formula"], XR["t_formula"])
    m = lgbm().fit(A[FEATURES], A["target_time_s"])
    score("pure ML, raw features (like v2)", m.predict(B[FEATURES]), m.predict(XR[FEATURES]))
    feats = FEATURES + V3_EXTRA
    m = lgbm().fit(A[feats], A["target_time_s"])
    score("pure ML + physics features", m.predict(B[feats]), m.predict(XR[feats]))
    m = _hyb_fit(A)
    score("hybrid: physics x learned correction (v3)", _hyb_pred(m, B), _hyb_pred(m, XR))
    return out


def building_transfer(df):
    """Train without one building, test on it: does the model work on a floor plan it has never seen?"""
    rows = []
    for held in ("hospital", "campus"):
        tr = df[df["building"] != held]
        te = df[df["building"] == held]
        m = _hyb_fit(tr)
        rows.append({"test building (never seen)": held, "trained on": " + ".join(sorted(set(tr["building"]))),
                     "MAE formula": _mae(te["target_time_s"], te["t_formula"]),
                     "MAE ML": _mae(te["target_time_s"], _hyb_pred(m, te))})
        print(f"  transfer: {rows[-1]}", flush=True)
    return rows


def train_final(df):
    import joblib
    from evac.learned_planner import V3_EXTRA
    os.makedirs(MODELS, exist_ok=True)
    models = {"": _hyb_fit(df), "_q05": _hyb_fit(df, "quantile", 0.05), "_q95": _hyb_fit(df, "quantile", 0.95)}
    for suf, m in models.items():
        joblib.dump(m, os.path.join(MODELS, f"delay_lgbm_v3{suf}.joblib"))
    with open(os.path.join(MODELS, "delay_model_v3_meta.json"), "w") as f:
        json.dump({"features": FEATURES + V3_EXTRA, "train_rows": int(len(df)),
                   "rows_by_source": {k: int(v) for k, v in df["building"].value_counts().items()},
                   "simulator": "calibrated on Juelich experiments (evac/calibrated_params.json) + smoke slowdown",
                   "model": "hybrid: time = formula_time x exp(LightGBM(x)); evac/learned_planner.hybrid_predict",
                   "target": "log(target_time_s / formula_time); target_time_s = seconds from reaching the edge "
                             "to leaving it (queue + walk)",
                   "quantiles": ["q05", "q95"]}, f, indent=1)
    return models


def real_test(models, df_sim):
    """Step 5: the REAL crossings. Compare: formula with handbook values, formula calibrated, old model v2
    (trained on the uncalibrated simulator), new model v3."""
    import joblib
    import numpy as np
    R = real_rows()
    y = R["target_time_s"].to_numpy()
    Xj, Xs = real_X(R, "juelich"), real_X(R, "sfpe")
    preds = {"formula, handbook values": formula(Xs, "sfpe"),
             "formula, calibrated": formula(Xj, "juelich")}
    v2p = os.path.join(MODELS, "delay_lgbm_v2.joblib")
    if os.path.exists(v2p):
        v2 = joblib.load(v2p)
        cols = list(v2.feature_name_)
        Xs2 = Xs.copy()
        Xs2["density"] = Xs2["on_edge"] / (R["length_m"] * R["width_m"])     # v2 measured density over the door width
        preds["ML v2 (uncalibrated simulator)"] = v2.predict(Xs2[cols])
    XJ = with_physics(Xj)
    med = _hyb_pred(models[""], XJ)
    preds["ML v3 (this work)"] = med
    lo, hi = np.minimum(_hyb_pred(models["_q05"], XJ), med), np.maximum(_hyb_pred(models["_q95"], XJ), med)
    # Real people vary more than simulated ones, so the simulated 90 % range is too narrow on real data.
    # Conformal calibration (CQR, Romano et al. 2019) on the log scale: widen by the 90 % quantile of
    # max(log(lo/y), log(y/hi)) measured on HALF of the real runs, check coverage on the OTHER half.
    runs = sorted(R["run"].unique())
    cal_runs = set(runs[::2])
    score = np.maximum(np.log(lo / np.maximum(y, 0.1)), np.log(np.maximum(y, 0.1) / hi))
    cal = R["run"].isin(cal_runs).to_numpy()
    q_half = float(np.quantile(score[cal], 0.9))
    test = ~cal
    cover_raw = float(np.mean((y[test] >= lo[test]) & (y[test] <= hi[test])))
    cover_cqr = float(np.mean((y[test] >= lo[test] * np.exp(-q_half)) & (y[test] <= hi[test] * np.exp(q_half))))
    margin = float(np.quantile(score, 0.9))          # final margin from all real runs, used by route_eta
    meta_p = os.path.join(MODELS, "delay_model_v3_meta.json")
    with open(meta_p, encoding="utf-8") as f:
        meta = json.load(f)
    meta["real_log_margin"] = margin
    meta["real_conformal_check"] = {"calibration_runs": sorted(cal_runs), "test_coverage_before": cover_raw,
                                    "test_coverage_after": cover_cqr, "margin_from_half": q_half}
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    rows = []
    for test in ["all"] + sorted(R["test"].unique()):
        m = np.ones(len(R), bool) if test == "all" else (R["test"] == test).to_numpy()
        r = {"real test set": test, "crossings": int(m.sum()), "mean measured time (s)": float(y[m].mean())}
        for k, p in preds.items():
            r[k] = _mae(y[m], p[m])
        r["v3 90% range covers"] = float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))
        rows.append(r)
    rows[0]["conformal"] = {"before": cover_raw, "after": cover_cqr, "margin": q_half, "factor": float(np.exp(margin))}
    return rows, R, preds


# ================================================================== step 6: in the planner
def _loop_job(args):
    building, name, method, seed = args
    from evac import config as C
    from evac.building import load_building
    from evac.simulator import run
    C.DELAY_MODEL = "v3"
    if building not in _G:
        _G[building] = load_building(building)
    if building == "hospital":
        from evac.hospital import scenarios
        sc = scenarios()[name]
    else:
        with open(os.path.join(HERE, "scenarios.json"), encoding="utf-8") as f:
            sc = json.load(f)[name]
    m, recs = run(_G[building].copy(), sc, method, seed)
    amb = [r["t_out"] for r in recs if r["t_out"] is not None and r.get("profile") in ("walk", "slow", "step_free")]
    return {"building": building, "scenario": name, "method": method, "seed": seed,
            "last": m["evac_time_last_s"], "ambulatory_last": max(amb) if amb else None,
            "mean": m["evac_time_mean_s"], "exposed": m["exposed_agents"]}


LOOP_CAMPUS = ["floor3_full_200", "lecture_hall_f1_120", "floor3_200_spreading_fire_stairs1",
               "building_250_spreading_fire_j19", "crowd_250_fire_j19"]


def in_loop(workers=2, seeds=10):
    out = os.path.join(RES, "ml_v3_loop.csv")
    import pandas as pd
    if os.path.exists(out):
        return pd.read_csv(out)
    import multiprocessing as mp
    from evac.hospital import scenarios
    jobs = []
    for b, names in (("campus", LOOP_CAMPUS), ("hospital", list(scenarios()))):
        for name in names:
            for sd in range(seeds):
                for meth in ("lookahead", "lookahead_ml", "pred_formula", "pred_ml"):
                    jobs.append((b, name, meth, sd))
    rows = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_loop_job, jobs, chunksize=2)):
            rows.append(r)
            if (k + 1) % 100 == 0:
                print(f"  in-loop {k + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


# ================================================================== report
def report(cv, transfer, real, R, preds, loop, n_rows, ablation_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from evac.final_report import BLUE, ORANGE, GREY, MUTED, _style
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    CV = pd.DataFrame(cv)
    sub = CV[CV["subset"].isin(["all crossings", "someone queued ahead", "in smoke (hazard >= 0.3)",
                                "slow movers (patients, wheelchairs, beds)"])]
    x = np.arange(len(sub))
    ax[0].bar(x - 0.2, sub["MAE formula"], 0.4, color=GREY, label="queueing formula")
    ax[0].bar(x + 0.2, sub["MAE ML"], 0.4, color=BLUE, label="ML v3")
    ax[0].set_xticks(x, ["all", "queue ahead", "in smoke", "slow movers"], fontsize=8)
    ax[0].set_ylabel("Error (s), held-out evacuations", fontsize=9, color=MUTED)
    ax[0].set_title("Simulated test data", loc="left", fontsize=11, fontweight="bold")
    ax[0].legend(fontsize=8, frameon=False)
    _style(ax[0])
    RT = pd.DataFrame(real)
    keys = [k for k in preds]
    cols = ["#e0b48a", GREY, ORANGE, BLUE][-len(keys):]
    tests = list(RT["real test set"])
    x = np.arange(len(tests))
    wd = 0.8 / len(keys)
    for i, (k, c) in enumerate(zip(keys, cols)):
        ax[1].bar(x + (i - (len(keys) - 1) / 2) * wd, RT[k], wd, color=c, label=k)
    ax[1].set_xticks(x, tests, fontsize=8)
    ax[1].set_ylabel("Error (s) on REAL crossings", fontsize=9, color=MUTED)
    ax[1].set_title("Real Juelich experiments", loc="left", fontsize=11, fontweight="bold")
    ax[1].legend(fontsize=7, frameon=False)
    _style(ax[1])
    y = R["target_time_s"].to_numpy()
    p = preds["ML v3 (this work)"]
    bott = (R["test"] == "bottleneck").to_numpy()
    ax[2].scatter(y[~bott], p[~bott], s=4, color="#cfcdc4", label="corridors")
    ax[2].scatter(y[bott], p[bott], s=6, color=BLUE, label="bottlenecks")
    lim = float(np.percentile(np.r_[y, p], 99.5))
    ax[2].plot([0, lim], [0, lim], color=GREY, lw=1)
    ax[2].set(xlim=(0, lim), ylim=(0, lim), xlabel="Measured time (s)", ylabel="ML v3 predicted (s)")
    ax[2].set_title("Each real person", loc="left", fontsize=11, fontweight="bold")
    ax[2].legend(fontsize=8, frameon=False)
    _style(ax[2])
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)
    plt.close(fig)

    allr = RT[RT["real test set"] == "all"].iloc[0]
    allc = CV[CV["subset"] == "all crossings"].iloc[0]
    crowd = CV[CV["subset"] == "someone queued ahead"].iloc[0]
    smoke = CV[CV["subset"] == "in smoke (hazard >= 0.3)"].iloc[0]
    bt = RT[RT["real test set"] == "bottleneck"].iloc[0]
    L = ["# Improving ML: one model, calibrated simulator, tested on real data", "",
         "See `evac/ml_v3.py`. Calibration: `docs/calibration.md`.", "", "## Findings", "",
         f"1. **Dataset**: {n_rows:,} edge crossings from the CALIBRATED simulator (smoke slows walking): campus, "
         "hospital and 600 random simple geometries (doors 0.5-2.4 m, corridors 1-5.5 m, walkers, patients, "
         "wheelchairs, beds). Labels are simulated crossing times, not a formula.",
         f"2. **Simulated test data** (whole evacuations held out, 5-fold): error {allc['MAE formula']:.2f} s "
         f"(formula) -> {allc['MAE ML']:.2f} s (ML), {1 - allc['MAE ML'] / allc['MAE formula']:.0%} lower. Where it "
         f"matters most - someone already queued ahead - {crowd['MAE formula']:.1f} s -> {crowd['MAE ML']:.1f} s; "
         f"in smoke {smoke['MAE formula']:.1f} s -> {smoke['MAE ML']:.1f} s. Its 90% range contains the true time for "
         f"{allc['90% range covers']:.0%} of crossings.",
         f"3. **Real data** ({int(allr['crossings']):,} people measured in the Juelich experiments, never used for "
         "training): error " + ", ".join(f"{k} {allr[k]:.2f} s" for k in preds) + ". "
         f"On the bottlenecks (queues): " + ", ".join(f"{k} {bt[k]:.2f} s" for k in preds) + ". "
         f"The simulated 90% range covers only {allr['v3 90% range covers']:.0%} of real crossings (real people vary "
         f"more than simulated ones); after conformal calibration on half of the real runs it covers "
         f"{allr['conformal']['after']:.0%} on the other half (was {allr['conformal']['before']:.0%}). Time-to-exit "
         f"ranges on signs use this widened range (x/÷ {allr['conformal']['factor']:.2f} per crossing).",
         ]
    AB = pd.DataFrame(ablation_rows)
    pure = AB.iloc[1]
    hyb = AB.iloc[-1]
    phy = AB.iloc[0]
    L.append(f"4. **Why a hybrid model.** A pure ML model trained on the same data (like v2) was accurate on simulated "
             f"data ({pure['MAE simulated (held out)']:.2f} s) but poor on real people ({pure['MAE real (all)']:.2f} s): "
             f"it had never seen e.g. 5 m wide corridors. v3 instead learns a CORRECTION to the physics formula "
             f"(time = formula x learned factor) from physics-based inputs: simulated {hyb['MAE simulated (held out)']:.2f} s, "
             f"real {hyb['MAE real (all)']:.2f} s (formula alone: {phy['MAE simulated (held out)']:.2f} s / "
             f"{phy['MAE real (all)']:.2f} s).")
    TR = pd.DataFrame(transfer)
    L.append("5. **Never-seen building**: " + "; ".join(
        f"trained without the {r['test building (never seen)']}, tested on it: formula {r['MAE formula']:.2f} s, "
        f"ML {r['MAE ML']:.2f} s" for _, r in TR.iterrows()) + ".")
    loop = loop.copy()
    loop["metric"] = loop.apply(lambda r: r["ambulatory_last"] if r["building"] == "hospital" else r["last"], axis=1)
    piv = loop.groupby(["building", "scenario", "method"])["metric"].mean().unstack()
    pe = loop.groupby(["building", "scenario", "method"])["exposed"].mean().unstack()
    d1 = (piv["lookahead_ml"] / piv["lookahead"] - 1)
    d2 = (piv["pred_ml"] / piv["pred_formula"] - 1)
    L.append(f"6. **In the planner** (10 seeds x {len(piv)} scenarios): capacity planner with ML walking times vs "
             f"formula: mean {d1.mean():+.1%} (range {d1.min():+.1%} to {d1.max():+.1%}); simple re-router with ML "
             f"delays vs formula delays: mean {d2.mean():+.1%} (range {d2.min():+.1%} to {d2.max():+.1%}). "
             "Negative = ML faster.")
    L += ["", "![ML v3](figures/fig29_ml_v3.png)", "",
          "## Simulated test data (held-out evacuations)", "",
          "| Subset | Rows | Mean time (s) | MAE formula (s) | MAE ML (s) | 90% range covers |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in cv:
        L.append(f"| {r['subset']} | {r['rows']:,} | {r['mean_time_s']:.1f} | {r['MAE formula']:.2f} | "
                 f"{r['MAE ML']:.2f} | {r['90% range covers']:.0%} |")
    L += ["", "## Real data (Juelich experiments)", "",
          "Each person walking through a 4 m corridor section (corridor runs) or through the last 3 m and the door "
          "(bottleneck runs) is one test row, with the same features the simulator logs (people ahead, density, "
          "people still to come). Mean absolute error in seconds:", "",
          "| Real test set | Crossings | Mean measured (s) | " + " | ".join(preds) + " | v3 90% range covers |",
          "| --- | --- | --- | " + " | ".join("---" for _ in preds) + " | --- |"]
    for r in real:
        r = {k: v for k, v in r.items() if k != "conformal"}
        L.append(f"| {r['real test set']} | {r['crossings']:,} | {r['mean measured time (s)']:.1f} | " +
                 " | ".join(f"{r[k]:.2f}" for k in preds) + f" | {r['v3 90% range covers']:.0%} |")
    L += ["", "## Design choices (ablation: one grouped 75/25 split + all real crossings)", "",
          "| Variant | " + " | ".join(c for c in AB.columns if c != "variant") + " |",
          "| --- | " + " | ".join("---" for c in AB.columns if c != "variant") + " |"]
    for _, r in AB.iterrows():
        L.append(f"| {r['variant']} | " + " | ".join(f"{r[c]:.2f}" for c in AB.columns if c != "variant") + " |")
    L += ["", "## A building the model has never seen", "",
          "| Test building | Trained on | MAE formula (s) | MAE ML (s) |", "| --- | --- | --- | --- |"]
    for r in transfer:
        L.append(f"| {r['test building (never seen)']} | {r['trained on']} | {r['MAE formula']:.2f} | {r['MAE ML']:.2f} |")
    L += ["", "## In the planner (mean of 10 seeds; hospital = last walking person, campus = last person)", "",
          "| Building | Scenario | Look-ahead, formula | Look-ahead, ML v3 | Re-router, formula | Re-router, ML v3 | Exposed (LA formula / LA ML) |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for (b, s), r in piv.iterrows():
        e = pe.loc[(b, s)]
        L.append(f"| {b} | {s} | {r['lookahead']:.0f} | {r['lookahead_ml']:.0f} | {r['pred_formula']:.0f} | "
                 f"{r['pred_ml']:.0f} | {e['lookahead']:.1f} / {e['lookahead_ml']:.1f} |")
    L += ["", "## What changed from the old ML", "",
          "| | v1 Random Forest | v2 LightGBM (Session 7) | v3 LightGBM (this work) |", "| --- | --- | --- | --- |",
          "| Labels | a hand-written cost formula + noise | simulated crossing times | simulated crossing times |",
          "| Simulator checked against real people? | - | no (doors at ~half the real flow) | yes, calibrated on Juelich data |",
          "| Smoke slows people | - | no | yes (Frantzich & Nilsson 2003) |",
          "| Model | Random Forest | LightGBM | hybrid: physics formula x LightGBM correction |",
          "| Training geometry | random numbers | campus + hospital | campus + hospital + 600 random geometries |",
          "| Tested on real crossings | no | no | yes |",
          "| Gives a range (for time-to-exit signs) | no | 95% upper only | 5%-95% range |", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    from evac import config as C
    C.use_params("juelich")
    os.makedirs(RES, exist_ok=True)
    generate(a.workers)
    if a.generate:
        return
    df = load_sim()
    print(f"rows: {len(df):,}", df["building"].value_counts().to_dict(), flush=True)
    cache = os.path.join(RES, "ml_v3_eval.json")
    if os.path.exists(cache) and os.path.exists(os.path.join(MODELS, "delay_lgbm_v3.joblib")):
        with open(cache) as f:
            ev = json.load(f)
        import joblib
        models = {s: joblib.load(os.path.join(MODELS, f"delay_lgbm_v3{s}.joblib")) for s in ("", "_q05", "_q95")}
    else:
        print("cross-validation ...", flush=True)
        print("ablation ...", flush=True)
        ab = ablation(df)
        for r in ab:
            print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        ev = {"ablation": ab, "transfer": building_transfer(df), "cv": cross_validate(df)}
        models = train_final(df)
        with open(cache, "w") as f:
            json.dump(ev, f, indent=1)
    real, R, preds = real_test(models, df)
    for r in real:
        print(" ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    print("in the planner ...", flush=True)
    loop = in_loop(a.workers)
    for line in report(ev["cv"], ev["transfer"], real, R, preds, loop, len(df), ev["ablation"])[4:11]:
        print(line)


if __name__ == "__main__":
    main()
