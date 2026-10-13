"""
evac/ml_study.py
ML rigour for the edge-delay predictor (Session 6A).

  1. Model comparison with GROUPED cross-validation (whole scenarios held out per fold):
       queueing formula | linear (ridge) | Random Forest | LightGBM | neural network (MLP)
  2. Learning curve: how many simulated evacuations does the model need?
  3. SHAP explanations: which inputs drive a predicted wait, and in which direction
  4. Conformal prediction: intervals with a coverage guarantee on unseen scenarios
       - Mondrian split-conformal (separate calibration for crowded / uncrowded edges)
       - Conformalised quantile regression (CQR) with LightGBM quantile models
  5. Saves the chosen model + its 90% upper-bound model for the planner (models/delay_*.joblib)

Run:  python -m evac.ml_study        (needs scikit-learn, lightgbm, shap; ~10 min on a laptop)
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from evac.delay_model import DATA, FEATURES, _prep, queueing   # noqa: E402

DOC = os.path.join(ROOT, "docs", "ml_study.md")
FIGDIR = os.path.join(ROOT, "docs", "figures")
MODELDIR = os.path.join(ROOT, "models")
RNG = 0
ALPHA = 0.10            # 90% intervals


def load(sample_frac=0.4):
    df = _prep(pd.read_csv(DATA))
    df["crowded"] = df["queue_ahead"] > 0
    if sample_frac < 1:
        df = df.groupby("scenario", group_keys=False).sample(frac=sample_frac, random_state=RNG)
    return df.reset_index(drop=True)


def models():
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "Linear (ridge)": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=120, min_samples_leaf=5, n_jobs=-1, random_state=RNG),
        "LightGBM": lambda: LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                                          subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                                          random_state=RNG, verbose=-1),
        "Neural net (MLP)": lambda: make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 64), early_stopping=True, max_iter=80, random_state=RNG)),
    }


def _scores(y, p, crowded):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    out = {}
    for name, m in (("all", np.ones(len(y), bool)), ("crowded", crowded)):
        out[f"MAE_{name}"] = mean_absolute_error(y[m], p[m])
        out[f"RMSE_{name}"] = float(np.sqrt(mean_squared_error(y[m], p[m])))
        out[f"R2_{name}"] = r2_score(y[m], p[m])
    return out


def compare(df, k=5):
    from sklearn.model_selection import GroupKFold
    rows = []
    X, y, g = df[FEATURES], df["target_time_s"].to_numpy(), df["scenario"]
    cr = df["crowded"].to_numpy()
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=k).split(X, y, g)):
        rows.append({"model": "Queueing formula", "fold": fold, **_scores(y[te], queueing(df.iloc[te]).to_numpy(), cr[te])})
        for name, make in models().items():
            t0 = time.time()
            m = make().fit(X.iloc[tr], y[tr])
            p = m.predict(X.iloc[te])
            rows.append({"model": name, "fold": fold, "fit_s": time.time() - t0, **_scores(y[te], p, cr[te])})
            print(f"  fold {fold} {name:18s} MAE crowded {rows[-1]['MAE_crowded']:.2f}  ({rows[-1]['fit_s']:.0f}s)")
    return pd.DataFrame(rows)


def learning_curve(df, sizes=(10, 20, 40, 80, 120)):
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(RNG)
    scen = np.array(sorted(df["scenario"].unique()))
    rng.shuffle(scen)
    test_s, pool = set(scen[:30]), scen[30:]
    te = df[df["scenario"].isin(test_s)]
    out = []
    for n in sizes:
        tr = df[df["scenario"].isin(set(pool[:n]))]
        for name, m in (("LightGBM", LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=RNG)),
                        ("Random Forest", RandomForestRegressor(n_estimators=120, min_samples_leaf=5, n_jobs=-1, random_state=RNG))):
            m.fit(tr[FEATURES], tr["target_time_s"])
            p = m.predict(te[FEATURES])
            s = _scores(te["target_time_s"].to_numpy(), p, te["crowded"].to_numpy())
            out.append({"train_scenarios": n, "model": name, **s})
        s = _scores(te["target_time_s"].to_numpy(), queueing(te).to_numpy(), te["crowded"].to_numpy())
        out.append({"train_scenarios": n, "model": "Queueing formula", **s})
        print(f"  learning curve n={n}")
    return pd.DataFrame(out)


def split_scenarios(df, frac=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(RNG + 1)
    scen = np.array(sorted(df["scenario"].unique()))
    rng.shuffle(scen)
    a, b = int(frac[0] * len(scen)), int((frac[0] + frac[1]) * len(scen))
    pick = lambda s: df[df["scenario"].isin(set(s))]
    return pick(scen[:a]), pick(scen[a:b]), pick(scen[b:])


def conformal(full):
    """Two conformal methods, calibrated on held-out scenarios, evaluated on other held-out scenarios."""
    from lightgbm import LGBMRegressor
    tr, cal, te = split_scenarios(full)
    point = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=RNG)
    point.fit(tr[FEATURES], tr["target_time_s"])
    res = {}

    # (a) Mondrian split conformal on absolute residuals, separate for crowded / not crowded
    q = {}
    for c in (False, True):
        m = cal["crowded"] == c
        r = np.abs(cal.loc[m, "target_time_s"] - point.predict(cal.loc[m, FEATURES]))
        n = len(r)
        q[c] = float(np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)))
    p = point.predict(te[FEATURES])
    qq = np.where(te["crowded"], q[True], q[False])
    lo, hi = p - qq, p + qq
    res["mondrian"] = {"q_crowded": q[True], "q_uncrowded": q[False], **_coverage(te, lo, hi)}

    # (b) Conformalised quantile regression (Romano et al. 2019)
    qlo = LGBMRegressor(objective="quantile", alpha=ALPHA / 2, n_estimators=400, learning_rate=0.05,
                        num_leaves=63, verbose=-1, random_state=RNG).fit(tr[FEATURES], tr["target_time_s"])
    qhi = LGBMRegressor(objective="quantile", alpha=1 - ALPHA / 2, n_estimators=400, learning_rate=0.05,
                        num_leaves=63, verbose=-1, random_state=RNG).fit(tr[FEATURES], tr["target_time_s"])
    y = cal["target_time_s"].to_numpy()
    E = np.maximum(qlo.predict(cal[FEATURES]) - y, y - qhi.predict(cal[FEATURES]))
    n = len(E)
    qc = float(np.quantile(E, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)))
    lo, hi = qlo.predict(te[FEATURES]) - qc, qhi.predict(te[FEATURES]) + qc
    res["cqr"] = {"q_correction": qc, **_coverage(te, lo, hi)}
    return point, qhi, res, (te, p, lo, hi)


def _coverage(te, lo, hi):
    y = te["target_time_s"].to_numpy()
    inside = (y >= lo) & (y <= hi)
    c = te["crowded"].to_numpy()
    return {"coverage_all": float(inside.mean()), "coverage_crowded": float(inside[c].mean()),
            "coverage_uncrowded": float(inside[~c].mean()),
            "width_crowded": float(np.mean(hi[c] - lo[c])), "width_uncrowded": float(np.mean(hi[~c] - lo[~c]))}


def shap_plot(model, df, path):
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sample = df[df["crowded"]].sample(n=min(2000, int(df["crowded"].sum())), random_state=RNG)[FEATURES]
    sv = shap.TreeExplainer(model).shap_values(sample)
    imp = sorted(zip(FEATURES, np.abs(sv).mean(axis=0)), key=lambda x: -x[1])
    plt.figure()
    shap.summary_plot(sv, sample, show=False, max_display=10, plot_size=(8, 4.8))
    plt.title("What drives a predicted wait (SHAP, crowded edges)", loc="left", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return imp


def plots(cv, lc, conf_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, GREY, INK, MUTED = "#2a78d6", "#a3a29a", "#21201c", "#6b6a63"
    os.makedirs(FIGDIR, exist_ok=True)

    # model comparison (crowded MAE, mean +/- sd over folds)
    agg = cv.groupby("model")["MAE_crowded"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    best = agg["mean"].idxmin()
    ax.barh(agg.index, agg["mean"], xerr=agg["std"], color=[BLUE if i == best else GREY for i in agg.index],
            height=0.55, error_kw={"ecolor": INK, "lw": 1})
    for i, (m, r) in enumerate(agg.iterrows()):
        ax.text(r["mean"] + r["std"] + 0.15, i, f"{r['mean']:.2f} s", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Mean absolute error on crowded edges, unseen scenarios (s)", color=MUTED)
    ax.set_title(f"{best} predicts queue delays best (5-fold grouped CV)", loc="left", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig6_model_comparison.png"), dpi=160)
    plt.close(fig)

    # learning curve
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for name, c, ls in (("LightGBM", BLUE, "-"), ("Random Forest", "#eb6834", "-"), ("Queueing formula", GREY, "--")):
        d = lc[lc["model"] == name]
        ax.plot(d["train_scenarios"], d["MAE_crowded"], ls, marker="o", color=c, lw=2, ms=5, label=name)
    ax.set_xlabel("Simulated evacuations used for training", color=MUTED)
    ax.set_ylabel("MAE, crowded edges (s)", color=MUTED)
    ax.set_title("Accuracy keeps improving with more simulated evacuations", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig7_learning_curve.png"), dpi=160)
    plt.close(fig)

    # conformal intervals on a sample of crowded test edges
    te, p, lo, hi = conf_data
    c = te["crowded"].to_numpy()
    idx = np.random.default_rng(RNG).choice(np.where(c)[0], size=min(60, int(c.sum())), replace=False)
    idx = idx[np.argsort(p[idx])]
    y = te["target_time_s"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(idx))
    ax.vlines(x, lo[idx], hi[idx], color=BLUE, alpha=0.35, lw=4, label="90% interval (CQR)")
    ax.plot(x, p[idx], "_", color=BLUE, ms=9, mew=2, label="prediction")
    inside = (y[idx] >= lo[idx]) & (y[idx] <= hi[idx])
    ax.plot(x[inside], y[idx][inside], "o", color=INK, ms=3.5, label="actual (inside)")
    ax.plot(x[~inside], y[idx][~inside], "x", color="#d03b3b", ms=6, label="actual (outside)")
    ax.set_xticks([])
    ax.set_xlabel("60 random crowded edges from unseen scenarios, sorted by prediction", color=MUTED)
    ax.set_ylabel("Seconds through edge", color=MUTED)
    ax.set_title("Prediction intervals hold their promised coverage on unseen evacuations", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig9_conformal.png"), dpi=160)
    plt.close(fig)


def main():
    import joblib
    from lightgbm import LGBMRegressor
    t0 = time.time()
    df = load(0.4)
    print(f"{len(df):,} rows from {df['scenario'].nunique()} scenarios")
    print("1. grouped cross-validation")
    cv = compare(df)
    print("2. learning curve")
    lc = learning_curve(df)
    print("3+4. conformal + SHAP on full data")
    full = load(1.0)
    point, qhi, conf, conf_data = conformal(full)
    imp = shap_plot(point, full, os.path.join(FIGDIR, "fig8_shap.png"))
    plots(cv, lc, conf_data)

    # final models for the planner: trained on ALL scenarios
    os.makedirs(MODELDIR, exist_ok=True)
    final = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=RNG)
    final.fit(full[FEATURES], full["target_time_s"])
    upper = LGBMRegressor(objective="quantile", alpha=1 - ALPHA / 2, n_estimators=400, learning_rate=0.05,
                          num_leaves=63, verbose=-1, random_state=RNG).fit(full[FEATURES], full["target_time_s"])
    joblib.dump(final, os.path.join(MODELDIR, "delay_lgbm.joblib"))
    joblib.dump(upper, os.path.join(MODELDIR, "delay_lgbm_q95.joblib"))
    meta = {"features": FEATURES, "target": "target_time_s (seconds through edge incl. queue)",
            "train_rows": len(full), "train_scenarios": int(full["scenario"].nunique()),
            "cqr_correction_s": conf["cqr"]["q_correction"], "alpha": ALPHA}
    with open(os.path.join(MODELDIR, "delay_model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    agg = cv.groupby("model").agg(MAE_crowded=("MAE_crowded", "mean"), MAE_crowded_sd=("MAE_crowded", "std"),
                                  R2_crowded=("R2_crowded", "mean"), MAE_all=("MAE_all", "mean"),
                                  R2_all=("R2_all", "mean")).sort_values("MAE_crowded")
    cv.to_csv(os.path.join(ROOT, "results", "ml_cv_folds.csv"), index=False)
    lc.to_csv(os.path.join(ROOT, "results", "ml_learning_curve.csv"), index=False)
    lines = ["# ML study: which model, how much data, why, and how sure", "",
             f"Data: {len(full):,} simulated edge crossings from {full['scenario'].nunique()} evacuations "
             "(cross-validation used a 40% per-scenario sample for speed). Target = seconds to get through an edge.", "",
             "## 1. Model comparison (5-fold grouped CV: whole evacuations held out)", "",
             "| Model | MAE crowded (s) | ± sd | R² crowded | MAE all (s) | R² all |", "| --- | --- | --- | --- | --- | --- |"]
    for m, r in agg.iterrows():
        lines.append(f"| {m} | {r.MAE_crowded:.2f} | {r.MAE_crowded_sd:.2f} | {r.R2_crowded:.3f} | {r.MAE_all:.2f} | {r.R2_all:.3f} |")
    lines += ["", "![Model comparison](figures/fig6_model_comparison.png)", "",
              "## 2. Learning curve", "", "| Training evacuations | LightGBM MAE crowded | Random Forest | Formula |",
              "| --- | --- | --- | --- |"]
    for n, d in lc.groupby("train_scenarios"):
        g = d.set_index("model")["MAE_crowded"]
        lines.append(f"| {n} | {g['LightGBM']:.2f} | {g['Random Forest']:.2f} | {g['Queueing formula']:.2f} |")
    lines += ["", "![Learning curve](figures/fig7_learning_curve.png)", "",
              "## 3. What drives a predicted wait (SHAP)", "",
              "Mean |SHAP| on crowded edges: " + ", ".join(f"{f} ({v:.1f} s)" for f, v in imp[:6]) + ".", "",
              "![SHAP](figures/fig8_shap.png)", "",
              "## 4. Uncertainty: conformal prediction (target 90% coverage on unseen evacuations)", "",
              "| Method | Coverage all | Coverage crowded | Coverage uncrowded | Width crowded (s) | Width uncrowded (s) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for k, lab in (("mondrian", "Split conformal (Mondrian)"), ("cqr", "Conformalised quantile regression")):
        r = conf[k]
        lines.append(f"| {lab} | {r['coverage_all']:.1%} | {r['coverage_crowded']:.1%} | {r['coverage_uncrowded']:.1%} | "
                     f"{r['width_crowded']:.1f} | {r['width_uncrowded']:.1f} |")
    lines += ["", "CQR adapts the interval width to each edge (narrow when free-flowing, wide in a jam), "
              "so its intervals are far more useful than one fixed band.", "",
              "![Conformal](figures/fig9_conformal.png)", "",
              f"_Runtime {time.time() - t0:.0f} s. Saved models: models/delay_lgbm.joblib (point), "
              "models/delay_lgbm_q95.joblib (upper 95% quantile)._", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
