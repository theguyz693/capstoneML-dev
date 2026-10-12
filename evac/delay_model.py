"""
evac/delay_model.py
Does machine learning earn its place?  (answers the panel's "why ML / overfitting / labels" questions)

OLD model (models/edge_cost_rf.joblib): learns a hand-written cost formula from random
numbers -> it can only copy the formula (it scores R^2 0.990 vs the formula's own 0.999).

NEW model (this file): predicts how long it ACTUALLY takes to get through an edge
(queue wait + walk) from what a person sees when they arrive at it. Labels come from
the simulator, not from a formula, and include queueing effects of everyone else.

  features  edge geometry (type, length, width, sustainable capacity, free speed), hazard,
            people queued ahead for this edge, people at the node, people on the edge,
            density, number of other people whose planned route uses this edge
  target    seconds from joining the queue at u to stepping off edge u->v

Compared against (all evaluated on data the RF never saw):
  free-flow   length / free speed                       (ignores crowds)
  queueing    queue_ahead / capacity + length / speed at current density   (textbook formula)
  RF          Random Forest on the features above

Splits (why they matter):
  random rows      the naive split used before - rows from the SAME run land in train and test
  by scenario      whole simulated evacuations held out - no leakage between train and test
  by fire location fires at nodes never seen in training - a harder generalisation test

Run:  python -m evac.delay_model            (generates data, trains, writes docs/ml_delay_results.md)
"""

import csv
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from evac.building import load_campus           # noqa: E402
from evac.physics import weidmann_speed_factor  # noqa: E402
from evac.simulator import run                  # noqa: E402

FEATURES = ["is_stair", "is_room_door", "is_exit_door", "is_corridor", "length_m", "width_m",
            "eff_cap", "free_speed", "hazard", "queue_ahead", "queue_at_node", "on_edge", "density",
            "planned_users"]
DATA = os.path.join(ROOT, "results", "delay_dataset.csv")
DOC = os.path.join(ROOT, "docs", "ml_delay_results.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig4_ml_delay.png")


def random_scenarios(G, n, seed=0):
    rng = random.Random(seed)
    junctions = sorted(x for x in G if G.nodes[x]["kind"] == "junction")
    big_rooms = sorted(x for x in G if G.nodes[x]["kind"] == "room" and
                       any(w in x for w in ("sem hall", "lab", "classroom", "conference")))
    out = []
    for i in range(n):
        sc = {"_name": f"gen{i:03d}", "n_agents": rng.choice([60, 100, 150, 200, 250]),
              "step_free_frac": 0.05, "compliance": rng.choice([1.0, 1.0, 0.8, 0.6]),
              "herding": rng.choice([0.0, 0.5])}
        mode = rng.choice(["building", "floor", "floor", "single"])
        sc["spawn"] = mode
        if mode == "floor":
            sc["floor"] = rng.randint(1, 6)
        if mode == "single":
            sc["origin"] = rng.choice(big_rooms)
            sc["n_agents"] = rng.choice([60, 100, 120])
        kind = rng.choice(["none", "static", "spreading"])
        if kind != "none":
            f = {"node": rng.choice(junctions), "intensity": 1.0}
            if kind == "static":
                f["radius"] = rng.choice([300, 500, 780])
            else:
                f.update(radius=150, spread_m_per_s=rng.choice([0.3, 0.5]), max_radius=900)
            sc["fires"] = [f]
        else:
            sc["fires"] = []
        sc["_method"] = rng.choice(["safe_shortest", "reactive", "ccrp"])
        out.append(sc)
    return out


def generate(n_scenarios=160, seed=0, path=DATA):
    G0 = load_campus()
    rows = []
    t0 = time.time()
    for i, sc in enumerate(random_scenarios(G0, n_scenarios, seed)):
        m, _ = run(G0.copy(), {**sc, "log_edges": True}, sc["_method"], seed=i)
        for r in m["edge_rows"]:
            r.pop("_t_join", None)
            r["method"] = sc["_method"]
            rows.append(r)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_scenarios} scenarios, {len(rows)} rows, {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def _prep(df):
    for et in ("stair", "room_door", "exit_door", "corridor"):
        df[f"is_{et}"] = (df["etype"] == et).astype(float)
    return df


def free_flow(df):
    return df["length_m"] / df["free_speed"]


def queueing(df):
    walk = df["length_m"] / (df["free_speed"] * df["density"].map(weidmann_speed_factor))
    return df["queue_ahead"] / df["eff_cap"] + walk


def evaluate(path=DATA, seed=0):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    df = _prep(pd.read_csv(path))
    rng = np.random.default_rng(seed)
    scen = df["scenario"].unique()
    fires = sorted(f for f in df["fire_node"].unique() if f != "none")

    test_scen = set(rng.choice(scen, size=int(0.25 * len(scen)), replace=False))
    test_fires = set(rng.choice(fires, size=max(1, int(0.3 * len(fires))), replace=False))
    splits = {
        "random rows (leaky)": rng.random(len(df)) < 0.25,
        "by scenario": df["scenario"].isin(test_scen).to_numpy(),
        "by fire location": df["fire_node"].isin(test_fires).to_numpy(),
    }
    results, imp, scatter = [], None, None
    for name, test in splits.items():
        tr, te = df[~test], df[test]   # 'by fire location': train has no row from a held-out fire
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed)
        rf.fit(tr[FEATURES], tr["target_time_s"])
        preds = {"free-flow": free_flow(te), "queueing formula": queueing(te), "Random Forest": rf.predict(te[FEATURES])}
        for subset, mask in (("all edges", np.ones(len(te), bool)), ("crowded (queue ahead > 0)", (te["queue_ahead"] > 0).to_numpy())):
            if mask.sum() < 20:
                continue
            for model, p in preds.items():
                p = np.asarray(p)[mask]
                y = te["target_time_s"].to_numpy()[mask]
                results.append({"split": name, "subset": subset, "model": model, "n_test": int(mask.sum()),
                                "MAE_s": round(float(mean_absolute_error(y, p)), 2),
                                "R2": round(float(r2_score(y, p)), 3)})
        if name == "by scenario":
            imp = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])
            crowded = (te["queue_ahead"] > 0).to_numpy()
            scatter = (te["target_time_s"].to_numpy()[crowded], np.asarray(preds["Random Forest"])[crowded],
                       np.asarray(preds["queueing formula"])[crowded])
    return df, results, imp, scatter, sorted(test_fires)


def write_report(df, results, imp, scatter, test_fires):
    def row_of(split, subset, model):
        return next(r for r in results if r["split"] == split and r["subset"] == subset and r["model"] == model)

    lines = ["# Does ML earn its place? Learned delay predictor vs formulas", "",
             f"Data: {len(df):,} edge crossings from {df['scenario'].nunique()} simulated evacuations "
             "(random crowd sizes, layouts, static/spreading fires, 60-100% compliance). "
             "Target = actual seconds to get through an edge (queue wait + walk). Labels come from the simulator, not a formula.", "",
             "| Split | Edges | Model | MAE (s) | R² |", "| --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r['split']} | {r['subset']} (n={r['n_test']:,}) | {r['model']} | {r['MAE_s']} | {r['R2']} |")
    lines += ["", f"Fire locations held out in the 'by fire location' split: {', '.join(test_fires)}.", "",
              "Most important features (Random Forest, scenario split): " +
              ", ".join(f"{f} ({v:.2f})" for f, v in imp[:6]) + ".", ""]
    a = row_of("by scenario", "crowded (queue ahead > 0)", "Random Forest")
    b = row_of("by scenario", "crowded (queue ahead > 0)", "queueing formula")
    c = row_of("random rows (leaky)", "all edges", "Random Forest")
    d = row_of("by scenario", "all edges", "Random Forest")
    verdict = "beats" if a["MAE_s"] < b["MAE_s"] else "does NOT beat"
    lines += ["## Verdict", "",
              f"- On crowded edges in unseen scenarios, the Random Forest {verdict} the queueing formula "
              f"(MAE {a['MAE_s']} s vs {b['MAE_s']} s).",
              f"- Leakage check: random-row split R² {c['R2']} vs scenario split R² {d['R2']}. "
              "A large drop would mean the old random split overstated accuracy.",
              "- The old model (trained on a hand-written formula) should be retired: it cannot learn anything the formula does not already say.",
              "", "![ML delay](figures/fig4_ml_delay.png)", ""]
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        y, prf, pq = scatter
        lim = float(min(max(y.max(), prf.max()), 300))
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
        for axi, p, title, col in ((ax[0], pq, "Queueing formula", "#a3a29a"), (ax[1], prf, "Random Forest", "#2a78d6")):
            axi.scatter(y, p, s=5, alpha=0.35, color=col, lw=0)
            axi.plot([0, lim], [0, lim], color="#21201c", lw=1, ls=(0, (3, 3)))
            axi.set(xlim=(0, lim), ylim=(0, lim), xlabel="Actual time through edge (s)", title=title)
            axi.spines[["top", "right"]].set_visible(False)
        ax[0].set_ylabel("Predicted (s)")
        fig.suptitle("Crowded edges in scenarios never seen in training (dashed = perfect prediction)",
                     x=0.01, ha="left", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG, dpi=160)
    except ImportError:
        pass


def main():
    if not os.path.exists(DATA) or "--regen" in sys.argv:
        print("Generating training data from the simulator ...")
        generate()
    df, results, imp, scatter, test_fires = evaluate()
    write_report(df, results, imp, scatter, test_fires)
    for r in results:
        print(r)
    print(f"\nWrote {DOC}")


if __name__ == "__main__":
    main()
