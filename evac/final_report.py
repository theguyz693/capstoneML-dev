"""
evac/final_report.py
Turns results/final_*.csv (from evac.final_experiments) into the final results document and figures.

  docs/final_results.md
  docs/figures/fig12_main.png        improvement over shortest safe path, every scenario, 95% CI
  docs/figures/fig13_fairness.png    who pays: step-free / upper floors / lower floors
  docs/figures/fig14_mlloop.png      predictor-in-the-loop vs coordination
  docs/figures/fig15_tipping.png     crowd size at which coordination starts to matter
  docs/figures/fig16_freshness.png   how fresh the crowd data must be
  docs/figures/fig17_ablation.png    which planner ingredient matters

Statistics: means with 95% bootstrap CIs over paired seeds; paired Wilcoxon signed-rank tests with
Holm-Bonferroni correction across the scenarios of each comparison.

Run:  python -m evac.final_report
"""

import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import pandas as pd                                    # noqa: E402

from evac import config as C                           # noqa: E402
from evac.summarize import boot_ci, wilcoxon_p         # noqa: E402

RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "docs", "figures")
DOC = os.path.join(ROOT, "docs", "final_results.md")

BLUE, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#d9822b", "#a3a29a", "#21201c", "#6b6a63", "#e6e5df"
LABEL = {"nearest": "Nearest exit", "safe_shortest": "Shortest safe path", "reactive": "Reactive rerouting",
         "ccrp": "Capacity booking (one-shot)", "lookahead": "Look-ahead re-planning",
         "lookahead_ml": "Look-ahead + ML walk times", "pred_formula": "Re-route on queue formula",
         "pred_ml": "Re-route on ML prediction", "pred_ml_upper": "Re-route on ML 95% upper bound",
         "optimum": "Theoretical lower bound", "ga": "Genetic algorithm (offline)"}
METHODS = ["nearest", "safe_shortest", "reactive", "ccrp", "lookahead", "lookahead_ml"]
TITLE = {"baseline_150_no_fire": "150 spread, no fire", "fire_j19_150": "150 spread, fire j19",
         "fire_j6_east_exit_150": "150 spread, fire east exit", "lecture_hall_f1_120": "120 in one hall",
         "crowd_250_fire_j19": "250 spread, fire j19", "floor3_full_200": "200 on floor 3",
         "floor3_full_200_fire_stairs1": "200 floor 3, fire stairs 1",
         "floor3_200_spreading_fire_stairs1": "200 floor 3, spreading fire",
         "building_250_spreading_fire_j19": "250 spread, spreading fire",
         "floor3_200_compliance60_herding": "200 floor 3, 60% comply + herding"}


def _load(part):
    p = os.path.join(RES, f"final_{part}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if "evac_time_last_s" in df:
        # a run that hit the time limit gets the limit as its last-out time (conservative for every method)
        df["last"] = df["evac_time_last_s"].fillna(C.MAX_TIME_S)
    return df


def _holm(ps):
    """Holm-Bonferroni adjusted p-values (None stays None)."""
    idx = sorted([i for i, p in enumerate(ps) if p is not None], key=lambda i: ps[i])
    out, m, run = list(ps), len(idx), 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (m - k) * ps[i]))
        out[i] = run
    return out


def _paired(df, scen, a, b, col="last"):
    """b - a per seed (negative = b better)."""
    x = df[(df.scenario == scen) & (df.method == a)].set_index("seed")[col]
    y = df[(df.scenario == scen) & (df.method == b)].set_index("seed")[col]
    common = x.index.intersection(y.index)
    return list((y[common] - x[common]).astype(float)), list(x[common].astype(float))


def _fmt_p(p):
    if p is None:
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _style(ax, grid="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------- A: main comparison
def section_main(A, D, lines, findings):
    scen = [s for s in TITLE if s in set(A.scenario)]
    extra = []
    opt = D.groupby("scenario")["evac_time_last_s"].mean().to_dict() if D is not None else {}
    lines += ["## 1. Main comparison: 10 scenarios x 6 methods x 30 paired seeds", "",
              "Last-out time in seconds, mean [95% bootstrap CI]. Same 30 seeds (same people, same start "
              "positions, same pre-movement delays) for every method. Lower is better. "
              "**Bold** = fastest method among those that expose no more people to smoke/fire than the safest "
              "method in that row (+1 person or +10%) - a method that is faster only because it walks people "
              "through smoke does not count as best. Exposure is in the next table. "
              "Lower bound = time-expanded max-flow (3 seeds): it assumes everyone obeys, moves at free speed between "
              "bottlenecks and that the fire keeps its initial size, so it is looser for the compliance and "
              "spreading-fire scenarios.", ""]
    hdr = "| Scenario | " + " | ".join(LABEL[m] for m in METHODS) + " | Lower bound |"
    lines += [hdr, "|" + " --- |" * (len(METHODS) + 2)]
    for s in scen:
        means = {m: A[(A.scenario == s) & (A.method == m)]["last"].mean() for m in METHODS}
        expo = {m: A[(A.scenario == s) & (A.method == m)]["exposed_agents"].mean() for m in METHODS}
        lim = min(expo.values()) * 1.1 + 1
        best = min((m for m in METHODS if expo[m] <= lim), key=means.get)
        cells = []
        for m in METHODS:
            v = list(A[(A.scenario == s) & (A.method == m)]["last"])
            lo, hi = boot_ci(v)
            c = f"{means[m]:.0f} [{lo:.0f}-{hi:.0f}]"
            cells.append(f"**{c}**" if m == best else c)
        o = opt.get(s)
        lines.append(f"| {TITLE[s]} | " + " | ".join(cells) + f" | {o:.0f} |" if o == o and o is not None
                     else f"| {TITLE[s]} | " + " | ".join(cells) + " | - |")

    lines += ["", "Mean number of people exposed to smoke or fire (entered a node above the exposure threshold):", "",
              "| Scenario | " + " | ".join(LABEL[m] for m in METHODS) + " |", "|" + " --- |" * (len(METHODS) + 1)]
    for s in scen:
        if A[A.scenario == s]["exposed_agents"].max() == 0:
            continue
        lines.append(f"| {TITLE[s]} | " + " | ".join(
            f"{A[(A.scenario == s) & (A.method == m)]['exposed_agents'].mean():.1f}" for m in METHODS) + " |")
    lines.append("")
    lines.append("(Scenarios without fire, or where no method exposed anyone, are omitted.)")
    near = A[A.method == "nearest"]
    fire_sc = [s for s in scen if A[(A.scenario == s) & (A.method == "nearest")]["exposed_agents"].mean() >
               A[(A.scenario == s) & (A.method == "safe_shortest")]["exposed_agents"].mean() + 5]
    if fire_sc:
        e_n = near[near.scenario.isin(fire_sc)]["exposed_agents"].mean()
        e_s = A[(A.method == "safe_shortest") & A.scenario.isin(fire_sc)]["exposed_agents"].mean()
        extra.append(f"'Nearest exit' is sometimes the fastest - but only because it walks people through the fire: "
                        f"in the {len(fire_sc)} fire scenarios where it differs it exposed {e_n:.0f} people on average "
                        f"vs {e_s:.0f} for hazard-aware routing. Speed and safety must be read together.")
    spread = [s for s in scen if "spreading" in s]
    if spread:
        parts = []
        for s in spread:
            x = lambda m, c: A[(A.scenario == s) & (A.method == m)][c].mean()
            parts.append(f"{TITLE[s]}: one-shot {x('ccrp', 'last'):.0f} s / {x('ccrp', 'exposed_agents'):.0f} exposed, "
                         f"look-ahead {x('lookahead', 'last'):.0f} s / {x('lookahead', 'exposed_agents'):.0f} exposed")
        extra.append("In a spreading fire, re-planning trades a little time for much less smoke exposure (" +
                        "; ".join(parts) + ").")

    # paired test lookahead vs shortest
    lines += ["", "### Look-ahead re-planning vs shortest safe path (paired, per scenario)", "",
              "Difference = look-ahead minus shortest (negative = look-ahead faster). Wilcoxon signed-rank, "
              "Holm-corrected over the 10 scenarios. 'Wins' = seeds where look-ahead was strictly faster / tied / slower.", "",
              "| Scenario | Mean diff (s) [95% CI] | Change | Wins / ties / losses | p (Holm) | Exposed people: shortest -> look-ahead |",
              "| --- | --- | --- | --- | --- | --- |"]
    rows, ps = [], []
    for s in scen:
        d, base = _paired(A, s, "safe_shortest", "lookahead")
        rows.append((s, d, base))
        ps.append(wilcoxon_p(d))
    adj = _holm(ps)
    sig_better, tot_w, tot_l, tot_t = 0, 0, 0, 0
    for (s, d, base), p in zip(rows, adj):
        lo, hi = boot_ci(d)
        mean_d = statistics.mean(d)
        rel = mean_d / statistics.mean(base)
        w, t, l = sum(x < 0 for x in d), sum(x == 0 for x in d), sum(x > 0 for x in d)
        tot_w, tot_t, tot_l = tot_w + w, tot_t + t, tot_l + l
        if p is not None and p < 0.05 and mean_d < 0:
            sig_better += 1
        e0 = A[(A.scenario == s) & (A.method == "safe_shortest")]["exposed_agents"].mean()
        e1 = A[(A.scenario == s) & (A.method == "lookahead")]["exposed_agents"].mean()
        lines.append(f"| {TITLE[s]} | {mean_d:+.1f} [{lo:+.1f}, {hi:+.1f}] | {rel:+.1%} | {w} / {t} / {l} | "
                     f"{_fmt_p(p)} | {e0:.1f} -> {e1:.1f} |")
    worse = [TITLE[s] for (s, d, b), p in zip(rows, adj) if p is not None and p < 0.05 and statistics.mean(d) > 0]
    findings.append(f"Look-ahead re-planning was significantly faster than the shortest safe path in "
                    f"**{sig_better} of {len(scen)}** scenarios (Wilcoxon, Holm-corrected p<0.05) and "
                    f"significantly slower in {len(worse)}{' (' + ', '.join(worse) + ')' if worse else ''}. "
                    f"Over all {tot_w + tot_t + tot_l} paired runs it was faster in {tot_w}, tied in {tot_t}, slower in {tot_l}.")
    big = sorted(((statistics.mean(d) / statistics.mean(b), s) for s, d, b in rows))[:3]
    findings.append("Largest gains (concentrated crowds): " + "; ".join(f"{TITLE[s]} {r:+.0%}" for r, s in big) + ".")
    if opt:
        gaps = []
        for s in scen:
            o = opt.get(s)
            if o and o == o:
                la = A[(A.scenario == s) & (A.method == "lookahead")]["last"].mean()
                ss = A[(A.scenario == s) & (A.method == "safe_shortest")]["last"].mean()
                if ss - o > 1:
                    gaps.append((s, (ss - la) / (ss - o)))
        if gaps:
            med = statistics.median(g for _, g in gaps)
            findings.append(f"Look-ahead closes a median **{med:.0%}** of the gap between shortest path and the "
                            "theoretical lower bound (time-expanded max-flow, a bound no real routing can beat).")
            lines += ["", "Share of the shortest-path-to-lower-bound gap closed by look-ahead: " +
                      ", ".join(f"{TITLE[s]} {g:.0%}" for s, g in gaps) + "."]

    # reactive's cost: route changes
    rc = A.groupby("method")["route_changes"].mean()
    lines += ["", "Mean route changes per run (instructions people must follow): " +
              ", ".join(f"{LABEL[m]} {rc.get(m, 0):.0f}" for m in METHODS) + "."]

    findings.extend(extra)

    # figure 12
    plt = _plt()
    show = ["reactive", "ccrp", "lookahead", "lookahead_ml"]
    colors = {"reactive": GREY, "ccrp": ORANGE, "lookahead": BLUE, "lookahead_ml": "#7aa6e0"}
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(scen) + 1.4))
    for i, s in enumerate(scen):
        y0 = len(scen) - 1 - i
        for j, m in enumerate(show):
            d, base = _paired(A, s, "safe_shortest", m)
            bm = statistics.mean(base)
            rel = [x / bm * 100 for x in d]
            lo, hi = boot_ci(rel)
            y = y0 + (j - 1.5) * 0.17
            ax.plot([lo, hi], [y, y], color=colors[m], lw=2)
            ax.plot(statistics.mean(rel), y, "o", color=colors[m], ms=5, mec="white", mew=1,
                    label=LABEL[m] if i == 0 else None)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(scen)), [TITLE[s] for s in scen][::-1], fontsize=9)
    ax.set_xlabel("Change in last-out time vs shortest safe path (%), mean and 95% CI  (left = faster)",
                  fontsize=9, color=MUTED)
    ax.set_title("Coordinated re-planning wins where crowds concentrate, ties where they are spread",
                 loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig12_main.png"), dpi=170)
    plt.close(fig)
    lines += ["", "![Main comparison](figures/fig12_main.png)", ""]


# ---------------------------------------------------------------- F: fairness
def section_fairness(A, lines, findings):
    wide = [s for s in TITLE if s in set(A[A.upper_max_s.notna()].scenario)]
    A = A[A.scenario.isin(wide)]
    lines += ["## 2. Fairness: who pays for a faster average?", "",
              "A planner that speeds up the average by sending some people the long way could make the worst-off "
              "worse. Per group: mean exit time and the group's LAST person out, averaged over the building-wide "
              "scenarios (" + ", ".join(TITLE[s] for s in wide) + ") x 30 seeds. Upper floors = 4-6, lower floors = 0-2.", "",
              "| Method | Step-free: mean / last (s) | Upper floors: mean / last | Lower floors: mean / last | Everyone: mean / last |",
              "| --- | --- | --- | --- | --- |"]
    agg = {}
    for m in METHODS:
        x = A[A.method == m]
        agg[m] = {g: (x[f"{g}_mean_s"].mean(), x[f"{g}_max_s"].mean()) for g in ("sf", "upper", "lower")}
        agg[m]["all"] = (x["evac_time_mean_s"].mean(), x["last"].mean())
        lines.append(f"| {LABEL[m]} | " + " | ".join(f"{agg[m][g][0]:.0f} / {agg[m][g][1]:.0f}"
                                                     for g in ("sf", "upper", "lower", "all")) + " |")
    s0, s1 = agg["safe_shortest"], agg["lookahead"]
    worse = [g for g in ("sf", "upper", "lower") if s1[g][1] > s0[g][1] * 1.02]
    names = {"sf": "step-free users", "upper": "upper-floor occupants", "lower": "lower-floor occupants"}
    if worse:
        findings.append("Fairness: in building-wide scenarios look-ahead made the last person out later for " +
                        ", ".join(f"{names[g]} ({s0[g][1]:.0f} -> {s1[g][1]:.0f} s)" for g in worse) +
                        f", while upper floors improved ({s0['upper'][1]:.0f} -> {s1['upper'][1]:.0f} s): the planner "
                        "shares stair capacity with people coming from above. A cost to state openly.")
    else:
        findings.append("Fairness: look-ahead did not make any group's last person out later than shortest path "
                        f"(step-free {s0['sf'][1]:.0f}->{s1['sf'][1]:.0f} s, upper floors {s0['upper'][1]:.0f}->"
                        f"{s1['upper'][1]:.0f} s, lower floors {s0['lower'][1]:.0f}->{s1['lower'][1]:.0f} s).")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    groups = [("sf", "Step-free users"), ("upper", "Upper floors (4-6)"), ("lower", "Lower floors (0-2)")]
    ms = ["safe_shortest", "reactive", "ccrp", "lookahead"]
    col = {"safe_shortest": GREY, "reactive": "#cfcdc4", "ccrp": ORANGE, "lookahead": BLUE}
    w = 0.2
    for j, m in enumerate(ms):
        ax.bar([i + (j - 1.5) * w for i in range(len(groups))], [agg[m][g][1] for g, _ in groups], w,
               color=col[m], label=LABEL[m])
    ax.set_xticks(range(len(groups)), [t for _, t in groups])
    ax.set_ylabel("Group's last person out (s)", color=MUTED, fontsize=9)
    ax.set_title("Worst-off person in each group (lower is fairer)", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(agg[m][g][1] for m in ms for g, _ in groups) * 1.3)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    _style(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig13_fairness.png"), dpi=170)
    plt.close(fig)
    lines += ["", "![Fairness](figures/fig13_fairness.png)", ""]


# ---------------------------------------------------------------- B: ML in the loop
def section_mlloop(A, B, lines, findings):
    scen = [s for s in TITLE if s in set(B.scenario)]
    both = pd.concat([B, A[A.scenario.isin(scen) & A.method.isin(["safe_shortest", "lookahead", "lookahead_ml"])]])
    ms = ["safe_shortest", "pred_formula", "pred_ml", "pred_ml_upper", "lookahead", "lookahead_ml"]
    lines += ["## 3. Does a better predictor give better evacuations?", "",
              "Same re-routing controller, three predictors of edge delay (queue formula / LightGBM / LightGBM 95% "
              "upper quantile), against the capacity-booking planner with formula or ML walk times. 30 seeds.", "",
              "| Scenario | " + " | ".join(LABEL[m] for m in ms) + " |", "|" + " --- |" * (len(ms) + 1)]
    for s in scen:
        cells = []
        for m in ms:
            v = list(both[(both.scenario == s) & (both.method == m)]["last"])
            lo, hi = boot_ci(v)
            cells.append(f"{statistics.mean(v):.0f} [{lo:.0f}-{hi:.0f}]")
        lines.append(f"| {TITLE[s]} | " + " | ".join(cells) + " |")
    lines += ["", "Paired tests (Holm-corrected over scenarios):", ""]
    for a, b in (("pred_formula", "pred_ml"), ("pred_ml", "pred_ml_upper"), ("lookahead", "lookahead_ml"),
                 ("pred_ml", "lookahead")):
        ds = [_paired(both, s, a, b) for s in scen]
        ps = _holm([wilcoxon_p(d) for d, _ in ds])
        parts = [f"{TITLE[s]} {statistics.mean(d) / statistics.mean(base):+.1%} (p {_fmt_p(p)})"
                 for s, (d, base), p in zip(scen, ds, ps)]
        lines.append(f"- {LABEL[b]} vs {LABEL[a]}: " + "; ".join(parts))
    mean = lambda m: both[both.method == m]["last"].mean()
    gain_pred = (mean("pred_formula") - mean("pred_ml")) / mean("pred_formula")
    gain_coord = (mean("pred_ml") - mean("lookahead")) / mean("pred_ml")
    gain_mlwalk = (mean("lookahead") - mean("lookahead_ml")) / mean("lookahead")
    findings.append(f"ML vs coordination: swapping the queue formula for the ML predictor changed mean last-out by "
                    f"{-gain_pred:+.1%}; adding ML walk times to the coordinated planner {-gain_mlwalk:+.1%}; but "
                    f"coordinating (capacity booking) instead of individually re-routing on the best prediction "
                    f"changed it by {-gain_coord:+.1%}. **The gain comes from coordination, not from prediction accuracy.**")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.3))
    col = {"safe_shortest": GREY, "pred_formula": "#cfcdc4", "pred_ml": ORANGE, "pred_ml_upper": "#f0b77a",
           "lookahead": BLUE, "lookahead_ml": "#7aa6e0"}
    w = 0.13
    for j, m in enumerate(ms):
        vals = [both[(both.scenario == s) & (both.method == m)]["last"].mean() for s in scen]
        ax.bar([i + (j - 2.5) * w for i in range(len(scen))], vals, w, color=col[m], label=LABEL[m])
    ax.set_xticks(range(len(scen)), [TITLE[s] for s in scen], fontsize=8)
    ax.set_ylabel("Mean last-out (s)", color=MUTED, fontsize=9)
    ax.set_title("Prediction quality matters little; coordination matters a lot", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    _style(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig14_mlloop.png"), dpi=170)
    plt.close(fig)
    lines += ["", "![ML in the loop](figures/fig14_mlloop.png)", ""]


# ---------------------------------------------------------------- C, D: GA and lower bound
def section_ga(A, Cdf, D, lines, findings):
    lines += ["## 4. Offline benchmark: genetic algorithm", "",
              "A genetic algorithm searches exit assignments (one exit per room-subgroup) by running the full "
              "simulation for every candidate - the kind of optimiser used in published evacuation work "
              "(e.g. Abdelghany et al. 2014). It needs hundreds of simulations, so it is an offline benchmark, "
              "not something that can run during a fire.", "",
              "| Scenario | Seed | GA last-out (s) | GA simulations | GA compute (s) | Look-ahead last-out (s) | Look-ahead compute (s) | Shortest (s) | Lower bound (s) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rel = []
    for _, r in Cdf.iterrows():
        la = A[(A.scenario == r.scenario) & (A.method == "lookahead") & (A.seed == r.seed)]
        ss = A[(A.scenario == r.scenario) & (A.method == "safe_shortest") & (A.seed == r.seed)]
        ob = D[(D.scenario == r.scenario) & (D.seed == r.seed)]["evac_time_last_s"] if D is not None else []
        la_t, la_c = (float(la["last"].iloc[0]), float(la["compute_s"].iloc[0])) if len(la) else (None, None)
        lines.append(f"| {TITLE[r.scenario]} | {r.seed} | {r.evac_time_last_s:.0f} | {r.ga_evaluations} | "
                     f"{r.compute_s:.0f} | {la_t:.0f} | {la_c:.1f} | {float(ss['last'].iloc[0]):.0f} | "
                     f"{(float(ob.iloc[0]) if len(ob) else float('nan')):.0f} |")
        if la_t:
            rel.append((la_t - r.evac_time_last_s) / r.evac_time_last_s)
    if rel:
        parts, ga_better = [], []
        for sc in dict.fromkeys(Cdf.scenario):
            g = Cdf[Cdf.scenario == sc]
            la = A[(A.scenario == sc) & (A.method == "lookahead") & A.seed.isin(g.seed)]["last"].mean()
            ga = g.evac_time_last_s.mean()
            parts.append(f"{TITLE[sc]} GA {ga:.0f} s vs look-ahead {la:.0f} s")
            if ga < la * 0.98:
                ga_better.append(TITLE[sc])
        verdict = ("beat look-ahead only in: " + ", ".join(ga_better)) if ga_better else "never beat look-ahead"
        findings.append(f"A genetic algorithm with {int(Cdf.ga_evaluations.max())} full simulations per case "
                        f"(~{Cdf.compute_s.mean() / 60:.0f} min, offline) {verdict} (look-ahead: ~2 s, live): "
                        + "; ".join(parts) + ". Optimising exit choice alone cannot balance stairwells; in a single "
                        "hall with several doors it can, but only offline.")
    lines += ["", "Why the GA can lose badly on a full floor: its genome chooses only the EXIT for each group of people; "
              "the route to that exit (and so the stairwell) is still the shortest one, and everyone leaves as soon as "
              "they are ready. Look-ahead chooses the whole route and, through capacity booking, the timing. When the "
              "bottleneck is the stairwells rather than the exits (200 people on one floor), exit choice alone cannot "
              "balance them. A richer genome (stairwell + exit) would close some of this gap at a higher search cost.", ""]


# ---------------------------------------------------------------- E: tipping point
def section_tipping(E, lines, findings):
    E = E.copy()
    E["n"] = E.tag.str.replace("n=", "").astype(int)
    ns = sorted(E.n.unique())
    ms = ["safe_shortest", "reactive", "ccrp", "lookahead"]
    lines += ["## 5. Tipping point: at what crowd size does coordination start to matter?", "",
              "200-person floor-3 scenario with the head-count varied (10 seeds each). Mean last-out (s):", "",
              "| People | " + " | ".join(LABEL[m] for m in ms) + " | Look-ahead advantage | p |",
              "|" + " --- |" * (len(ms) + 3)]
    adv = []
    for n in ns:
        x = E[E.n == n]
        means = [x[x.method == m]["last"].mean() for m in ms]
        sub = x.assign(scenario=str(n))
        d, base = _paired(sub, str(n), "safe_shortest", "lookahead")
        a = -statistics.mean(d) / statistics.mean(base)
        adv.append((n, a, wilcoxon_p(d)))
        lines.append(f"| {n} | " + " | ".join(f"{v:.0f}" for v in means) + f" | {a:.0%} | {_fmt_p(adv[-1][2])} |")
    tip = next((n for n, a, p in adv if a >= 0.10 and p is not None and p < 0.05), None)
    findings.append(f"Tipping point: on one floor, coordination gives a significant >=10% gain from about "
                    f"**{tip} people** upward" if tip else "Tipping point: no crowd size gave a significant >=10% gain.")
    findings[-1] += "; below that, shortest path is already as good (there are no queues to manage)."
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.3))
    col = {"safe_shortest": GREY, "reactive": "#cfcdc4", "ccrp": ORANGE, "lookahead": BLUE}
    for m in ms:
        ax[0].plot(ns, [E[(E.n == n) & (E.method == m)]["last"].mean() for n in ns], "o-", color=col[m],
                   label=LABEL[m], ms=4)
    ax[0].set(xlabel="People on floor 3", ylabel="Mean last-out (s)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Evacuation time vs crowd size", loc="left", fontsize=11, fontweight="bold")
    ax[1].plot(ns, [a * 100 for _, a, _ in adv], "o-", color=BLUE)
    ax[1].axhline(0, color=INK, lw=1)
    ax[1].set(xlabel="People on floor 3", ylabel="Look-ahead gain over shortest (%)")
    ax[1].set_title("Coordination pays off once queues form", loc="left", fontsize=11, fontweight="bold")
    for a in ax:
        _style(a, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig15_tipping.png"), dpi=170)
    plt.close(fig)
    lines += ["", "![Tipping point](figures/fig15_tipping.png)", ""]


# ---------------------------------------------------------------- G: data freshness
def section_fresh(G_, A, lines, findings):
    G_ = G_.copy()
    G_["r"] = G_.tag.str.replace("replan=", "").astype(float)
    rs = sorted(G_.r.unique())
    scen = [s for s in TITLE if s in set(G_.scenario)]
    lines += ["## 6. How fresh must the crowd data be?", "",
              "Look-ahead re-planning with the crowd state refreshed every 2-80 s (10 seeds). In a real building "
              "this is the update interval of the occupancy sensors / camera counts. Mean last-out (s) and exposed people:", "",
              "| Re-plan interval (s) | " + " | ".join(TITLE[s] for s in scen) + " |", "|" + " --- |" * (len(scen) + 1)]
    for r in rs:
        cells = []
        for s in scen:
            x = G_[(G_.r == r) & (G_.scenario == s)]
            cells.append(f"{x['last'].mean():.0f} s, {x['exposed_agents'].mean():.1f} exposed")
        lines.append(f"| {r:g} | " + " | ".join(cells) + " |")
    def stats(r, s):
        x = G_[(G_.r == r) & (G_.scenario == s)]
        return x["last"].mean(), x["exposed_agents"].mean()
    ok = {}
    for s in scen:
        t0, e0 = stats(min(rs), s)
        good = [r for r in rs if stats(r, s)[0] <= t0 * 1.05 and stats(r, s)[1] <= e0 * 1.25 + 2]
        ok[s] = max(r for r in rs if all(rr in good for rr in rs if rr <= r))
    findings.append("Data freshness: keeping last-out within 5% and exposure within 25% of the 2-second case needs crowd "
                    "data refreshed every " + "; ".join(f"**{ok[s]:g} s** ({TITLE[s]})" for s in scen) +
                    ". When the fire is growing, stale data mostly costs SAFETY (exposure rises steeply), not time.")
    lines += ["", "Acceptable refresh interval (last-out within 5% and exposure within 25% + 2 people of the 2 s case): " +
              "; ".join(f"{TITLE[s]} {ok[s]:g} s" for s in scen) + "."]
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.3))
    for s, c in zip(scen, (BLUE, ORANGE)):
        axs[0].plot(rs, [stats(r, s)[0] for r in rs], "o-", color=c, label=TITLE[s])
        axs[1].plot(rs, [stats(r, s)[1] for r in rs], "o-", color=c, label=TITLE[s])
    for ax, yl, t in ((axs[0], "Mean last-out (s)", "Time barely changes..."),
                      (axs[1], "Mean people exposed", "...but stale data costs safety in a growing fire")):
        ax.set_xscale("log")
        ax.set_xticks(rs, [f"{r:g}" for r in rs])
        ax.minorticks_off()
        ax.set(xlabel="Crowd-data refresh / re-plan interval (s)", ylabel=yl)
        ax.set_title(t, loc="left", fontsize=11, fontweight="bold")
        _style(ax, "y")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig16_freshness.png"), dpi=170)
    plt.close(fig)
    lines += ["", "![Freshness](figures/fig16_freshness.png)", ""]


# ---------------------------------------------------------------- H: ablations
def section_ablation(H, lines, findings):
    scen = [s for s in TITLE if s in set(H.scenario)]
    tags = list(dict.fromkeys(H.tag))
    lines += ["## 7. Ablations: remove one ingredient at a time", "",
              "Look-ahead planner with one ingredient switched off (10 seeds per scenario). Each cell: mean last-out (s) / "
              "mean exposed people / mean route changes, and change in last-out vs the full planner.", "",
              "| Variant | " + " | ".join(TITLE[s] for s in scen) + " |", "|" + " --- |" * (len(scen) + 1)]
    full = {s: H[(H.tag == "full") & (H.scenario == s)] for s in scen}
    effect = {}
    for t in tags:
        cells, rels = [], []
        for s in scen:
            x = H[(H.tag == t) & (H.scenario == s)]
            rel = (x["last"].mean() - full[s]["last"].mean()) / full[s]["last"].mean()
            rels.append(rel)
            cells.append(f"{x['last'].mean():.0f} s / {x['exposed_agents'].mean():.1f} exp / "
                         f"{x['route_changes'].mean():.0f} chg ({rel:+.0%})")
        effect[t] = rels
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    worst = sorted(((max(v), t) for t, v in effect.items() if t != "full"), reverse=True)[:2]
    findings.append("Ablations: the ingredients that matter most are " +
                    " and ".join(f"'{t}' (up to {v:+.0%} slower when removed)" for v, t in worst) + ".")
    if "no switching threshold" in effect:
        x = H[H.tag == "no switching threshold"]
        f_ = H[H.tag == "full"]
        findings.append(f"The 15% switching threshold is a deliberate trade: without it the planner is "
                        f"{-statistics.mean(effect['no switching threshold']):.0%} faster on average but issues "
                        f"{x['route_changes'].mean() / max(1, f_['route_changes'].mean()):.0f}x as many route changes "
                        f"({f_['route_changes'].mean():.0f} -> {x['route_changes'].mean():.0f} per run) - instructions "
                        "that real people may not follow when signs keep flipping.")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ts = [t for t in tags if t != "full"]
    col = (BLUE, ORANGE, GREY)
    w = 0.25
    for j, s in enumerate(scen):
        ax.barh([i + (j - 1) * w for i in range(len(ts))], [effect[t][j] * 100 for t in ts], w, color=col[j],
                label=TITLE[s])
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(ts)), ts, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Change in last-out vs full planner (%)  (right = removing it hurts)", color=MUTED, fontsize=9)
    ax.set_title("What each part of the planner is worth", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig17_ablation.png"), dpi=170)
    plt.close(fig)
    lines += ["", "![Ablations](figures/fig17_ablation.png)", ""]


LIMITATIONS = """## Limitations (stated openly)

1. **Simulated, not observed in this building.** All results come from our own discrete-time simulator. Its
   speed-density curve and door flow are CALIBRATED on real Juelich crowd experiments (`docs/calibration.md`),
   but those were students in a laboratory, and no drill has been timed in this building. Absolute times are
   therefore estimates; the *ranking* of methods is what the sensitivity analysis and both parameter sets
   (calibrated and handbook, `docs/archive_handbook_params/`) show to be robust.
2. **Two building geometries.** The framework is building-agnostic (`build_graph`); this report covers the
   campus, `docs/hospital_results.md` the hospital.
3. **Agents are simple.** No group behaviour beyond herding toward the crowd, no counter-flow of responders, no
   panic model, no physical pushing; compliance is a fixed probability per person.
4. **Fire and smoke are geometric.** Fire spreads as a growing radius and smoke climbs stair shafts on a timer;
   there is no CFD smoke model (e.g. FDS). Exposure = entering a node above the hazard threshold. Smoke slows
   walking (Frantzich & Nilsson 2003), but the mapping from our 0-1 hazard index to smoke density is assumed.
5. **Perfect sensing is assumed** for the planner (it knows everyone's position at each re-plan). Section 6 tests
   *staleness* of that data but not noise or missing counts.
6. **Lifts** are used only for step-free users without fire; lift dispatching is not modelled in detail.
7. **The ML delay model (v3) is trained on simulated crossings.** It is tested on real Juelich crossings
   (`docs/ml_v3.md`): about as accurate as the calibrated formula overall, better in queues, and its uncertainty
   range needed widening on real data. It has not been tested on real sensor data from a building.
8. **The theoretical lower bound** relaxes walking speeds to free-flow and discretises time (2 s steps), so it is
   a bound, not an achievable plan.
"""


def main():
    os.makedirs(FIGS, exist_ok=True)
    A, B, Cdf, D, E, G_, H = (_load(p) for p in "ABCDEGH")
    if A is None:
        sys.exit("results/final_A.csv missing - run: python -m evac.final_experiments")
    lines, findings = [], []
    section_main(A, D, lines, findings)
    section_fairness(A, lines, findings)
    if B is not None:
        section_mlloop(A, B, lines, findings)
    if Cdf is not None:
        section_ga(A, Cdf, D, lines, findings)
    if E is not None:
        section_tipping(E, lines, findings)
    if G_ is not None:
        section_fresh(G_, A, lines, findings)
    if H is not None:
        section_ablation(H, lines, findings)
    n_runs = sum(len(x) for x in (A, B, Cdf, D, E, G_, H) if x is not None)
    head = ["# Final results", "",
            f"Generated by `python -m evac.final_report` from {n_runs} simulation runs in `results/final_*.csv`. "
            "Every number below can be regenerated with `python -m evac.final_experiments` (seeded, deterministic). "
            "Simulator parameters: CALIBRATED on Juelich experiments, smoke slows walking (Improving-ML session; "
            "`docs/calibration.md`); ML predictor = v3 (`docs/ml_v3.md`). Results with the earlier handbook values: "
            "`docs/archive_handbook_params/final_results.md`.",
            "", "## Key findings", ""] + [f"{i}. {f}" for i, f in enumerate(findings, 1)] + [
            "", "Related analyses: [calibration](calibration.md) (fit to real crowd data), [ML v3](ml_v3.md) "
            "(the learned crossing-time model, tested on real data), [sensitivity](sensitivity.md) (robustness to physical assumptions), "
            "[tuning](tuning.md) (Bayesian optimisation of planner settings), [weak spots](weakspots.md).", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(head + lines + [LIMITATIONS]))
    print("\n".join(head))
    print(f"-> {DOC}")


if __name__ == "__main__":
    main()
