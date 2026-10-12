"""
evac/plots.py
Report figures from experiment CSVs.

  python -m evac.plots --runs results/session3_runs.csv --optimum results/session3_optimum.csv \
                       --ablation results/session3_ablation.csv --out docs/figures

Figure 1  last-out time per method with 95% CI, theoretical best marked, per scenario
Figure 2  look-ahead horizon ablation (prediction too short is worse than none)
Figure 3  route changes per person (oscillation) per method
"""

import argparse
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from evac.summarize import load, boot_ci   # noqa: E402

BLUE, GREY, INK, MUTED = "#2a78d6", "#a3a29a", "#21201c", "#6b6a63"
ORDER = ["nearest", "safe_shortest", "reactive", "ccrp", "lookahead"]
LABEL = {"nearest": "Nearest exit", "safe_shortest": "Shortest safe path", "reactive": "Reactive rerouting",
         "ccrp": "Capacity booking (CCRP)", "lookahead": "Look-ahead re-planning"}
PROPOSED = {"ccrp", "lookahead"}
TITLES = {"lecture_hall_f1_120": "120 people, one seminar hall",
          "floor3_full_200": "200 people on floor 3",
          "floor3_full_200_fire_stairs1": "200 on floor 3, fire at stairs 1",
          "crowd_250_fire_j19": "250 across building, fire at j19",
          "baseline_150_no_fire": "150 across building, no fire"}


def _style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color="#e6e5df", lw=0.8)
    ax.set_axisbelow(True)


def fig_methods(runs, opt, out, scenarios):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.6 * len(scenarios), 3.4), sharey=True)
    for ax, sc in zip(axes, scenarios):
        ys = list(range(len(ORDER)))[::-1]
        for y, m in zip(ys, ORDER):
            vals = [r["evac_time_last_s"] for r in runs if r["scenario"] == sc and r["method"] == m]
            if not vals:
                continue
            mu, (lo, hi) = statistics.mean(vals), boot_ci(vals)
            c = BLUE if m in PROPOSED else GREY
            ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round")
            ax.plot(mu, y, "o", ms=8, color=c, mec="white", mew=2)
            ax.text(hi + 6, y, f"{mu:.0f} s", va="center", fontsize=9, color=INK)
        ov = [r["evac_time_last_s"] for r in opt if r["scenario"] == sc and r["evac_time_last_s"] is not None]
        if ov:
            b = statistics.mean(ov)
            ax.axvline(b, color=INK, ls=(0, (3, 3)), lw=1)
            ax.text(b + 3, -0.45, f"theoretical best {b:.0f} s", fontsize=8, color=MUTED, ha="left", va="center")
        ax.set_yticks(ys, [LABEL[m] for m in ORDER])
        ax.set_ylim(-0.8, len(ORDER) - 0.5)
        ax.set_title(TITLES.get(sc, sc), fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Time until last person is out (s)", fontsize=9, color=MUTED)
        lo_x = min(ov) * 0.85 if ov else None
        if lo_x:
            ax.set_xlim(left=lo_x)
        _style(ax)
    fig.suptitle("Capacity-aware methods (blue) cut evacuation time when crowds are concentrated",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_horizon(abl, runs, opt, out, scenarios):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.2 * len(scenarios), 3.3))
    for ax, sc in zip(axes, scenarios):
        hs, mus, los, his = [], [], [], []
        for r_sc in sorted({r["scenario"] for r in abl if r["scenario"].startswith(sc + "__H")},
                           key=lambda s: int(s.split("__H")[1])):
            vals = [r["evac_time_last_s"] for r in abl if r["scenario"] == r_sc]
            hs.append(int(r_sc.split("__H")[1]))
            mus.append(statistics.mean(vals))
            lo, hi = boot_ci(vals)
            los.append(lo); his.append(hi)
        ax.fill_between(hs, los, his, color=BLUE, alpha=0.15, lw=0)
        ax.plot(hs, mus, "-o", color=BLUE, lw=2, ms=6, mec="white", mew=1.5)
        ax.text(hs[2], mus[2] + 8, "look-ahead", color=BLUE, fontsize=9, ha="left", va="bottom")
        base = [r["evac_time_last_s"] for r in runs if r["scenario"] == sc and r["method"] == "safe_shortest"]
        if base:
            b = statistics.mean(base)
            ax.axhline(b, color=GREY, lw=1.5)
            ax.text(hs[-1], b - 5, f"no planning: {b:.0f} s", color=MUTED, fontsize=8, va="top", ha="right")
        ov = [r["evac_time_last_s"] for r in opt if r["scenario"] == sc and r["evac_time_last_s"] is not None]
        if ov:
            b = statistics.mean(ov)
            ax.axhline(b, color=INK, ls=(0, (3, 3)), lw=1)
            ax.text(hs[-1], b + 4, f"theoretical best: {b:.0f} s", color=MUTED, fontsize=8, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_xticks(hs, [str(h) for h in hs])
        ax.minorticks_off()
        ax.set_xlabel("How far ahead the planner predicts (s, log scale)", fontsize=9, color=MUTED)
        ax.set_ylabel("Last person out (s)", fontsize=9, color=MUTED)
        ax.set_title(TITLES.get(sc, sc), fontsize=10, color=INK, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color="#e6e5df", lw=0.8)
    fig.suptitle("Predicting too short a time ahead is worse than not planning at all",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_changes(runs, out, sc):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ys = list(range(len(ORDER)))[::-1]
    for y, m in zip(ys, ORDER):
        rs = [r for r in runs if r["scenario"] == sc and r["method"] == m]
        if not rs:
            continue
        per = statistics.mean(r["route_changes"] / r["n_agents"] if "n_agents" in r else r["route_changes"] for r in rs)
        ax.barh(y, max(per, 0.0), height=0.5, color=BLUE if m in PROPOSED else GREY)
        ax.text(max(per, 0.0) + 0.1, y, f"{per:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys, [LABEL[m] for m in ORDER])
    ax.set_xlabel("Route changes per person", fontsize=9, color=MUTED)
    ax.set_title(f"Reactive rerouting flip-flops; look-ahead barely does\n{TITLES.get(sc, sc)}",
                 fontsize=10.5, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/session3_runs.csv")
    ap.add_argument("--optimum", default="results/session3_optimum.csv")
    ap.add_argument("--ablation", default="results/session3_ablation.csv")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(a.out, exist_ok=True)
    runs = load(a.runs)
    for r in runs:
        r["n_agents"] = float(r["n_agents"])
    opt = load(a.optimum) if os.path.exists(a.optimum) else []
    fig_methods(runs, opt, os.path.join(a.out, "fig1_methods.png"),
                ["lecture_hall_f1_120", "floor3_full_200", "floor3_full_200_fire_stairs1"])
    if os.path.exists(a.ablation):
        fig_horizon(load(a.ablation), runs, opt, os.path.join(a.out, "fig2_horizon.png"),
                    ["lecture_hall_f1_120", "floor3_full_200"])
    fig_changes(runs, os.path.join(a.out, "fig3_route_changes.png"), "lecture_hall_f1_120")
    print(f"Figures written to {a.out}")


if __name__ == "__main__":
    main()
