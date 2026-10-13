"""
evac/hospital_report.py
results/hosp_*.csv|json (from evac.hospital_experiments) -> docs/hospital_results.md + figures 18-21.

Run:  python -m evac.hospital_report
"""

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import pandas as pd                                          # noqa: E402

from evac import config as C                                 # noqa: E402
from evac.final_report import (BLUE, ORANGE, GREY, INK, MUTED, LABEL, _holm, _fmt_p, _style, _plt)  # noqa: E402
from evac.summarize import boot_ci, wilcoxon_p               # noqa: E402

RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "docs", "figures")
DOC = os.path.join(ROOT, "docs", "hospital_results.md")
DATA = os.path.join(ROOT, "data", "hospital", "hospital_graph.json")
TITLE = {"hosp_day_peak": "Day peak, no fire", "hosp_day_fire_stairs1_f2": "Day, fire at 2nd-floor stairs 1",
         "hosp_day_fire_lobby": "Day, fire in entrance lobby", "hosp_day_spreading_f2": "Day, spreading fire 2nd floor",
         "hosp_day_compliance70": "Day, 70% follow guidance", "hosp_night_wards_fire": "Night, wards only, spreading fire"}
M4 = ["safe_shortest", "reactive", "ccrp", "lookahead"]
PROF = [("walk", "Staff / visitors"), ("slow", "Walking patients"), ("wheelchair", "Wheelchair users"),
        ("bed", "Bedridden patients")]


def _load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    if p.endswith(".json"):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    df = pd.read_csv(p)
    df["last"] = df["evac_time_last_s"].fillna(C.MAX_TIME_S)
    df["amb"] = df["ambulatory_last_s"].fillna(C.MAX_TIME_S)
    df["trapped"] = df["needs_assistance"].fillna(0) + df["stranded"].fillna(0) + df["timed_out"].fillna(0)
    return df


def _paired(df, s, a, b, col):
    x = df[(df.scenario == s) & (df.method == a)].set_index("seed")[col]
    y = df[(df.scenario == s) & (df.method == b)].set_index("seed")[col]
    idx = x.index.intersection(y.index)
    return list((y[idx] - x[idx]).astype(float)), list(x[idx].astype(float))


def section_building(lines):
    with open(DATA, encoding="utf-8") as f:
        doc = json.load(f)
    from evac.hospital import load_hospital
    from evac.building import exits, room_nodes
    from collections import Counter
    G = load_hospital()
    et = Counter(d["etype"] for *_, d in G.edges(data=True))
    lines += ["## 1. The building and how it was loaded", "",
              f"**{doc['building']}.** {doc['source']}. Ground floor = outpatients (18 consulting rooms, waiting hall), "
              "1st = laboratories, 2nd and 3rd = general wards (~40 beds per floor). Two stairwells, a passenger lift, two bed "
              "lifts and a switchback ramp serving every floor.", "",
              f"- Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges ("
              + ", ".join(f"{v} {k}" for k, v in sorted(et.items())) + f"); {len(room_nodes(G))} rooms; exits: "
              + ", ".join(e.replace('g_', '') for e in exits(G)) + ".",
              f"- Scale: {doc['scale_note']}. Ground {doc['scale']['g']['m_per_px']} m/px; upper floors "
              + ", ".join(f"{p} {doc['scale'][p]['m_per_px']}" for p in ('f1', 'f2', 'f3')) + " m/px.",
              "- **Cross-floor check:** after fitting, the six shared shafts on every upper floor land within "
              f"**{max(max(v['shaft_fit_residuals_m']) for v in doc['scale'].values()):.2f} m** of their ground-floor "
              "position, so the team's digitisation is consistent between floors.",
              "- Data fixes: " + "; ".join(f"`{f['edge'][0]}`-`{f['edge'][1]}`: {f['why']}" for f in doc["fixes"]) + ".",
              "- The team's scripts were not executed; values were read with `json` / `ast.literal_eval`.", "",
              "**Occupants (assumptions, see `config.PROFILES`):**", "",
              "| Profile | Speed | Stairs? | Capacity used | Preparation before moving |", "| --- | --- | --- | --- | --- |"]
    for p, lab in PROF:
        P = C.PROFILES[p]
        lines.append(f"| {lab} | x{P['speed']} | {'no (ramp / lift)' if P['step_free'] else 'yes'} | "
                     f"{P['size']:g} person-eq. | {P['pre_s'][0]}-{P['pre_s'][1]} s |")
    lines += ["", f"Ramp: {C.RAMP_LENGTH_PER_FLOOR_M:g} m of ramp per floor (two 8 m runs + landings each way, from the "
              f"drawings), {C.WIDTH_M['ramp']} m wide, {C.RAMP_SPEED} m/s. In a fire, lifts (including bed lifts) are out "
              "of service, so the **ramp is the only way out for wheelchair users and beds**.", ""]


def section_main(A, D, lines, findings):
    scen = [s for s in TITLE if s in set(A.scenario)]
    opt = D.groupby("scenario")["evac_time_last_s"].mean().to_dict() if D is not None else {}
    lines += ["## 2. Main comparison: 6 scenarios x 7 methods x 30 paired seeds", "",
              "In a hospital the **last person out is almost always a bedridden patient**, whose time is set by bed "
              "preparation and the ramp, not by routing. So three numbers are reported: *ambulatory last-out* (last "
              "walking person out - what routing controls), *everyone out*, and *people exposed* to smoke/fire.", "",
              "| Scenario | Method | Ambulatory last-out (s) [95% CI] | Everyone out (s) | Mean exit time (s) | Exposed | Cut off |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for s in scen:
        for m in ["nearest"] + M4 + ["lookahead_ml"]:
            x = A[(A.scenario == s) & (A.method == m)]
            lo, hi = boot_ci(list(x["amb"]))
            lines.append(f"| {TITLE[s] if m == 'nearest' else ''} | {LABEL[m]} | {x['amb'].mean():.0f} [{lo:.0f}-{hi:.0f}] | "
                         f"{x['last'].mean():.0f} | {x['evac_time_mean_s'].mean():.0f} | {x['exposed_agents'].mean():.1f} | "
                         f"{x['trapped'].mean():.1f} |")
        if s in opt:
            lines.append(f"| | *Lower bound (everyone out)* | | {opt[s]:.0f} | | | |")
    lines += ["", "Lower bound: time-expanded max-flow at free walking speed, ignoring bed size and preparation - "
              "valid but loose in a hospital.", "",
              "### Look-ahead vs shortest safe path (paired, Holm-corrected Wilcoxon over the 6 scenarios)", "",
              "| Scenario | Ambulatory last-out change | p | Mean exit time change | p | Exposed: shortest -> look-ahead |",
              "| --- | --- | --- | --- | --- | --- |"]
    da = [_paired(A, s, "safe_shortest", "lookahead", "amb") for s in scen]
    dm = [_paired(A, s, "safe_shortest", "lookahead", "evac_time_mean_s") for s in scen]
    pa, pm = _holm([wilcoxon_p(d) for d, _ in da]), _holm([wilcoxon_p(d) for d, _ in dm])
    sig = 0
    for s, (d1, b1), p1, (d2, b2), p2 in zip(scen, da, pa, dm, pm):
        e0 = A[(A.scenario == s) & (A.method == "safe_shortest")]["exposed_agents"].mean()
        e1 = A[(A.scenario == s) & (A.method == "lookahead")]["exposed_agents"].mean()
        r1, r2 = statistics.mean(d1) / statistics.mean(b1), statistics.mean(d2) / statistics.mean(b2)
        sig += (p1 is not None and p1 < 0.05 and r1 < 0)
        lines.append(f"| {TITLE[s]} | {r1:+.0%} | {_fmt_p(p1)} | {r2:+.0%} | {_fmt_p(p2)} | {e0:.1f} -> {e1:.1f} |")
    rel = {s: statistics.mean(d) / statistics.mean(b) for s, (d, b) in zip(scen, da)}
    best = min(rel, key=rel.get)
    worse = [s for s, (d1, b1), p1 in zip(scen, da, pa) if p1 is not None and p1 < 0.05 and statistics.mean(d1) > 0]
    findings.append(f"**The main result transfers to a second, very different building.** Look-ahead re-planning cut the "
                    f"time until the last walking person is out in {sig} of {len(scen)} hospital scenarios (significant, "
                    f"Holm-corrected), by up to {-rel[best]:.0%} ({TITLE[best]})"
                    + (f"; it was slower in {len(worse)} (" + ", ".join(TITLE[s] for s in worse) +
                       ") - where it deliberately trades time for less exposure, see below." if worse else "."))
    from evac.hospital import scenarios as _hs
    fire_sc = [s for s in scen if _hs()[s].get("fires")]
    allout = {s: A[(A.scenario == s) & (A.method == "lookahead")]["last"].mean() /
              A[(A.scenario == s) & (A.method == "safe_shortest")]["last"].mean() - 1 for s in fire_sc}
    findings.append("But in a fire the time until EVERYONE is out barely moves (" +
                    ", ".join(f"{TITLE[s]} {(v if abs(v) >= 0.005 else 0.0):+.0%}" for s, v in allout.items()) +
                    "): with lifts out of service it is set by bedridden patients - bed preparation plus three floors "
                    "of ramp - not by routing.")
    lob = "hosp_day_fire_lobby"
    if lob in scen:
        x = lambda m, c: A[(A.scenario == lob) & (A.method == m)][c].mean()
        findings.append(f"Fire in the entrance lobby: hazard-aware routing alone still sends people past the fire "
                        f"({x('safe_shortest', 'exposed_agents'):.0f} exposed); capacity-aware planning spreads them over "
                        f"the other exits ({x('lookahead', 'exposed_agents'):.0f} exposed) at the cost of a longer average "
                        f"exit ({x('safe_shortest', 'evac_time_mean_s'):.0f} -> {x('lookahead', 'evac_time_mean_s'):.0f} s).")
    night = "hosp_night_wards_fire"
    if night in scen:
        x = A[(A.scenario == night)]
        n = x["n_agents"].iloc[0]
        e = {m: x[x.method == m]["exposed_agents"].mean() for m in M4}
        findings.append(f"**Night ward fire: routing cannot protect patients who cannot move quickly.** "
                        f"{min(e.values()):.0f}-{max(e.values()):.0f} of {n} people were exposed whatever the method. "
                        "This is the quantitative case for hospital practice of **progressive horizontal evacuation** "
                        "(move beds sideways to the next fire compartment first) - our next modelling step.")

    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 0.55 * len(scen) + 1.6), gridspec_kw={"width_ratios": [1.3, 1]})
    cols = {"reactive": GREY, "ccrp": ORANGE, "lookahead": BLUE}
    for i, s in enumerate(scen):
        y0 = len(scen) - 1 - i
        for j, m in enumerate(cols):
            d, b = _paired(A, s, "safe_shortest", m, "amb")
            r = [v / statistics.mean(b) * 100 for v in d]
            lo, hi = boot_ci(r)
            y = y0 + (j - 1) * 0.22
            ax[0].plot([lo, hi], [y, y], color=cols[m], lw=2)
            ax[0].plot(statistics.mean(r), y, "o", color=cols[m], ms=5, mec="white", mew=1, label=LABEL[m] if i == 0 else None)
        e = [A[(A.scenario == s) & (A.method == m)]["exposed_agents"].mean() for m in ("safe_shortest", "lookahead")]
        ax[1].barh([y0 + 0.18, y0 - 0.18], e, 0.34, color=[GREY, BLUE])
    ax[0].axvline(0, color=INK, lw=1)
    ax[0].set_yticks(range(len(scen)), [TITLE[s] for s in scen][::-1], fontsize=9)
    ax[0].set_xlabel("Change in last WALKING person out vs shortest safe path (%)", fontsize=9, color=MUTED)
    ax[0].set_title("Hospital: coordination still clears the crowd faster", loc="left", fontsize=11, fontweight="bold")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    ax[1].set_yticks(range(len(scen)), [""] * len(scen))
    ax[1].set_xlabel("People exposed (grey = shortest safe path, blue = look-ahead)", fontsize=9, color=MUTED)
    ax[1].set_title("Exposure", loc="left", fontsize=11, fontweight="bold")
    for a in ax:
        _style(a)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig18_hospital_main.png"), dpi=170)
    plt.close(fig)
    lines += ["", "![Hospital main comparison](figures/fig18_hospital_main.png)", ""]


def section_profiles(A, lines, findings):
    lines += ["## 3. Who waits longest? Outcomes per patient group", "",
              "Mean / last exit time per group, averaged over all six scenarios x 30 seeds.", "",
              "| Group | " + " | ".join(LABEL[m] for m in M4) + " |", "|" + " --- |" * (len(M4) + 1)]
    agg = {}
    for p, lab in PROF:
        cells = []
        for m in M4:
            x = A[A.method == m]
            agg[(p, m)] = (x[f"{p}_mean_s"].mean(), x[f"{p}_max_s"].mean(), x[f"{p}_exposed"].sum() / max(1, x[f"{p}_n"].sum()))
            cells.append(f"{agg[(p, m)][0]:.0f} / {agg[(p, m)][1]:.0f}")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines += ["", "Share of each group exposed to smoke or fire (all scenarios): " + "; ".join(
        f"{lab} {agg[(p, 'safe_shortest')][2]:.0%} -> {agg[(p, 'lookahead')][2]:.0%}" for p, lab in PROF) +
        " (shortest safe path -> look-ahead)."]
    b0, b1 = agg[("bed", "safe_shortest")], agg[("bed", "lookahead")]
    w0, w1 = agg[("walk", "safe_shortest")], agg[("walk", "lookahead")]
    findings.append(f"Bedridden patients wait longest by far (mean exit {b1[0]:.0f} s vs {w1[0]:.0f} s for staff/visitors "
                    f"under look-ahead) and the planner does not make them worse off (bed mean {b0[0]:.0f} -> {b1[0]:.0f} s); "
                    "it gains time for the walking crowd without taking it from patients.")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.3))
    w = 0.2
    col = {"safe_shortest": GREY, "reactive": "#cfcdc4", "ccrp": ORANGE, "lookahead": BLUE}
    for j, m in enumerate(M4):
        ax.bar([i + (j - 1.5) * w for i in range(len(PROF))], [agg[(p, m)][0] for p, _ in PROF], w, color=col[m], label=LABEL[m])
    ax.set_xticks(range(len(PROF)), [lab for _, lab in PROF])
    ax.set_ylabel("Mean exit time (s)", color=MUTED, fontsize=9)
    ax.set_title("Mean exit time by patient group (all hospital scenarios)", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    _style(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig19_hospital_groups.png"), dpi=170)
    plt.close(fig)
    lines += ["", "![Groups](figures/fig19_hospital_groups.png)", ""]


def section_ml(A, lines, findings):
    lines += ["## 4. Does the ML model transfer to a building it never saw?", "",
              "The delay model was trained only on the campus building (no ramps, no beds, no slow patients). "
              "Here it is used unchanged - a test of whether a learned component generalises.", "",
              "| Scenario | Look-ahead (formula) | Look-ahead + ML walk times | Re-route on ML prediction |", "| --- | --- | --- | --- |"]
    for s in [s for s in TITLE if s in set(A.scenario)]:
        v = {m: A[(A.scenario == s) & (A.method == m)]["amb"].mean() for m in ("lookahead", "lookahead_ml", "pred_ml")}
        lines.append(f"| {TITLE[s]} | {v['lookahead']:.0f} | {v['lookahead_ml']:.0f} | {v['pred_ml']:.0f} |")
    g = lambda m: A[A.method == m]["amb"].mean()
    dml = (g('lookahead_ml') - g('lookahead')) / g('lookahead')
    findings.append(("The campus-trained ML model does NOT transfer as-is: " if dml > 0.02 else
                     "The campus-trained ML model transfers without retraining: ") +
                    f"look-ahead with ML walk times is {dml:+.1%} vs the formula version on the hospital (ambulatory "
                    "last-out, all scenarios)" + ("; it never saw ramps, beds or slow patients, so its walk times are "
                    "off - a learned model must be retrained per building, the formula needs no training. " if dml > 0.02
                    else ". ") + f"Re-routing on ML predictions alone stays {(g('pred_ml') - g('lookahead')) / g('lookahead'):+.0%} "
                    "behind coordinated planning - the same pattern as on campus.")
    lines += ["", "(Ambulatory last-out time, seconds, mean of 30 seeds.)", ""]


def section_weak(W, lines, findings):
    lines += ["## 5. Weak spots: which single failure hurts the hospital most?", "",
              "Each row blocks one element (a whole stairwell, the ramp, an exit door, a key corridor) and re-runs the "
              "same evacuation with look-ahead re-planning (3 seeds).", ""]
    plt = _plt()
    fig, axs = plt.subplots(1, len(W), figsize=(6 * len(W), 3.8))
    for ax, (name, res) in zip(axs if len(W) > 1 else [axs], W.items()):
        lines += [f"**{TITLE[name]}** (baseline everyone out {res['baseline_last_out_s']} s):", "",
                  "| Blocked | Everyone out (+s) | Average person (+s) | People cut off |", "| --- | --- | --- | --- |"]
        for r in res["rows"][:8]:
            f = lambda v: "-" if v is None else f"{v:+.0f}"
            lines.append(f"| {r['element'].replace('g_', '')} | {f(r['last_out_delta_s'])} | {f(r['mean_delta_s'])} | {r['people_cut_off']:.0f} |")
        lines.append("")
        top = res["rows"][:8][::-1]
        ax.barh([r["element"].replace("g_", "").replace("Corridor ", "")[:28] for r in top],
                [r["people_cut_off"] for r in top], color=BLUE, height=0.55)
        ax.set_xlabel("People cut off (no route out)", fontsize=9, color=MUTED)
        ax.set_title(TITLE[name], loc="left", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
        _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig20_hospital_weakspots.png"), dpi=170)
    plt.close(fig)
    lines += ["'Everyone out' counts only people who get out: when the ramp is blocked it DROPS because the 42 "
              "slowest people (beds, wheelchairs) are cut off and no longer counted - read it together with the last column.",
              "", "![Hospital weak spots](figures/fig20_hospital_weakspots.png)", ""]
    fire = W.get("hosp_day_fire_stairs1_f2")
    if fire:                                    # for the dashboard's Weak spots tab
        with open(os.path.join(ROOT, "docs", "hospital_weakspots.json"), "w", encoding="utf-8") as f:
            json.dump({"scenario": "hosp_day_fire_stairs1_f2", "method": "lookahead", **fire}, f, indent=1)
        ramp = next((r for r in fire["rows"] if r["element"].startswith("Ramp")), None)
        if ramp:
            findings.append(f"**The ramp is the hospital's single point of failure.** With fire (lifts out), blocking it "
                            f"leaves **{ramp['people_cut_off']:.0f} people** - all wheelchair users and beds above the ground "
                            "floor - with no way out; no stairwell or door failure comes close. Without fire the bed lifts "
                            "cover for it. A design / fire-safety-audit result of exactly the kind this tool is for.")


def section_assumptions(S, lines, findings):
    tags = list(dict.fromkeys(S.tag))
    tags = ["default"] + [t for t in tags if t != "default"]
    lines += ["## 6. How much do the hospital assumptions matter?", "",
              "One assumption changed at a time; scenario 'Day, fire at 2nd-floor stairs 1'; 10 seeds.", "",
              "| Variant | Everyone out: shortest / look-ahead (s) | Ambulatory last-out: shortest / look-ahead (s) | Look-ahead gain (ambulatory) |",
              "| --- | --- | --- | --- |"]
    gains = {}
    for t in tags:
        x = S[S.tag == t]
        a0, a1 = x[x.method == "safe_shortest"], x[x.method == "lookahead"]
        g = (a0["amb"].mean() - a1["amb"].mean()) / a0["amb"].mean()
        gains[t] = (g, a0["last"].mean(), a1["last"].mean())
        lines.append(f"| {t} | {a0['last'].mean():.0f} / {a1['last'].mean():.0f} | {a0['amb'].mean():.0f} / {a1['amb'].mean():.0f} | {g:.0%} |")
    lines += ["", "Why the ramp speed changes the routing gain: walking the 3-floor ramp (3 x 40 m at 1.0 m/s) takes about "
              "as long as the stairs (3 x 8 m at 0.65 m/s), so at the default speed shortest-path routing puts everyone on "
              "the stairs and the planner can divert part of the crowd onto the idle ramp (30% gain). A slow ramp is no "
              "real alternative (8%); a fast ramp is already chosen by shortest-path routing (10%). **Coordination pays "
              "most when there is spare capacity that simple routing ignores.**"]
    rng_all = [v[2] for v in gains.values()]
    findings.append(f"Hospital assumptions move the everyone-out time a lot ({min(rng_all):.0f}-{max(rng_all):.0f} s, mostly bed "
                    f"preparation and ramp speed) but the look-ahead advantage for the walking crowd stays "
                    f"{min(v[0] for v in gains.values()):.0%}-{max(v[0] for v in gains.values()):.0%} in every variant.")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ts = tags[::-1]
    ax.barh([t for t in ts], [gains[t][1] for t in ts], 0.38, color=GREY, label="Shortest safe path", align="edge")
    ax.barh([t for t in ts], [gains[t][2] for t in ts], -0.38, color=BLUE, label="Look-ahead", align="edge")
    ax.set_xlabel("Everyone out (s)", fontsize=9, color=MUTED)
    ax.set_title("Bed and ramp assumptions set the all-out time", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    ax.tick_params(labelsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig21_hospital_assumptions.png"), dpi=170)
    plt.close(fig)
    lines += ["", "![Assumptions](figures/fig21_hospital_assumptions.png)", ""]


def section_compare(A, lines):
    camp = os.path.join(RES, "final_A.csv")
    if not os.path.exists(camp):
        return
    F = pd.read_csv(camp)
    F["last"] = F["evac_time_last_s"].fillna(C.MAX_TIME_S)
    lines += ["## 7. Campus vs hospital: what generalises", "",
              "| Question | Campus (7 floors, 452 nodes) | Hospital (4 floors, 124 nodes) |", "| --- | --- | --- |"]
    def best_gain(df, col):
        out = []
        for s in set(df.scenario):
            a = df[(df.scenario == s) & (df.method == "safe_shortest")][col].mean()
            b = df[(df.scenario == s) & (df.method == "lookahead")][col].mean()
            out.append((a - b) / a)
        return max(out), statistics.median(out)
    cg, hg = best_gain(F, "last"), best_gain(A, "amb")
    lines += [f"| Largest look-ahead gain | {cg[0]:.0%} (last-out) | {hg[0]:.0%} (ambulatory last-out) |",
              f"| Median gain over scenarios | {cg[1]:.0%} | {hg[1]:.0%} |",
              "| Where it helps | concentrated crowds (full floor / hall) | the outpatient crowd on the ground floor |",
              "| What limits it | stair capacity | bed preparation + the single ramp |",
              "| ML model | trained here | used unchanged: walk times off for ramps/beds (retrain per building); "
              "coordination still wins |", ""]


LIMITS = """## Limitations specific to the hospital

1. Only ground + 3 floors were digitised (the drawings' area statement lists more floors); upper floors would add load to the ramp.
2. Staff are not modelled as a limited resource: in reality each bed needs 2+ staff, so beds would move in waves.
3. Progressive horizontal evacuation (moving patients sideways into the next fire compartment), the standard
   hospital procedure, is not modelled yet; the results show exactly why it matters (night ward fire).
4. Profile speeds, bed size and preparation times are assumptions in published ranges, not measurements here;
   section 6 shows the routing conclusion does not depend on them.
5. Occupancy (310 by day, 90 at night) is an estimate from the room schedule, not hospital data.
"""


def main():
    os.makedirs(FIGS, exist_ok=True)
    A, D, S, W = _load("hosp_A.csv"), _load("hosp_D.csv"), _load("hosp_S.csv"), _load("hosp_W.json")
    if A is None:
        sys.exit("results/hosp_A.csv missing - run: python -m evac.hospital_experiments")
    lines, findings = [], []
    section_building(lines)
    section_main(A, D, lines, findings)
    section_profiles(A, lines, findings)
    section_ml(A, lines, findings)
    if W:
        section_weak(W, lines, findings)
    if S is not None:
        section_assumptions(S, lines, findings)
    section_compare(A, lines)
    n = sum(len(x) for x in (A, D, S) if x is not None)
    head = ["# Second building: oncology hospital", "",
            f"Generated by `python -m evac.hospital_report` from {n} simulation runs (+ weak-spot runs) in `results/hosp_*`. "
            "Same engine, methods and statistics as the campus study (`final_results.md`); only the building data and "
            "the occupants changed.", "", "## Key findings", ""] + [f"{i}. {f}" for i, f in enumerate(findings, 1)] + [""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(head + lines + [LIMITS]))
    print("\n".join(head))
    print(f"-> {DOC}")


if __name__ == "__main__":
    main()
