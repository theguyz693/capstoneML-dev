"""
evac/session7.py
Session 7 - making it practical. Resumable experiments (results/s7_<part>.csv) + report.

  K  knowledge   What if the system does not know exactly where people are?
  Z              + one pre-computed plan per fire-alarm zone (knows the fire, not the people)
                 - no live data at all: a PRINTED plan made once from the timetable (evac/fixed_plan.py),
                   also when the timetable is 30-50% wrong
                 - live but imperfect head-counts: people unseen, phantom people, one floor's sensors down
                   (evac/knowledge.py)
  P  procedures  Hospital: full vertical evacuation vs progressive horizontal evacuation (move the fire
                 compartment's patients sideways behind fire doors, everyone else defends in place), with
                 a limited number of staff to move beds

Run:  python -m evac.session7 [--parts K P] [--workers 2]     -> docs/session7_results.md, figures 22-23
      (python -m evac.ml_transfer covers the third Session 7 question: retraining the ML model)
"""

import argparse
import json
import os
import statistics
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
DOC = os.path.join(ROOT, "docs", "session7_results.md")
PLANS = os.path.join(ROOT, "docs", "plans")
FIGS = os.path.join(ROOT, "docs", "figures")
SEEDS = range(10)

K_SCEN = [("campus", "floor3_full_200", 3), ("campus", "lecture_hall_f1_120", 1),
          ("campus", "floor3_200_compliance60_herding", 3), ("campus", "floor3_full_200_fire_stairs1", 3),
          ("campus", "floor3_200_spreading_fire_stairs1", 3),
          ("hospital", "hosp_day_peak", 0), ("hospital", "hosp_day_fire_stairs1_f2", 0),
          ("hospital", "hosp_day_compliance70", 0)]
TITLE = {"floor3_full_200": "Campus: 200 on floor 3", "lecture_hall_f1_120": "Campus: 120 in one hall",
         "floor3_200_compliance60_herding": "Campus: 60% comply + herding",
         "floor3_full_200_fire_stairs1": "Campus: fire at stairs 1",
         "floor3_200_spreading_fire_stairs1": "Campus: spreading fire",
         "hosp_day_peak": "Hospital: day peak", "hosp_day_fire_stairs1_f2": "Hospital: fire at stairs 1 (2nd)",
         "hosp_day_compliance70": "Hospital: 70% comply",
         "hosp_night_wards_fire": "Hospital night, ward fire", "hosp_day_spreading_f2": "Hospital day, ward fire"}

_G = {}


def _scen(building):
    if building == "hospital":
        from evac.hospital import scenarios
        return scenarios()
    with open(os.path.join(HERE, "scenarios.json"), encoding="utf-8") as f:
        return json.load(f)


def _graph(building):
    if building not in _G:
        from evac.building import load_building
        _G[building] = load_building(building)
    return _G[building]


def _job(args):
    part, building, name, tag, method, over, seed = args
    from evac.simulator import run
    sc = {**_scen(building)[name], **over}
    m, recs = run(_graph(building).copy(), sc, method, seed)
    G = _graph(building)
    amb = [r["t_out"] for r in recs if r["t_out"] is not None and r.get("profile") in ("walk", "slow", "step_free")]
    beds = [r["t_out"] for r in recs if r["t_out"] is not None and r.get("profile") == "bed"]
    fire_comp = {G.nodes[f["node"]].get("compartment") for f in sc.get("fires", []) if f.get("node") in G}
    fc = [r for r in recs if G.nodes[r["origin"]].get("compartment") in fire_comp and
          G.nodes[r["origin"]]["floor"] in {G.nodes[f["node"]]["floor"] for f in sc.get("fires", [])}]
    fc_out = [r["t_out"] for r in fc if r["t_out"] is not None]
    fc_cut = sum(1 for r in fc if r["state"] in ("needs_assistance", "stranded"))      # no route from the start
    fc_late = sum(1 for r in fc if r["state"] in ("queued", "moving", "waiting"))       # still inside at 30 min
    return {"part": part, "building": building, "scenario": name, "tag": tag, "method": method, "seed": seed,
            "n": m["n_agents"], "evacuated": m["evacuated"], "last": m["evac_time_last_s"],
            "mean": m["evac_time_mean_s"], "ambulatory_last": max(amb) if amb else None,
            "bed_last": max(beds) if beds else None, "exposed": m["exposed_agents"],
            "in_place": m.get("in_place", 0), "timed_out": m["timed_out"],
            "cut_off": m["needs_assistance"] + m["stranded"], "route_changes": m["route_changes"],
            "fire_comp_n": len(fc),
            # time until everyone in the fire compartment who CAN move is safe (still inside at 30 min = 1800 s)
            "fire_comp_safe_last": (1800.0 if fc_late else max(fc_out)) if fc_out else None,
            "fire_comp_cut_off": fc_cut, "fire_comp_late": fc_late}


def _run(jobs, out, workers):
    import pandas as pd
    t0, rows = time.time(), []
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            rows.append(r)
            if (k + 1) % 100 == 0 or k + 1 == len(jobs):
                print(f"    {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def part_K():
    from evac.fixed_plan import make_plan, write_markdown
    os.makedirs(PLANS, exist_ok=True)
    jobs = []
    for building, name, busy_floor in K_SCEN:
        G = _graph(building)
        sc = _scen(building)[name]
        plans = {e: make_plan(G, sc, count_error=e) for e in (-0.5, -0.3, 0.0, 0.3, 0.5)}
        if name in ("floor3_full_200", "lecture_hall_f1_120", "hosp_day_peak"):
            write_markdown(G, plans[0.0], os.path.join(PLANS, f"plan_{name}.md"),
                           f"Printed evacuation plan - {TITLE[name]}")
        variants = [("shortest path (no plan)", "safe_shortest", {}),
                    ("one-shot plan, perfect data", "ccrp", {}),
                    ("live planner, perfect data", "lookahead", {}),
                    ("printed plan: 1 route per room", "fixed_plan", {"route_plan": plans[0.0], "route_plan_mode": "single"}),
                    ("printed plan: split per room", "fixed_plan", {"route_plan": plans[0.0], "route_plan_mode": "split"})]
        for e in (-0.5, -0.3, 0.3, 0.5):
            variants.append((f"printed plan, timetable {e:+.0%} off", "fixed_plan",
                             {"route_plan": plans[e], "route_plan_mode": "split"}))
        for v in (0.1, 0.3, 0.5):
            variants.append((f"live, {v:.0%} of people unseen", "lookahead", {"sense_miss": v}))
        for v in (0.1, 0.3, 0.5):
            variants.append((f"live, +{v:.0%} phantom people", "lookahead", {"sense_false": v}))
        variants.append(("live, busiest floor's sensors down", "lookahead", {"sense_dead_floor": busy_floor}))
        for tag, method, over in variants:
            for sd in SEEDS:
                jobs.append(("K", building, name, tag, method, over, sd))
    return jobs


def part_Z():
    """Pre-computed plan per fire alarm zone: knows where the fire is, still no live data on people."""
    from evac.fixed_plan import make_plan
    jobs = []
    for building, name, _f in K_SCEN:
        sc = _scen(building)[name]
        plan = make_plan(_graph(building), sc, with_fire=True)
        for sd in SEEDS:
            jobs.append(("K", building, name, "printed plan per fire-alarm zone", "fixed_plan",
                         {"route_plan": plan, "route_plan_mode": "split"}, sd))
    return jobs


def _procedures():
    out = []
    for name, staff, spreads in (("hosp_night_wards_fire", 5, (0.1, 0.5)), ("hosp_day_spreading_f2", 10, (0.1, 0.5))):
        base = _scen("hospital")[name]
        for sp in spreads:
            fire = {**base["fires"][0], "spread_m_per_s": sp}
            open_fire = {"fires": [fire]}
            cont = {"fires": [{**fire, "contained": True}]}
            st = {"staff": {"2": staff, "3": staff}}
            h1 = {"refuge": [{"compartment": "f2-W", "target": "f2_j8"}], "in_place": "rest"}
            h2 = {"refuge": [{"compartment": "f2-W", "target": "f2_j8"}, {"compartment": "f3-W", "target": "f3_j8"}],
                  "in_place": "rest"}
            tag = f"spread {sp:g} m/s"
            out += [(name, tag, "A. vertical, no fire doors, unlimited staff", open_fire),
                    (name, tag, "B. vertical, fire doors, unlimited staff", cont),
                    (name, tag, f"C. vertical, fire doors, {staff} staff per floor", {**cont, **st}),
                    (name, tag, "D. horizontal (fire compartment), rest in place", {**cont, **st, **h1}),
                    (name, tag, "E. horizontal (fire compartment + one above), rest in place", {**cont, **st, **h2})]
    return out


def part_P():
    jobs = []
    for name, tag, strat, over in _procedures():
        for sd in SEEDS:
            jobs.append(("P", "hospital", name, f"{tag} | {strat}", "safe_shortest", over, sd))
    return jobs


# ------------------------------------------------------------------ report
def report():
    import pandas as pd
    from evac.final_report import BLUE, ORANGE, GREY, INK, MUTED, _plt, _style
    from evac.summarize import boot_ci
    lines, findings = [], []
    K = pd.read_csv(os.path.join(RES, "s7_K.csv")) if os.path.exists(os.path.join(RES, "s7_K.csv")) else None
    if K is not None and os.path.exists(os.path.join(RES, "s7_Z.csv")):
        Z = pd.read_csv(os.path.join(RES, "s7_Z.csv"))
        K = pd.concat([K, Z])
        # show the zone plan right after the other printed plans
        order = list(dict.fromkeys(K.tag))
        order.remove("printed plan per fire-alarm zone")
        order.insert(order.index("printed plan: split per room") + 1, "printed plan per fire-alarm zone")
        K["tag"] = pd.Categorical(K["tag"], order, ordered=True)
        K = K.sort_values("tag")
        K["tag"] = K["tag"].astype(str)
    P = pd.read_csv(os.path.join(RES, "s7_P.csv")) if os.path.exists(os.path.join(RES, "s7_P.csv")) else None
    if K is not None:
        K["metric"] = K.apply(lambda r: r["ambulatory_last"] if r["building"] == "hospital" else r["last"], axis=1)
        tags = list(dict.fromkeys(K.tag))
        lines += ["## 1. What if the system does not know exactly where people are?", "",
                  "Metric: last person out (campus) / last WALKING person out (hospital), mean of 10 seeds. "
                  "**Gain kept** = share of the live planner's improvement over shortest path that a variant keeps "
                  "(100% = as good as perfect live data, 0% = no better than shortest path). Same people in every row.", ""]
        scen = list(dict.fromkeys(K.scenario))
        lines += ["| Variant | " + " | ".join(TITLE[s] for s in scen) + " | Gain kept (median) |",
                  "|" + " --- |" * (len(scen) + 2)]
        kept = {}
        for t in tags:
            cells, ks = [], []
            for s in scen:
                x = K[(K.scenario == s) & (K.tag == t)]["metric"].mean()
                b = K[(K.scenario == s) & (K.tag == "shortest path (no plan)")]["metric"].mean()
                l = K[(K.scenario == s) & (K.tag == "live planner, perfect data")]["metric"].mean()
                g = (b - x) / (b - l) if b - l > 1 else float("nan")
                ks.append(g)
                cells.append(f"{x:.0f}")
            gk = statistics.median([v for v in ks if v == v])
            kept[t] = (gk, min(v for v in ks if v == v), max(v for v in ks if v == v))
            lines.append(f"| {t} | " + " | ".join(cells) + f" | {gk:.0%} |")
        ex = K[K.scenario.str.contains("fire")].groupby("tag")["exposed"].mean()
        lines += ["", "People exposed to smoke, mean over the fire scenarios: " +
                  "; ".join(f"{t} {ex[t]:.1f}" for t in tags if t in ex) + ".", "",
                  "Printed plans for three scenarios: `docs/plans/` (room -> stairs -> exit, with the share of the room).", ""]
        p1 = kept["printed plan: 1 route per room"]

        def kept_on(tag, fire):
            vals = []
            for s in scen:
                if ("fire" in s) != fire:
                    continue
                x = K[(K.scenario == s) & (K.tag == tag)]["metric"].mean()
                b = K[(K.scenario == s) & (K.tag == "shortest path (no plan)")]["metric"].mean()
                l = K[(K.scenario == s) & (K.tag == "live planner, perfect data")]["metric"].mean()
                vals.append((b - x) / (b - l))
            return statistics.median(vals)
        nf, wf = kept_on("printed plan: split per room", False), kept_on("printed plan: split per room", True)
        findings.append(f"**No IoT needed for most of the gain - when there is no fire in the way.** A plan printed once from "
                        f"the expected occupancy (rooms split between stairwells) keeps a median **{nf:.0%}** of the live "
                        f"planner's gain in the scenarios without a fire on the escape routes, but only {wf:.0%} in the fire "
                        f"scenarios: a printed plan cannot know where the fire is. One route per room keeps just {p1[0]:.0%} overall.")
        if "printed plan per fire-alarm zone" in kept:
            zf = kept_on("printed plan per fire-alarm zone", True)
            findings.append(f"Fix without crowd sensors: pre-compute one plan **per fire-alarm zone** (the fire panel already "
                            f"knows where the alarm is) - it keeps {zf:.0%} of the gain in the fire scenarios.")
        tw = [kept[f"printed plan, timetable {e:+.0%} off"][0] for e in (-0.5, -0.3, 0.3, 0.5)]
        findings.append(f"The printed plan is robust to a wrong timetable: with 30-50% fewer or more people than "
                        f"expected it still keeps {min(tw):.0%}-{max(tw):.0%} of the gain.")
        um = [kept[f"live, {v:.0%} of people unseen"][0] for v in (0.1, 0.3, 0.5)]
        ph = [kept[f"live, +{v:.0%} phantom people"][0] for v in (0.1, 0.3, 0.5)]
        dd = kept["live, busiest floor's sensors down"][0]
        findings.append(f"With live but imperfect counts, **missing people hurt far more than phantom ones**: "
                        f"10/30/50% unseen keeps {um[0]:.0%}/{um[1]:.0%}/{um[2]:.0%} of the gain, while +10/30/50% "
                        f"phantoms keep {ph[0]:.0%}/{ph[1]:.0%}/{ph[2]:.0%} - over 100%, i.e. planning for more people "
                        "than seen is even slightly BETTER than exact counts (it acts as a safety margin on stair capacity). "
                        f"Design rule: over-count rather than under-count. With the busiest floor's sensors down: {dd:.0%}.")
        # figure 22
        plt = _plt()
        order = [t for t in tags if t != "shortest path (no plan)"]
        fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(order) + 1.3))
        for i, t in enumerate(order):
            g, lo, hi = kept[t]
            y = len(order) - 1 - i
            c = BLUE if t.startswith("live") else ORANGE if t.startswith("printed") else GREY
            ax.plot([lo * 100, hi * 100], [y, y], color=c, lw=2, alpha=0.5)
            ax.plot(g * 100, y, "o", color=c, ms=6, mec="white", mew=1)
        ax.axvline(0, color=INK, lw=1)
        ax.axvline(100, color=INK, lw=1, ls=(0, (3, 3)))
        ax.set_yticks(range(len(order))[::-1], order, fontsize=8)
        ax.set_xlabel("Share of the live planner's gain kept (%) - median and range over 8 scenarios", fontsize=9, color=MUTED)
        ax.set_title("Imperfect knowledge: printed plans (orange) and noisy live counts (blue)", loc="left",
                     fontsize=11, fontweight="bold")
        _style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "fig22_knowledge.png"), dpi=170)
        plt.close(fig)
        lines += ["![Knowledge](figures/fig22_knowledge.png)", ""]
    if P is not None:
        P[["spread", "strategy"]] = P["tag"].str.split(" \\| ", expand=True)
        lines += ["## 2. Hospital: evacuate everyone, or move patients sideways first?", "",
                  "Fire starts in the 2nd-floor store room (west compartment). 'Fire doors' = the fire is contained in "
                  "its compartment (other compartment gets only light smoke). Staff: 2 per bed, they walk back for the "
                  "next bed. Horizontal = everyone in the fire compartment goes to the east compartment of the same "
                  "floor; everyone else stays where they are (defend in place). 10 seeds.", "",
                  "| Scenario | Fire spread | Strategy | Fire compartment safe (s) | Everyone safe / out (s) | Still inside at 30 min | Exposed | Defending in place |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for (s, sp, st), x in P.groupby(["scenario", "spread", "strategy"], sort=False):
            fc_txt = f"{x['fire_comp_safe_last'].mean():.0f}"
            if x["fire_comp_late"].mean() > 0:
                fc_txt += f" ({x['fire_comp_late'].mean():.1f} still inside at 30 min)"
            if x["fire_comp_cut_off"].mean() > 0:
                fc_txt += f"; {x['fire_comp_cut_off'].mean():.1f} cut off by the fire from the start"
            lines.append(f"| {TITLE[s]} | {sp} | {st} | {fc_txt} | {x['last'].mean():.0f} | {x['timed_out'].mean():.1f} | "
                         f"{x['exposed'].mean():.1f} | {x['in_place'].mean():.0f} |")
        lines += ["", "Fire compartment safe = time until everyone there who can move is at an exit or in the refuge "
                  "(1800 s if anyone is still inside at 30 min). 'Cut off' = people in the room where the fire starts, "
                  "with no route from the first moment - the same in every strategy.", ""]
        g = lambda s, sp, st, c: P[(P.scenario == s) & (P.spread == sp) & (P.strategy.str.startswith(st))][c].mean()
        n0 = "hosp_night_wards_fire"
        sp5 = "spread 0.5 m/s"
        findings.append(f"**Fire doors matter most:** in the night ward fire they cut the people exposed from "
                        f"{g(n0, sp5, 'A.', 'exposed'):.0f} to {g(n0, sp5, 'B.', 'exposed'):.0f} - routing could not do that.")
        findings.append(f"**Staff, not routing, limits a vertical hospital evacuation.** With 5 night staff per ward floor "
                        f"(2 per bed), evacuating everyone down the ramp takes {g(n0, sp5, 'C.', 'last'):.0f} s and "
                        f"{g(n0, sp5, 'C.', 'timed_out'):.1f} people are still inside after 30 minutes "
                        f"(unlimited staff: {g(n0, sp5, 'B.', 'last'):.0f} s).")
        findings.append(f"**Progressive horizontal evacuation** (the standard hospital procedure) gets everyone in the "
                        f"fire compartment to safety in {g(n0, sp5, 'D.', 'fire_comp_safe_last'):.0f} s with the same "
                        f"staff, no one left behind, and no more people exposed "
                        f"({g(n0, sp5, 'D.', 'exposed'):.0f} vs {g(n0, sp5, 'C.', 'exposed'):.0f}). The model reproduces "
                        "why hospitals move patients sideways first.")
        plt = _plt()
        fig, axs = plt.subplots(1, 2, figsize=(14, 3.8))
        for ax, s in zip(axs, (n0, "hosp_day_spreading_f2")):
            x = P[(P.scenario == s) & (P.spread == sp5)]
            strat = list(dict.fromkeys(x.strategy))
            v = [x[x.strategy == t]["fire_comp_safe_last"].mean() for t in strat]
            ax.barh(range(len(strat))[::-1], v, color=[GREY, GREY, ORANGE, BLUE, BLUE], height=0.55)
            for i, t in enumerate(strat):
                left = x[x.strategy == t]["fire_comp_late"].mean()
                if left > 0:
                    ax.text(v[i] - 20, len(strat) - 1 - i, f"{left:.0f} still inside at 30 min", va="center",
                            ha="right", fontsize=8, color="white")
            ax.set_yticks(range(len(strat))[::-1], [t.split(". ", 1)[1] for t in strat], fontsize=8)
            ax.set_xlabel("Time until everyone in the fire compartment is safe (s)", fontsize=9, color=MUTED)
            ax.set_title(TITLE[s], loc="left", fontsize=11, fontweight="bold")
            _style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, "fig23_hospital_procedures.png"), dpi=170)
        plt.close(fig)
        lines += ["![Procedures](figures/fig23_hospital_procedures.png)", ""]
    mlt = os.path.join(ROOT, "docs", "ml_transfer.md")
    if os.path.exists(mlt):
        with open(mlt, encoding="utf-8") as f:
            t = f.read()
        block = t.split("## Findings", 1)[1].split("![", 1)[0]
        for ln in block.strip().splitlines():
            if ln[:2].rstrip(".").isdigit() or ln[:3].rstrip(". ").isdigit():
                findings.append("ML: " + ln.split(". ", 1)[1])
        lines += ["## 3. Retraining the ML model on both buildings", "", "See [ml_transfer.md](ml_transfer.md).",
                  "", "![ML transfer](figures/fig24_ml_transfer.png)", ""]
    lim = ["## Assumptions and limits of these experiments", "",
           "- Fire compartments are assumed at the bed-lift core (the drawings do not mark fire doors); "
           "light smoke through closed fire doors = 15% of the hazard.",
           "- Staff are modelled only as teams of 2 per bed with a fixed walk back; their own risk is not modelled.",
           "- Defend-in-place exposure is checked at the person's room; a real refuge also needs smoke control.",
           "- Sensor errors are random per update; real systems can have correlated errors (a whole camera zone).",
           "- A printed plan is made without knowing the fire; in a fire, routes through smoke fall back to the signs.", ""]
    head = ["# Session 7: making it practical", "",
            "Three questions a building manager would ask: *Do I need sensors? What if they are wrong? "
            "And in a hospital, should we evacuate at all?* Same engine and statistics as before.", "",
            "## Key findings", ""] + [f"{i}. {f}" for i, f in enumerate(findings, 1)] + [""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(head + lines + lim))
    print("\n".join(head))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="*", default=["K", "Z", "P"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    for p in a.parts:
        out = os.path.join(RES, f"s7_{p}.csv")
        if os.path.exists(out):
            print(f"part {p}: exists, skipping")
            continue
        print(f"part {p} ...", flush=True)
        _run({"K": part_K, "Z": part_Z, "P": part_P}[p](), out, a.workers)
    report()


if __name__ == "__main__":
    main()
