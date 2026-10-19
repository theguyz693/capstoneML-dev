"""
evac/calibration.py
Improving ML, step 2: calibrate the simulator on REAL crowd experiments (Forschungszentrum Juelich).

Data (data/juelich/, from the PedPy demo data, MIT licence; experiments of the Juelich Pedestrian Dynamics
group, Pedestrian Dynamics Data Archive):
  uo-BBB-180-180   one-way corridor, 1.8 m wide, entrance width BBB cm   (Zhang et al. 2011, J. Stat. Mech.)
  uo-180-180-BBB   same corridor, EXIT narrowed to BBB cm -> a bottleneck with a jam in front of it
  traj_UNI_CORR_500_*  one-way corridor 5 m wide                          (Zhang et al. 2011)
  040_c_56_h-      bottleneck 0.5 m wide, 75 people (Wuppertal 2018; Adrian, Seyfried & Sieben 2020)

What is calibrated:
  1. Speed-density curve (fundamental diagram): free speed v0, shape gamma and jam density of the
     Weidmann form are fitted to ~15,000 per-frame measurements (classic density in a measurement area,
     mean speed of the people inside it).
  2. Door (bottleneck) flow per metre of width: fitted to the three narrowed-exit runs (0.70, 0.95, 1.20 m);
     the 0.5 m Wuppertal run is held out as a test.
  3. Door edges: crowd density is measured over the approach space, not the door width
     (config.DOOR_APPROACH_WIDTH_M) - otherwise a crowded door edge slows its own outflow to about half
     the measured value (found in Session 9, docs/jupedsim_crosscheck.md).
Then the SIMULATOR (not just the formulas) is checked: the same number of people as in each experiment
are pushed through a door of the same width, and the simulated time for the whole group is compared
with the measured time, before (handbook values) and after calibration.

Not calibrated here (no Juelich data in the sample): stairs - they stay validated against the NIST
drill data (docs/validation.md, V2).

Run:  python -m evac.calibration   -> evac/calibrated_params.json, docs/calibration.md, fig28_calibration.png
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data", "juelich")
OUT = os.path.join(HERE, "calibrated_params.json")
DOC = os.path.join(ROOT, "docs", "calibration.md")
FIG = os.path.join(ROOT, "docs", "figures", "fig28_calibration.png")

# steady-state frame windows of the UO runs (from the PedPy fundamental-diagram notebook)
UO_STEADY = {"uo-050-180-180": (211, 800), "uo-060-180-180": (243, 771), "uo-070-180-180": (203, 1113),
             "uo-100-180-180": (200, 790), "uo-145-180-180": (300, 1097), "uo-180-180-070": (500, 1399),
             "uo-180-180-095": (400, 1350), "uo-180-180-120": (300, 1099), "uo-180-180-180": (400, 1284)}
UNI = ("traj_UNI_CORR_500_01", "traj_UNI_CORR_500_03", "traj_UNI_CORR_500_08")
# bottleneck runs: (file, width m, crossing line axis, crossing line value, fps, unit, role)
BOTTLENECKS = [("uo-180-180-070", 0.70, -4.0, 16, "cm", "fit"),
               ("uo-180-180-095", 0.95, -4.0, 16, "cm", "fit"),
               ("uo-180-180-120", 1.20, -4.0, 16, "cm", "fit"),
               ("040_c_56_h-", 0.50, -0.5, 25, "m", "test")]
DOOR_APPROACH_W = 2.0          # corridor width in our buildings (config.WIDTH_M["corridor"])


def load(name, unit, fps):
    d = pd.read_csv(os.path.join(DATA, name + ".txt"), sep=r"\s+", comment="#", header=None,
                    names=["id", "frame", "x", "y", "z"])
    if unit == "cm":
        d[["x", "y"]] /= 100.0
    d["t"] = d["frame"] / fps
    return d


def with_speed(d, fps, half_window):
    """Speed from the position change over +-half_window frames (smooths camera jitter)."""
    d = d.sort_values(["id", "frame"]).copy()
    g = d.groupby("id")
    dx = g["x"].shift(-half_window) - g["x"].shift(half_window)
    dy = g["y"].shift(-half_window) - g["y"].shift(half_window)
    d["v"] = np.hypot(dx, dy) / (2 * half_window / fps)
    return d.dropna(subset=["v"])


def fd_points():
    """Per-frame (density, mean speed) in a measurement area, every corridor run."""
    rows = []
    for name, (a, b) in UO_STEADY.items():
        d = with_speed(load(name, "cm", 16), 16, 5)
        d = d[d["frame"].between(a, b)]
        m = d[d["x"].between(0, 1.8) & d["y"].between(-2, 0)]          # 1.8 x 2 m area before the exit
        agg = m.groupby("frame").agg(n=("id", "size"), v=("v", "mean"))
        agg["rho"] = agg["n"] / 3.6
        agg["run"] = name
        rows.append(agg)
    for name in UNI:
        d = with_speed(load(name, "m", 25), 25, 8)
        m = d[d["x"].between(-2, 0) & d["y"].between(0, 5)]             # 2 x 5 m area mid-corridor
        agg = m.groupby("frame").agg(n=("id", "size"), v=("v", "mean"))
        agg = agg[agg["n"] >= 2]
        agg["rho"] = agg["n"] / 10.0
        agg["run"] = name
        rows.append(agg)
    D = pd.concat(rows)
    return D[D["v"] > 0.02].reset_index(drop=True)      # drop the 'stand still at the end' frames


def weidmann(rho, v0, gamma, rho_max):
    rho = np.maximum(np.asarray(rho, float), 1e-3)
    return v0 * np.clip(1 - np.exp(-gamma * (1 / rho - 1 / rho_max)), 0.0, 1.0)


def fit_fd(D):
    from scipy.optimize import curve_fit
    D = D.copy()
    D["bin"] = (D["rho"] / 0.25).round() * 0.25
    B = D.groupby("bin").agg(v=("v", "mean"), k=("v", "size"))
    B = B[B["k"] >= 20]
    p, _ = curve_fit(weidmann, B.index.values, B["v"].values, p0=[1.34, 1.913, 5.4],
                     sigma=1 / np.sqrt(B["k"].values), bounds=([0.8, 0.3, 3.0], [2.0, 6.0, 12.0]))
    return {"v0": float(p[0]), "gamma": float(p[1]), "rho_max": float(p[2])}, B


def crossings(name, unit, fps, line):
    """Times at which each person first crosses y = line (walking towards -y)."""
    d = load(name, unit, fps)
    out = []
    for _pid, g in d.groupby("id"):
        g = g.sort_values("frame")
        below = g[g["y"] < line]
        if len(below) and g["y"].iloc[0] > line:
            out.append(float(below["t"].iloc[0]))
    return np.sort(out)


def measured_bottlenecks():
    rows = []
    for name, w, line, fps, unit, role in BOTTLENECKS:
        ts = crossings(name, unit, fps, line)
        n = len(ts)
        i, j = int(0.1 * n), int(0.9 * n)
        rows.append({"run": name, "width_m": w, "people": n, "role": role,
                     "flow": (j - i) / (ts[j] - ts[i]),                      # steady flow, 10-90 % of people
                     "group_time_s": float(ts[-1] - ts[0])})                 # first to last person through
    return rows


def sim_bottleneck(width, n, seed=0):
    """Push n people through one door of the given width in OUR simulator; returns (steady flow, group time)
    measured where people LEAVE the door, like the experiments."""
    from evac import config as C
    from evac.building import build_graph
    from evac.simulator import run
    U = 1.0 / C.METERS_PER_UNIT
    G = build_graph({"room": (0, 0, 0), "test exit": (3 * U, 0, 0)}, [("room", "test exit")])
    d = G.edges["room", "test exit"]
    d["width_m"] = width
    d["flow_cap"] = C.SPECIFIC_FLOW["exit_door"] * width
    from evac.physics import density_width, sustainable_flow
    d["dens_width_m"] = density_width("exit_door", width)
    d["eff_cap"] = min(d["flow_cap"], sustainable_flow(d["dens_width_m"], d["free_speed"]))
    d["jam_cap"] = max(2, int(C.JAM_DENSITY * d["length_m"] * d["dens_width_m"]))
    sc = {"spawn": "single", "origin": "room", "pre_movement_max_s": 5.0, "fires": [], "n_agents": n}
    _, recs = run(G, sc, "nearest", seed=seed)
    ts = np.sort([r["t_out"] for r in recs if r["t_out"] is not None])
    i, j = int(0.1 * len(ts)), int(0.9 * len(ts))
    return (j - i) / (ts[j] - ts[i]), float(ts[-1] - ts[0])


def fit_door_flow(meas):
    """Specific door flow (persons/s/m) that makes the SIMULATED steady flow match the 'fit' runs."""
    from evac import config as C
    fit = [m for m in meas if m["role"] == "fit"]
    # flow is proportional to width in our model: least squares J = c * b, then check it in the simulator
    c = sum(m["flow"] * m["width_m"] for m in fit) / sum(m["width_m"] ** 2 for m in fit)
    old = C.SPECIFIC_FLOW["exit_door"]
    best = None
    for cand in np.arange(round(c - 0.3, 2), round(c + 0.31, 2), 0.05):
        for et in ("room_door", "exit_door"):
            C.SPECIFIC_FLOW[et] = float(cand)
        err = np.mean([abs(sim_bottleneck(m["width_m"], m["people"])[0] - m["flow"]) / m["flow"] for m in fit])
        if best is None or err < best[1]:
            best = (float(cand), float(err))
    for et in ("room_door", "exit_door"):
        C.SPECIFIC_FLOW[et] = old
    return round(best[0], 2), c


def compare_sim(meas, label):
    rows = []
    for m in meas:
        f, gt = sim_bottleneck(m["width_m"], m["people"])
        rows.append({"params": label, "run": m["run"], "role": m["role"], "width_m": m["width_m"],
                     "people": m["people"], "measured_flow": m["flow"], "sim_flow": f,
                     "measured_time": m["group_time_s"], "sim_time": gt})
    return rows


def main():
    from evac import config as C
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    print("fundamental diagram ...", flush=True)
    D = fd_points()
    fd, B = fit_fd(D)
    rmse = lambda p: float(np.sqrt(np.mean((weidmann(D["rho"], *p) - D["v"]) ** 2)))
    rmse_old = rmse([1.25, 1.913, 5.4])
    rmse_new = rmse([fd["v0"], fd["gamma"], fd["rho_max"]])
    print(f"  fit v0={fd['v0']:.3f} gamma={fd['gamma']:.3f} rho_max={fd['rho_max']:.2f}  "
          f"RMSE {rmse_old:.3f} -> {rmse_new:.3f} m/s", flush=True)

    meas = measured_bottlenecks()
    for m in meas:
        print(f"  measured {m['run']}: {m['width_m']} m, {m['people']} people, {m['flow']:.2f} p/s", flush=True)

    # BEFORE: handbook values
    C.use_params("sfpe")
    before = compare_sim(meas, "before (handbook)")

    # set the fitted speed-density curve + approach width, then fit door flow in the simulator
    C.FREE_WALK_SPEED, C.FD_GAMMA, C.JAM_DENSITY = round(fd["v0"], 3), round(fd["gamma"], 3), round(fd["rho_max"], 2)
    C.DOOR_APPROACH_WIDTH_M = DOOR_APPROACH_W
    from evac import physics
    physics.refresh()
    corridor_flow = round(physics.sustainable_flow(1.0, C.FREE_WALK_SPEED), 2)   # FD capacity per metre
    C.SPECIFIC_FLOW["corridor"] = C.SPECIFIC_FLOW["ramp"] = corridor_flow
    door, c_ls = fit_door_flow(meas)
    print(f"  door flow: least squares {c_ls:.2f}, simulator fit {door:.2f} p/s/m; corridor capacity {corridor_flow}",
          flush=True)
    params = {"FREE_WALK_SPEED": C.FREE_WALK_SPEED, "FD_GAMMA": C.FD_GAMMA, "JAM_DENSITY": C.JAM_DENSITY,
              "DOOR_APPROACH_WIDTH_M": DOOR_APPROACH_W,
              "SPECIFIC_FLOW": {"room_door": door, "exit_door": door, "corridor": corridor_flow, "ramp": corridor_flow}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"params": params,
                   "fit": {**fd, "rmse_before": rmse_old, "rmse_after": rmse_new, "fd_points": int(len(D)),
                           "door_flow_least_squares": c_ls},
                   "source": "Juelich experiments via PedPy demo data (data/juelich); evac/calibration.py"},
                  f, indent=1)
    C.use_params("juelich")
    after = compare_sim(meas, "after (calibrated)")
    report(D, B, fd, rmse_old, rmse_new, meas, before, after, params)


def report(D, B, fd, rmse_old, rmse_new, meas, before, after, params):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evac.final_report import BLUE, ORANGE, GREY, MUTED, _style
    T = pd.DataFrame(before + after)
    T["flow_err"] = (T["sim_flow"] - T["measured_flow"]) / T["measured_flow"]
    T["time_err"] = (T["sim_time"] - T["measured_time"]) / T["measured_time"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    s = D.sample(min(4000, len(D)), random_state=0)
    ax[0].scatter(s["rho"], s["v"], s=3, color="#cfcdc4", label="Juelich measurements (per frame)")
    ax[0].plot(B.index, B["v"], "o", color=GREY, ms=4, label="binned mean")
    r = np.linspace(0.05, 5.4, 300)
    ax[0].plot(r, weidmann(r, 1.25, 1.913, 5.4), color=ORANGE, lw=2, label=f"before: handbook (RMSE {rmse_old:.2f} m/s)")
    ax[0].plot(r, weidmann(r, fd["v0"], fd["gamma"], fd["rho_max"]), color=BLUE, lw=2,
               label=f"after: fitted (RMSE {rmse_new:.2f} m/s)")
    ax[0].set(xlabel="Density (persons/m²)", ylabel="Walking speed (m/s)", xlim=(0, 5), ylim=(0, 1.9))
    ax[0].set_title("Speed vs crowd density", loc="left", fontsize=11, fontweight="bold")
    ax[0].legend(fontsize=7, frameon=False)
    _style(ax[0])
    w = [m["width_m"] for m in meas]
    ax[1].plot(w, [m["flow"] for m in meas], "ks", ms=7, label="measured (Juelich / Wuppertal)")
    for lab, col in (("before (handbook)", ORANGE), ("after (calibrated)", BLUE)):
        sub = T[T["params"] == lab]
        ax[1].plot(sub["width_m"], sub["sim_flow"], "o", color=col, ms=7, label=f"our simulator, {lab}")
    ax[1].set(xlabel="Door width (m)", ylabel="Flow out of the door (persons/s)", xlim=(0.4, 1.3), ylim=(0, 2.8))
    ax[1].set_title("Bottleneck flow", loc="left", fontsize=11, fontweight="bold")
    ax[1].legend(fontsize=7, frameon=False)
    _style(ax[1])
    fig.tight_layout()
    fig.savefig(FIG, dpi=170)
    plt.close(fig)

    def mape(lab, col, role=None):
        sub = T[(T["params"] == lab) & ((T["role"] == role) if role else True)]
        return float(sub[col].abs().mean())
    L = ["# Calibration on real crowd experiments (Improving ML, step 2)", "",
         "The simulator's walking speed, speed-density curve and door flow were set from handbooks. Here they are "
         "**fitted to real trajectories** recorded by Forschungszentrum Juelich (one-way corridor and bottleneck "
         "experiments; `data/juelich/`, see `evac/calibration.py`).", "",
         "## Findings", "",
         f"1. **Speed vs density** ({len(D):,} per-frame measurements, 12 corridor runs): fitted free speed "
         f"{fd['v0']:.2f} m/s (was 1.25), jam density {fd['rho_max']:.2f} persons/m² (was 5.4), shape {fd['gamma']:.2f} "
         f"(was 1.913). Error of the curve: {rmse_old:.2f} -> {rmse_new:.2f} m/s.",
         f"2. **Door flow**: fitted {params['SPECIFIC_FLOW']['exit_door']:.2f} persons/s per metre of door width "
         f"(was 1.3, the SFPE design value). Corridors carry up to {params['SPECIFIC_FLOW']['corridor']:.2f} "
         "persons/s/m (the capacity of the fitted curve).",
         f"3. **Whole simulator vs the experiments** (same number of people, same door width): steady flow error "
         f"{mape('before (handbook)', 'flow_err'):.0%} before -> {mape('after (calibrated)', 'flow_err'):.0%} after; "
         f"time for the whole group {mape('before (handbook)', 'time_err'):.0%} -> {mape('after (calibrated)', 'time_err'):.0%}. "
         f"On the held-out 0.5 m Wuppertal bottleneck (not used for fitting): flow error "
         f"{mape('before (handbook)', 'flow_err', 'test'):.0%} -> {mape('after (calibrated)', 'flow_err', 'test'):.0%}.",
         "4. **Caveat**: the participants were students in a laboratory. Real mixed crowds (older people, luggage, "
         "smoke, stress) are usually slower, which is why design codes use 1.3 persons/s/m. Both parameter sets are "
         "kept: `PARAM_SET = \"juelich\"` (realistic, default from now on) and `\"sfpe\"` (conservative, used for "
         "all earlier results and for building audits). The IMO ship-evacuation test 4 caps door flow at 1.33 "
         "persons/s/m, so the calibrated set intentionally no longer meets that design cap.",
         "", "![Calibration](figures/fig28_calibration.png)", "",
         "## Simulator vs experiment, run by run", "",
         "| Run | Door width | People | Used for | Measured flow (p/s) | Sim before | Sim after | Measured group time (s) | Sim before | Sim after |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in meas:
        b = T[(T["run"] == m["run"]) & (T["params"] == "before (handbook)")].iloc[0]
        a = T[(T["run"] == m["run"]) & (T["params"] == "after (calibrated)")].iloc[0]
        L.append(f"| {m['run']} | {m['width_m']:.2f} m | {m['people']} | {m['role']} | {m['flow']:.2f} | "
                 f"{b['sim_flow']:.2f} | {a['sim_flow']:.2f} | {m['group_time_s']:.1f} | {b['sim_time']:.1f} | {a['sim_time']:.1f} |")
    L += ["", "## Calibrated parameters (`evac/calibrated_params.json`)", "", "```json",
          json.dumps(params, indent=1), "```", "",
          "Not calibrated here: stair speed and stair flow (no stair runs in the sample) - they stay validated "
          "against NIST drill data (docs/validation.md). Data: PedPy demo data (MIT licence), experiments of the "
          "Juelich Pedestrian Dynamics group: Zhang et al. (2011) J. Stat. Mech. P06004 (corridors); "
          "Adrian, Seyfried & Sieben (2020) J. R. Soc. Interface 17:20190871 (Wuppertal bottleneck). Cite the "
          "Pedestrian Dynamics Data Archive when publishing.", ""]
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    T.to_csv(os.path.join(ROOT, "results", "calibration_sim_vs_real.csv"), index=False)
    print("\n".join(L[5:10]))


if __name__ == "__main__":
    main()
