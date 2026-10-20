# evac — headless evacuation engine

Research engine for the capstone study. Runs thousands of seeded evacuation
simulations without the UI, so routing methods can be compared with statistics.

## Research dashboard
```bash
pip install fastapi uvicorn networkx
python research_server.py          # starts the existing server.py app + research endpoints
```
- `http://localhost:8000/` - original 3D dashboard (unchanged)
- `http://localhost:8000/research` - **Research Lab**: pick a scenario, methods, compliance and fire; watch the methods
  race side by side on each floor (queues show as blobs), with a live scoreboard and "people still inside" curve;
  **Kiosk signs** tab (the arrow every sign should show; click a spot for its kiosk screen);
  **Weak spots** tab (which single failure hurts most).

## Run experiments
```bash
pip install networkx              # scikit-learn/pandas/joblib only needed for method "ml"
python -m evac.runner                                    # all scenarios, 5 seeds
python -m evac.runner --scenarios fire_j19_150 --methods nearest safe_shortest ml --seeds 30
python -m evac.runner --methods nearest safe_shortest reactive --seeds 10 --out results/session2_runs.csv
python -m evac.summarize results/session2_runs.csv      # means, 95% CIs, paired tests -> *_summary.md
python -m evac.verification                             # simulator sanity tests -> docs/verification/
python tests/test_evac_session1.py                      # safety + reproducibility checks
python tests/test_evac_session2.py                      # verification + reactive method checks
python tests/test_evac_session3.py                      # planner checks
python tests/test_evac_session4.py                      # behaviour + spreading fire checks
python tests/test_evac_session5.py                      # replay, signs, weak spots, API
python -m evac.weakspots --scenario floor3_full_200      # single-failure ranking -> docs/weakspots.md
python -m evac.delay_model                              # ML: generate simulator data, train, compare vs formula (needs scikit-learn)
python tests/test_evac_session6.py                      # ML-in-loop, GA, sensitivity, ablation checks
```

### Session 6 (research depth)
```bash
pip install lightgbm shap optuna SALib pandas matplotlib scikit-learn joblib scipy
python -m evac.ml_study            # model comparison (grouped CV), learning curve, SHAP, conformal -> docs/ml_study.md, models/
python -m evac.sensitivity         # Morris sensitivity over 9 physical assumptions (~10 min) -> docs/sensitivity.md
python -m evac.tune --trials 30    # Bayesian optimisation of planner settings (~5 min) -> docs/tuning.md
python -m evac.ga                  # genetic-algorithm exit assignment (offline benchmark)
python -m evac.final_experiments   # full matrix: 30 seeds, ML-in-loop, GA, bound, tipping, freshness, ablations (~1 h, 2 cores)
python -m evac.final_report        # tables + figures -> docs/final_results.md
```

### Second building: oncology hospital
```bash
python -m evac.hospital --import <folder with the team's hospital files>   # -> data/hospital/hospital_graph.json (already done)
python -m evac.hospital_experiments    # 6 scenarios x 7 methods x 30 seeds, bound, weak spots, assumptions (~15 min)
python -m evac.hospital_report         # -> docs/hospital_results.md, figures 18-21
python tests/test_evac_hospital.py
```
The dashboard has a building selector (campus / hospital) in the top bar.

### Session 7: making it practical
```bash
python -m evac.session7            # printed plans, imperfect head-counts, hospital procedures (~15 min) -> docs/session7_results.md
python -m evac.ml_transfer         # retrain the delay model on both buildings (~12 min) -> docs/ml_transfer.md, models/delay_lgbm_v2*
python tests/test_evac_session7.py
```
- `fixed_plan.py`: a plan made ONCE from the expected occupancy (timetable / bed list), printable per room
  (`docs/plans/`), method `fixed_plan` with scenario key `route_plan`.
- `knowledge.py`: what the planner can see - `sense_miss` (people unseen), `sense_false` (phantom people),
  `sense_dead_floor` (one floor's sensors down).
- Hospital procedures (scenario keys): fires with `"contained": true` stay in their fire compartment;
  `refuge` (move a compartment to a safe place in the building), `in_place` (defend in place), `staff`
  (beds need 2 staff, who walk back for the next bed).
- `config.DELAY_MODEL = "v2"` switches the learned planner to the two-building model.

### Session 8: validation and the what-if page
```bash
python -m evac.validation                       # vs lab door flows, NIST stair speeds, IMO tests -> docs/validation.md
python -m evac.validation --drill data/field_drills.csv   # after a timed real drill (docs/field_validation_protocol.md)
python tests/test_evac_session8.py
```
### Session 9: published baselines and a JuPedSim cross-check
```bash
python -m evac.literature_experiments           # quickest path + CASPER, 30 paired seeds, both buildings -> docs/literature_baselines.md
pip install jupedsim shapely                     # only needed for the cross-check
python -m evac.jupedsim_check                    # same geometry + people in JuPedSim (CFSM, AVM, SFM) -> docs/jupedsim_crosscheck.md
python tests/test_evac_literature.py
```
New methods for `run()`: `quickest` (Kemloh Wagoum, Seyfried & Holl 2012) and `casper` (Shahabi & Wilson 2014).

Dashboard **What-if** tab (`/api/research/whatif`): take stairwells / the ramp / exit doors out of service, change
the number of people, add a fire - compare with the normal building (shortest path vs look-ahead), with queue hot-spots.
Hospital scenarios use `"spawn": "groups"` with occupant profiles (`config.PROFILES`: walk, slow, wheelchair,
bed - speed, stairs allowed, capacity used, preparation time). New edge type `ramp` (step-free, usable in a fire).
Results go to `results/` (one row per run). `scipy` is optional (Wilcoxon p-values).

### Improving ML: calibrated simulator, smoke slows people, one hybrid model tested on real data
```bash
python -m evac.calibration          # fit speed-density + door flow to Juelich trajectories (data/juelich) -> docs/calibration.md
python -m evac.ml_v3                # dataset from the calibrated simulator, hybrid model v3, real-data test -> docs/ml_v3.md
python tests/test_evac_improving_ml.py
```
- `config.PARAM_SET = "juelich"` (default, calibrated: `evac/calibrated_params.json`) or `"sfpe"` (the earlier
  handbook values; all results before this session). `config.use_params(name)` switches, then rebuild the graph.
- Smoke slows walking: `physics.smoke_speed_factor` (Frantzich & Nilsson 2003, as in FDS+Evac); used by the simulator,
  the planners and the baselines.
- Door edges measure crowd density over the approach space (`DOOR_APPROACH_WIDTH_M`), so crowded doors no longer
  slow their own outflow to half the measured value.
- `config.DELAY_MODEL = "v3"`: time = physics formula x learned correction (`learned_planner.hybrid_predict`), with
  5 %/95 % quantile models widened by conformal calibration on real crossings.
- Dashboard: the **Evac Route** tab of `index.html` now uses `/api/research/route` (research engine + ML v3) for both
  buildings, with a time-to-exit range and an optional crowd ("other people evacuating" = a simulated scenario
  snapshot, `stop_at_s`). The old Random-Forest route (`server.py /api/evacuate`, "v1") is no longer used.
- All main results were re-run with the calibrated parameters; the earlier reports are in `docs/archive_handbook_params/`.

### Phase 2 (half-done items)
```bash
python -m evac.safety_tradeoff      # 1. safety-vs-speed dial (config.SAFETY_WEIGHT) swept vs all methods -> docs/safety_tradeoff.md
python -m evac.eta_study           # 2. time-to-exit estimates checked on 130k simulated predictions -> docs/eta_study.md
python -m evac.priority_study      # 3. vulnerable people first (planner / queue priority) -> docs/priority.md
python -m evac.scalability         # 4. re-plan time vs building size and crowd (real buildings + towers) -> docs/scalability.md
python -m evac.jam_study           # 5. does guidance create new jams elsewhere? (secondary jams) -> docs/jam_study.md
python -m evac.compliance_study    # 6. planner that does not know who follows it (naive / compliance-aware) -> docs/compliance.md
python tests/test_evac_phase2.py
```
Time-to-exit ranges on the dashboard route and kiosk signs are calibrated per band of predicted time
(`evac/eta_calibration.json`); scenario key `eta_probe_at: [t, ...]` records predictions during a run.
Research Lab: "Safety vs speed" slider (sent as `safety_weight` to `/api/research/run`).

## Files
| File | What it does |
| --- | --- |
| `config.py` | Every physical assumption (scale, speeds, widths, flow rates). Change values here only. |
| `building.py` | Loads the building graph (same geometry as `ml_edge_updater.build_full_campus_graph`) and adds metres, widths, flow capacities. |
| `hazard.py` | Fire hazard per node/edge; smoke rises up stair shafts to higher floors. |
| `routing.py` | Hard safety rules + route methods (`nearest`, `safe_shortest`, `ml`, `reactive`); building-wide sign arrows. |
| `simulator.py` | Seeded agent simulation: pre-movement delay, door/stair queues, density-dependent speed, exposure tracking. |
| `runner.py` | Scenarios × methods × seeds → CSV. Seeds are paired across methods. |
| `scenarios.json` | Scenario definitions. |
| `verification.py` | Simulator sanity tests (free walk, door flow vs width, stairs, speed–density). |
| `planner.py` | Capacity booking (CCRP) and rolling-horizon re-planning with anti-oscillation rules. |
| `optimum.py` | Theoretical lower bound via max-flow on a time-expanded network. |
| `physics.py` | Speed–density relation and each edge's sustainable flow (used by simulator and planner). |
| `delay_model.py` | Learned edge-delay predictor trained on simulator data; scenario/fire-location splits vs formulas. |
| `weakspots.py` | Blocks each stairwell / exit door / key corridor in turn and ranks the damage (N-1 analysis). |
| `api.py` | FastAPI endpoints for the dashboard (`/api/research/...`), mounted by `research_server.py`. |
| `ml_study.py` | Grouped-CV comparison (formula, ridge, RF, LightGBM, MLP), learning curve, SHAP, split-conformal + CQR intervals. |
| `learned_planner.py` | ML in the loop: re-route on predicted edge delays (formula / LightGBM / LightGBM 95% upper bound); learned walk times for the planner. |
| `sensitivity.py` | Morris elementary-effects sensitivity of results to every physical assumption. |
| `tune.py` | Optuna (TPE) tuning of planner settings with held-out check. |
| `ga.py` | Genetic algorithm over exit assignments (fitness = full simulation). |
| `final_experiments.py` / `final_report.py` | Final experiment matrix and the report generator. |
| `hospital.py` | Imports the team's hospital drawings data (scale from the 54,675 mm dimension, floors aligned on shared shafts), loads it, hospital scenarios. |
| `fixed_plan.py` / `knowledge.py` / `session7.py` | Printed plans, imperfect knowledge of where people are, hospital procedures (Session 7). |
| `validation.py` | Validation against published measurements, IMO MSC.1/Circ.1533 tests, field-drill comparison (Session 8). |
| `ml_transfer.py` | Transfer of the learned model between buildings; two-building model v2 (Session 7). |
| `hospital_experiments.py` / `hospital_report.py` | Session 6 experiments on the hospital + patient groups, ramp weak spot, hospital assumptions. |
| `literature.py` / `literature_experiments.py` | Published baselines: quickest-path route choice and CASPER, and their experiment (Session 9). |
| `jupedsim_check.py` | Cross-check of the simulator against JuPedSim on the same geometry and people (Session 9). |
| `calibration.py` | Calibration of speed-density curve and door flow on Juelich trajectories; simulator vs experiment (Improving ML). |
| `ml_v3.py` | Crossing-time dataset from the calibrated simulator (campus, hospital, random geometries), hybrid model v3, real-data test, conformal ranges, in-the-loop test. |
| `safety_tradeoff.py` | Safety-vs-speed curve of the planner (SAFETY_WEIGHT 0-8) with every other method as a point (Phase 2). |
| `eta_study.py` | How good is "about X min to get out"? ML sum vs planner bookings vs actual, conformal ranges (Phase 2). |
| `priority_study.py` | Vulnerable-first planning (config.PRIORITY_VULNERABLE) and queue priority: who gains, who pays (Phase 2). |
| `scalability.py` | Planner compute time vs building size and crowd, incl. generated towers up to 20 floors (Phase 2). The planner's route search is A* (`planner._Fast`), same results as before, ~5x faster. |
| `jam_study.py` | Secondary jams (Choi et al. 2021): jams created by guidance that shortest path does not have; total queueing (Phase 2). |
| `compliance_study.py` | Planner without knowledge of who complies: oracle vs naive vs compliance-aware booking (Phase 2). |
| `summarize.py` | Report tables: mean + 95% bootstrap CI, paired differences vs a reference method, Wilcoxon test. |

`building.build_graph(nodes, edges)` loads **any** building from plain data — use it for the hospital.

## Methods compared so far
| Method | What it does |
| --- | --- |
| `nearest` | Shortest walking distance to any exit; ignores hazard except the hard rules. |
| `safe_shortest` | Travel time × hazard penalty (what the current system does). |
| `ml` | Existing Random Forest edge costs. |
| `reactive` | Starts as `safe_shortest`; every 5 s reroutes people whose route crosses a visible queue (≥ 8 people). No prediction. |
| `ccrp` | **Capacity booking (main contribution).** Plans everyone once, in order of readiness; each person takes the earliest-arrival route given door/stair capacity already booked by others (Capacity Constrained Route Planner, Lu et al. 2005). |
| `lookahead` | **Rolling-horizon re-planning (main contribution).** Re-books capacity every 5 s from where people really are, predicting 300 s ahead. Switches someone only if ≥ 15% faster, never by walking back. |
| `lookahead_ml` | `lookahead` with walking times from the LightGBM delay model instead of the formula. |
| `pred_formula` / `pred_ml` / `pred_ml_upper` | Same re-routing controller driven by three different delay predictors (queue formula, LightGBM, LightGBM 95% quantile). Isolates the value of prediction. |
| `ga` | Genetic algorithm exit plan (offline; hundreds of simulations). Via `evac.ga.optimise`. |
| `optimum` | **Theoretical best** last-out time (quickest flow on a time-expanded network). A lower bound — not a method anyone can follow. ~20–40 s to compute. |

## Hard safety rules (applied before any cost)
1. Edges with hazard ≥ 0.92 are removed.
2. In a fire, ordinary lifts are out of service.
3. Without fire, lifts are reserved for step-free users.
4. Step-free users never use stairs; if no step-free route exists they are counted as
   `needs_assistance` (wait at a refuge for assisted evacuation) — never sent down stairs.

## Data fix
`electrical room 2` had no edge in `campus_data.py` (it could not reach any exit). It is now linked to
its nearest corridor junction, `j19` (~5 m), in both `campus_data.py` and `campus_data.js`.
The graph is 452 nodes / 637 edges.

## Assumptions
All values in `config.py` are engineering assumptions (SFPE flow rates, Weidmann
speed–density relation), not measurements of this building.

## Scenario options (Session 4)
| Key | Meaning |
| --- | --- |
| `compliance` | share of people who follow guidance (default 1.0) |
| `herding` | chance a non-compliant person copies the majority exit of the queue they join |
| `familiar_exit` | where non-compliant people go (default `north main entrance`) |
| `fires[].spread_m_per_s`, `max_radius` | a fire that grows over time; smoke climbs stair shafts 30 s per floor |
| `log_edges` | record every edge crossing (features + actual time) for the ML model |

## Findings
Final numbers: `docs/final_results.md` (30 paired seeds, CIs, Holm-corrected Wilcoxon tests). Also `docs/ml_study.md`,
`docs/sensitivity.md`, `docs/tuning.md`. Earlier preliminary notes:
- Smarter routing only matters when crowds are concentrated (a full hall or floor). Spread over 7 floors, all methods tie.
- Reactive rerouting helps but oscillates: ~1,000 route changes for 120 people in the seminar-hall case.
- Capacity booking cuts last-out time by ~35–40% in concentrated scenarios with almost no route changes.
- **A short prediction horizon is worse than no prediction** (30–60 s horizons lost to plain routing); ≥ 300 s works.
- **Re-planning earns its place when people disobey or the fire grows**: at 60-80% compliance look-ahead beats a
  one-shot plan by 10-20%; in a spreading fire it cuts smoke exposure by ~60-70%.
- **The ML model now earns its place**: predicting real edge delay, it beats a queueing formula by ~40% MAE on
  crowded edges in unseen scenarios and fire locations.
- Stairs are the true bottleneck: under the speed–density model a 1.2 m stair sustains ~0.7 persons/s, about half
  its nominal door-style flow. The planner must use this sustainable capacity or it overbooks stairs.
