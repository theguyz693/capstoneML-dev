# Improving ML — work plan

Saved 19 Oct 2026. Start here in the next session (Claude Code): read this file first.

## Why
Mentor's criticism: the dataset is weak and the ML "just learns a formula".
- **v1 (Random Forest, `train_model.py` / `ml_routing_model.py`)** was trained on `generate_dataset.py`:
  random inputs labelled by our own cost formula (+2.5% noise), so it can only copy that formula.
- **v2 (LightGBM delay model, `evac/delay_model.py`, `evac/learned_planner.py`)** predicts how long a
  corridor/stair crossing takes, trained on simulator crossings (`results/delay_v2_*.csv`). It predicts
  well, but it learned only from our simulator and barely changes evacuation results.
- Our JuPedSim check (`docs/jupedsim_crosscheck.md`) also showed crowded 1 m doors in our simulator pass
  only ~0.8 persons/s, about half the lab value.

Goal: **one** ML model, trained on data from a simulator calibrated to **real crowd data**, tested on
real data, used by the planner. ML predicts time; fixed safety rules handle fire.

## Steps
1. **Get real data**: the Jülich Pedestrian Dynamics Data Archive (bottleneck / corridor experiments;
   see Boomers et al. 2023, Collective Dynamics, doi:10.17815/CD.2023.141, and Seyfried et al. 2009).
   If the sandbox can't download it, Pranav downloads 1–2 trajectory files into `data/juelich/`.
2. **Calibrate the simulator** (`evac/config.py`, `evac/physics.py`): fit the speed–density rule and
   door flow to the Jülich measurements; compare against Weidmann / SFPE values; pick the best fit and
   document it. This also fixes the slow-door issue.
3. **Generate a new dataset**: run the calibrated simulator (campus + hospital, fires, crowds, patients,
   partial compliance) with `log_edges`; one row per crossing (queue ahead, density, width, edge type,
   smoke, walker speed/size, planned users → actual crossing time).
4. **Train one model** (LightGBM) to predict crossing time. It replaces v1 (danger score) and v2.
   Drop the v1 formula CSV from all claims (keep only as "where we started").
5. **Test on real data**: convert Jülich crossings to the same row format, measure error; check stair
   times against NIST (Peacock et al. 2012); compare honestly with the plain formula.
6. **Plug into the planner**: edge cost = predicted time × smoke penalty; burning edges removed
   (rule, not ML); the same predictions give time-to-exit for signs; re-plan every 5 s as now.
7. **Merge the systems**: the campus "Evac Route" tab in `index.html` should use the research engine
   (`/api/research/...`) + the new model, like the hospital already does. Retire the old RF route
   (`server.py /api/evacuate`) or label it "v1".
8. **Re-run the main results** on the calibrated simulator (`evac/final_experiments.py`,
   `evac/hospital_experiments.py`, `evac/literature_experiments.py`) and report updated numbers honestly.
9. **Update documents**: paper (Claude Doc), slides deck, `Team212_Project_Summary.docx`, survey
   sections X (ML) and XIV (comparison), `docs/validation.md`. Also fix the old door-flow claim
   ("66–81% of lab values") everywhere.

## One-line claim at the end
"One ML model, trained on data from a simulator calibrated to real crowd experiments and tested on real
data, predicts how long each corridor takes; the planner uses it to route and to estimate exit times,
while safety rules handle the fire."

## Later (optional, after this)
- Fire-forecast planner (plan with where the fire will be; removes our only loss to CCRP).
- Wheelchair priority / fairness.
- One real timed drill (`docs/field_validation_protocol.md`, `python -m evac.validation --drill ...`).

## Full roadmap (agreed 19 Oct 2026)
Order: Phase 0 → Phase 1 (Improving ML, steps 1-9 above) → Phase 2 → Phase 3.

**Phase 0, before step 3 (dataset):** smoke slows walking speed (`evac/physics.py`), so the new
dataset and model learn a real smoke effect.

**Phase 1:** Improving ML, steps 1-9 above.

**Phase 2: finish the 8 half-done items**
1. Safety-vs-speed dial + time-vs-exposure curve (`HAZARD_DETOUR_PENALTY_S` in `evac/config.py`).
2. Time-to-exit range on signs (conformal intervals summed along the route; dashboard + `/api/research/signs`).
3. Secondary-jam metric for each method (extend `weakspots`).
4. One-command `reproduce_all` + public building pack.
5. Wheelchair / vulnerable-first priority in the planner (book them first); fairness figure already exists (fig13/fig19).
6. Compliance-aware planning (planner assumes some people won't follow).
7. Protected evacuation lifts with capacity for step-free users (hospital bed lifts).
8. Scalability: planning time vs building size.

**Phase 3:** fire-forecast planner (book routes using where the fire WILL be; target: remove the losses to CCRP).
Skip: groups moving together. Optional: exit blocked mid-evacuation (demo scenario).
Generality extras: random-building stress test, third building.

## Status (19 Oct 2026, Improving-ML session) - Phase 0 + Phase 1 DONE
- Phase 0 smoke slows walking: `physics.smoke_speed_factor` (Frantzich & Nilsson 2003 / FDS+Evac).
- Steps 1-2 real data + calibration: `evac/calibration.py`, `docs/calibration.md` (Juelich corridor + bottleneck runs).
- Steps 3-5 dataset, one model, real-data test: `evac/ml_v3.py`, `docs/ml_v3.md` (hybrid model v3, conformal ranges).
- Step 6 planner + time-to-exit: `learned_planner.hybrid_predict`, `route_eta`; `lookahead_ml` / `pred_ml` use v3.
- Step 7 systems merged: dashboard Evac Route -> `/api/research/route` for both buildings.
- Step 8 results re-run with calibrated parameters (old ones in `docs/archive_handbook_params/`).
- Step 9 docs: calibration.md, ml_v3.md, validation.md, final/hospital/literature reports regenerated, README.
  Still to update by hand: slides, paper (Claude Doc), Team212_Project_Summary.docx, survey sections X/XIV.
Next: Phase 2 (the 8 half-done items), then Phase 3 (fire-forecast planner).
