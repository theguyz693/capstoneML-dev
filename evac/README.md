# evac — headless evacuation engine

Research engine for the capstone study. Runs thousands of seeded evacuation
simulations without the UI, so routing methods can be compared with statistics.

## Run
```bash
pip install networkx              # scikit-learn/pandas/joblib only needed for method "ml"
python -m evac.runner                                    # all scenarios, 5 seeds
python -m evac.runner --scenarios fire_j19_150 --methods nearest safe_shortest ml --seeds 30
python -m evac.runner --methods nearest safe_shortest reactive --seeds 10 --out results/session2_runs.csv
python -m evac.summarize results/session2_runs.csv      # means, 95% CIs, paired tests -> *_summary.md
python -m evac.verification                             # simulator sanity tests -> docs/verification/
python tests/test_evac_session1.py                      # safety + reproducibility checks
python tests/test_evac_session2.py                      # verification + reactive method checks
```
Results go to `results/` (one row per run). `scipy` is optional (Wilcoxon p-values).

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

## Findings so far (preliminary, see `docs/`)
- Smarter routing only matters when crowds are concentrated (a full hall or floor). Spread over 7 floors, all methods tie.
- Reactive rerouting helps but oscillates: ~1,000 route changes for 120 people in the seminar-hall case.
- Capacity booking cuts last-out time by ~35–40% in concentrated scenarios with almost no route changes.
- **A short prediction horizon is worse than no prediction** (30–60 s horizons lost to plain routing); ≥ 300 s works.
- Stairs are the true bottleneck: under the speed–density model a 1.2 m stair sustains ~0.7 persons/s, about half
  its nominal door-style flow. The planner must use this sustainable capacity or it overbooks stairs.
