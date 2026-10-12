# evac — headless evacuation engine

Research engine for the capstone study. Runs thousands of seeded evacuation
simulations without the UI, so routing methods can be compared with statistics.

## Run
```bash
pip install networkx              # scikit-learn/pandas/joblib only needed for method "ml"
python -m evac.runner                                    # all scenarios, 5 seeds
python -m evac.runner --scenarios fire_j19_150 --methods nearest safe_shortest ml --seeds 30
python tests/test_evac_session1.py                       # safety + reproducibility checks
```
Results go to `results/runs.csv` (one row per run).

## Files
| File | What it does |
| --- | --- |
| `config.py` | Every physical assumption (scale, speeds, widths, flow rates). Change values here only. |
| `building.py` | Loads the building graph (same geometry as `ml_edge_updater.build_full_campus_graph`) and adds metres, widths, flow capacities. |
| `hazard.py` | Fire hazard per node/edge; smoke rises up stair shafts to higher floors. |
| `routing.py` | Hard safety rules + route methods (`nearest`, `safe_shortest`, `ml`); building-wide sign arrows. |
| `simulator.py` | Seeded agent simulation: pre-movement delay, door/stair queues, density-dependent speed, exposure tracking. |
| `runner.py` | Scenarios × methods × seeds → CSV. Seeds are paired across methods. |
| `scenarios.json` | Scenario definitions. |

## Hard safety rules (applied before any cost)
1. Edges with hazard ≥ 0.92 are removed.
2. In a fire, ordinary lifts are out of service.
3. Without fire, lifts are reserved for step-free users.
4. Step-free users never use stairs; if no step-free route exists they are counted as
   `needs_assistance` (wait at a refuge for assisted evacuation) — never sent down stairs.

## Known data issue
`electrical room 2` has no edge in `campus_data.py`, so it cannot reach any exit.
It is excluded from spawning until its connection is added.

## Assumptions
All values in `config.py` are engineering assumptions (SFPE flow rates, Weidmann
speed–density relation), not measurements of this building.
