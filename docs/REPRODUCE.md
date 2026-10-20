# Reproducing every result

All results in `docs/` come from seeded simulations: the same code gives the same numbers on any machine.

## One command

```bash
pip install -r requirements.txt          # networkx, numpy, pandas, scipy, matplotlib, lightgbm, scikit-learn, fastapi
python -m evac.reproduce_all             # every step, re-using finished runs in results/
python -m evac.reproduce_all --fresh     # start from scratch (old runs are MOVED to results/_previous/<time>/, not deleted)
python -m evac.reproduce_all --list      # the steps and their run times
python -m evac.reproduce_all --only eta jams      # some steps
python -m evac.reproduce_all --from safety        # a step and everything after it
```

A full run from scratch takes about 4 hours on a 2-core laptop; with the stored runs in `results/` it only
regenerates the reports and figures (a few minutes). A log is appended to `results/reproduce_log.txt`. The test
files in `tests/` run at the end.

## Steps

| Step | Command | ~min (2 cores) | Produces |
| --- | --- | --- | --- |
| calibration | `python -m evac.calibration` | 1 | simulator calibrated on Juelich crowd experiments -> docs/calibration.md |
| ml | `python -m evac.ml_v3` | 70 | crossing-time dataset, ML v3, real-data test -> docs/ml_v3.md, models/delay_lgbm_v3* |
| main | `python -m evac.final_experiments` | 40 | campus: 6 methods x 10 scenarios x 30 seeds, ML in the loop, GA, lower bound, tipping point, freshness, ablations |
| main_report | `python -m evac.final_report` | 1 | docs/final_results.md + figures 12-17 |
| hospital | `python -m evac.hospital_experiments` | 15 | hospital: methods, lower bound, assumptions, weak spots |
| hospital_report | `python -m evac.hospital_report` | 1 | docs/hospital_results.md + figures 18-21 |
| literature | `python -m evac.literature_experiments` | 12 | published baselines: quickest path, CASPER -> docs/literature_baselines.md |
| validation | `python -m evac.validation` | 6 | door flow, stairs (NIST), IMO tests -> docs/validation.md |
| safety | `python -m evac.safety_tradeoff` | 25 | safety-vs-speed curve -> docs/safety_tradeoff.md |
| eta | `python -m evac.eta_study` | 6 | time-to-exit on signs -> docs/eta_study.md |
| priority | `python -m evac.priority_study` | 15 | vulnerable people first -> docs/priority.md |
| scalability | `python -m evac.scalability` | 20 | compute time vs building size -> docs/scalability.md |
| jams | `python -m evac.jam_study` | 8 | secondary jams -> docs/jam_study.md |
| compliance | `python -m evac.compliance_study` | 15 | unknown compliance -> docs/compliance.md |
| lifts | `python -m evac.lift_study` | 10 | evacuation lifts -> docs/lifts.md |
| pack | `python -m evac.building_pack export` | 1 | building pack: buildings/campus.json, hospital.json |

## What is NOT part of the one command

- `python -m evac.jupedsim_check` (needs `pip install jupedsim shapely`; ~1 h): the JuPedSim cross-check, run once with the
  handbook parameters (docs/jupedsim_crosscheck.md).
- Earlier-session analyses kept as they were: `evac.sensitivity`, `evac.tune`, `evac.weakspots`, `evac.ml_study`,
  `evac.ml_transfer`, `evac.session7` (their reports are in docs/).
- The real-data sample `data/juelich/` is included; check the Juelich data archive's licence/citation before publishing.

## Parameter sets

`evac/config.py`: `PARAM_SET = "juelich"` (calibrated, default) or `"sfpe"` (the earlier handbook values; the reports
from before the calibration are in `docs/archive_handbook_params/`). Every Phase-2 switch (`SAFETY_WEIGHT`,
`PRIORITY_VULNERABLE`, `PLANNER_COMPLIANCE_MODEL`, `EVAC_LIFTS`, ...) is off / at its default unless a study sets it.
