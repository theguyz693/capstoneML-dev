# Does the learned model transfer between buildings? (Session 7)

## Findings

1. Campus-trained model on the hospital: MAE 8.52 s (formula 5.03 s) - worse than the formula; on slow movers 10.4 s vs 5.8 s.
2. Trained on BOTH buildings with mover speed/size and ramp features (v2): hospital MAE 2.87 s (slow movers 3.2 s) - now better than the formula - and campus accuracy does not suffer (1.55 s vs 1.94 s for the campus-only model).
3. In the planner, vs the formula: hospital v1 +8.1% -> v2 +2.9%; campus v1 +1.9% -> v2 -0.5%. Retraining shrinks the transfer penalty from +8.1% to +2.9%, but the plain formula is still as good inside the planner: a more accurate predictor does not make the plan much better - coordination does. Lesson: a learned component must be retrained for each new kind of building, the formula needs no training.

![Transfer](figures/fig24_ml_transfer.png)

Edge-delay prediction error (MAE, seconds) on evacuations never used for training. 'Crowded' = someone already queued for the edge; 'slow movers' = patients, wheelchairs, beds (hospital).

| Test set | Trained on | Model | MAE all | MAE crowded | MAE slow movers |
| --- | --- | --- | --- | --- | --- |
| campus (held-out evacuations) | - | queueing formula (no training) | 2.90 | 9.32 | 3.36 |
| campus (held-out evacuations) | campus only | LightGBM, v1 features | 1.94 | 5.34 | 2.43 |
| campus (held-out evacuations) | campus only | LightGBM, v2 features | 1.47 | 4.83 | 1.63 |
| campus (held-out evacuations) | hospital only | LightGBM, v1 features | 8.22 | 25.21 | 7.65 |
| campus (held-out evacuations) | hospital only | LightGBM, v2 features | 4.97 | 18.60 | 5.20 |
| campus (held-out evacuations) | both buildings | LightGBM, v1 features | 2.40 | 6.33 | 2.49 |
| campus (held-out evacuations) | both buildings | LightGBM, v2 features | 1.55 | 5.06 | 1.79 |
| hospital (held-out evacuations) | - | queueing formula (no training) | 5.03 | 21.05 | 5.77 |
| hospital (held-out evacuations) | campus only | LightGBM, v1 features | 8.52 | 24.71 | 10.41 |
| hospital (held-out evacuations) | campus only | LightGBM, v2 features | 7.81 | 24.35 | 9.58 |
| hospital (held-out evacuations) | hospital only | LightGBM, v1 features | 4.41 | 13.32 | 4.46 |
| hospital (held-out evacuations) | hospital only | LightGBM, v2 features | 2.83 | 10.98 | 3.18 |
| hospital (held-out evacuations) | both buildings | LightGBM, v1 features | 4.45 | 13.20 | 4.73 |
| hospital (held-out evacuations) | both buildings | LightGBM, v2 features | 2.87 | 11.00 | 3.23 |

## In the planner

Look-ahead re-planning with walking times from the formula, the campus-only model (v1) or the two-building model (v2). Mean of 10 seeds; hospital = last WALKING person out, campus = last person out.

| Building | Scenario | Formula | ML v1 (campus only) | ML v2 (both buildings) |
| --- | --- | --- | --- | --- |
| campus | floor3_200_spreading_fire_stairs1 | 294 | 310 | 304 |
| campus | floor3_full_200 | 208 | 215 | 208 |
| campus | lecture_hall_f1_120 | 209 | 203 | 200 |
| hospital | hosp_day_compliance70 | 249 | 256 | 264 |
| hospital | hosp_day_fire_lobby | 404 | 427 | 407 |
| hospital | hosp_day_fire_stairs1_f2 | 280 | 336 | 270 |
| hospital | hosp_day_peak | 219 | 229 | 231 |
| hospital | hosp_day_spreading_f2 | 225 | 257 | 243 |
| hospital | hosp_night_wards_fire | 198 | 199 | 198 |
