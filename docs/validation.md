# Validation: how close is the simulator to real people? (Session 8)

Verification (`docs/verification/`) showed the code implements its model. Validation asks whether the model's numbers match real crowds. Every published value below is cited.

## Findings

1. **Door flow is conservative.** The simulator passes 1.3 persons/s per metre (the SFPE engineering value, which the IMO test caps at 1.33): 66%-81% of the flow measured in laboratory bottleneck experiments with young adults (Seyfried et al. 2009). Real crowds can push more people through a door, so our evacuation times err on the long (safe) side.
2. **Stair speed matches real drills.** Across all simulated stair crossings the mean speed is 0.41 +- 0.18 m/s; NIST measured 0.48 +- 0.16 m/s in fire drills in four office buildings (about 3,000 evacuees analysed). Uncrowded, our people descend at 0.48 m/s, inside the handbook range (Fruin 0.48-0.76 m/s).
3. **Standard model tests:** IMO 1 - 40 m corridor, 1 person: PASS (33.5 s); IMO 4 - 100 persons through a 1 m door: PASS (1.30 persons/s); IMO 9 - 1000 persons, 4 doors vs 2 doors: PASS (243.5 s vs 467.0 s (ratio 1.92)).
4. **The conclusion survives measured values.** With door flow and stair speed set to the measured means, look-ahead still beats shortest path by 18%-35% (30 of 30 paired runs); with slow doors and slow stairs by 23%-50% (30 of 30).
5. **Field drill ready:** a 3-person stopwatch protocol (`docs/field_validation_protocol.md`) and `python -m evac.validation --drill` turn one timed real exit into a measured-vs-simulated comparison.

![Validation](figures/fig25_validation.png)

## V1. Flow through a door / bottleneck (persons per second)

| Width (m) | Our simulator | Seyfried et al. 2009 | Kretz 2006 | Muller | Nagai | IMO max (1.33 x w) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8 | 1.04 | 1.29 | 1.43 | - | 2.58 | 1.06 |
| 0.9 | 1.17 | 1.67 | 1.58 | 2.26 | - | 1.20 |
| 1.0 | 1.30 | 1.90 | 1.85 | - | - | 1.33 |
| 1.1 | 1.43 | 2.12 | - | - | - | 1.46 |
| 1.2 | 1.56 | 2.36 | 2.15 | 2.75 | 3.27 | 1.60 |
| 1.5 | 1.96 | - | - | 3.59 | - | 2.00 |
| 1.6 | 2.06 | - | - | - | 3.86 | 2.13 |

Lab values: Seyfried et al. (2009), *New insights into pedestrian flow through bottlenecks*, Transportation Science 43(3):395-406 (Table 2, dN = 60; Table 1 for the other groups). Higher values come from dense, motivated crowds; the 1.3 p/s/m engineering value is deliberately conservative.

## V2. Stair descent speed (m/s)

| Source | Mean | SD |
| --- | --- | --- |
| NIST drill, 10-storey | 0.44 | 0.19 |
| NIST drill, 18-storey | 0.44 | 0.15 |
| NIST drill, 24-storey | 0.56 | 0.12 |
| NIST drill, 31-storey | 0.52 | 0.1 |
| NIST drill, all 4 buildings | 0.48 | 0.16 |
| Fruin, optimum | 0.48 | - |
| Fruin, moderate | 0.61 | - |
| Fruin, maximum | 0.76 | - |
| Predtechenskii & Milinskii | 0.58 | - |
| Hostikka et al. 2007 | 0.64 | - |
| **Our simulator, uncrowded crossings** | 0.48 | 0.14 |
| **Our simulator, all 5,182 crossings** | 0.41 | 0.18 |

NIST: Peacock, Hoskins & Kuligowski (2012), *Overall and local movement speeds during fire drill evacuations in buildings up to 31 stories*, Safety Science 50:1655-1664, Table 3 (literature values from its Table 1). Simulated speed = flight length / time from joining the stair to leaving it (crowding and stair queues included).

## V3. IMO MSC.1/Circ.1533 verification tests

| Test | Expected | Result | |
| --- | --- | --- | --- |
| IMO 1 - 40 m corridor, 1 person | 32.0 s (40 m / 1.25 m/s) | 33.5 s | PASS |
| IMO 4 - 100 persons through a 1 m door | mean flow <= 1.33 persons/s | 1.30 persons/s | PASS |
| IMO 9 - 1000 persons, 4 doors vs 2 doors | time ratio about 2 | 243.5 s vs 467.0 s (ratio 1.92) | PASS |

IMO 9 uses the one-shot capacity planner for door choice: in a network model the whole room is one node, so with plain shortest path all 1000 people tie for the same nearest door (917 s whatever the number of doors) - itself a small demonstration of the problem this project solves.

Tests that need 2-D geometry (rounding corners, counter-flow in a corridor) do not apply to a network (graph) model and are listed as a limitation.

## V4. Does the conclusion survive measured values?

| Assumption set | Scenario | Shortest (s) | Look-ahead (s) | Gain | Look-ahead faster in |
| --- | --- | --- | --- | --- | --- |
| our assumptions | floor3_full_200 | 357 | 208 | 42% | 10/10 seeds |
| our assumptions | lecture_hall_f1_120 | 305 | 209 | 31% | 10/10 seeds |
| our assumptions | floor3_200_compliance60_herding | 409 | 299 | 27% | 10/10 seeds |
| measured values (lab door flow, NIST stair speed) | floor3_full_200 | 454 | 301 | 34% | 10/10 seeds |
| measured values (lab door flow, NIST stair speed) | lecture_hall_f1_120 | 395 | 256 | 35% | 10/10 seeds |
| measured values (lab door flow, NIST stair speed) | floor3_200_compliance60_herding | 506 | 414 | 18% | 10/10 seeds |
| conservative (slow doors, slow stairs) | floor3_full_200 | 491 | 298 | 39% | 10/10 seeds |
| conservative (slow doors, slow stairs) | lecture_hall_f1_120 | 455 | 227 | 50% | 10/10 seeds |
| conservative (slow doors, slow stairs) | floor3_200_compliance60_herding | 542 | 417 | 23% | 10/10 seeds |

## V5. Real drill

See `docs/field_validation_protocol.md`. No real drill has been timed yet; when one is, its comparison goes here.
