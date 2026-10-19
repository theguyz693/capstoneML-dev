"""
evac/config.py
All physical and behavioural assumptions in ONE place.

Every value here is an ASSUMPTION used for simulation, not a measurement of the
real building. Values are chosen from common fire-engineering references so the
simulator behaves plausibly; change them here and every experiment picks them up.

References (approximate engineering values):
  - Walking speed ~1.2-1.34 m/s in free flow; speed falls with density
    (Weidmann 1993; Seyfried et al. 2005, "The fundamental diagram of pedestrian
    movement revisited", J. Stat. Mech. P10002).
  - Specific flow through doors/corridors ~1.3 persons/s per metre of effective
    width; stairs somewhat lower (SFPE Handbook, hydraulic egress model).
  - Jam density ~5.4 persons/m^2 (Weidmann 1993).
"""

# ---------------------------------------------------------------- geometry
# Graph coordinates are floor-plan pixels. The building is ~2100 units across,
# so 1 unit ~= 0.04 m gives ~85 m. ASSUMPTION - replace with a measured scale.
METERS_PER_UNIT = 0.04

# Inter-floor stair edge: walking length of one flight + landing (m).
STAIR_FLIGHT_LENGTH_M = 8.0

# ---------------------------------------------------------------- parameter sets (Improving ML, step 2)
# PARAM_SET = "juelich": walking speed, speed-density curve and door flow CALIBRATED on real crowd
#             experiments from Forschungszentrum Juelich (evac/calibration.py -> evac/calibrated_params.json,
#             docs/calibration.md). Default from the Improving-ML session on.
# PARAM_SET = "sfpe":    the earlier engineering-handbook values (Weidmann 1993 / SFPE), more conservative.
#             All results before the Improving-ML session used these.
PARAM_SET = "juelich"

# ---------------------------------------------------------------- speeds
FREE_WALK_SPEED = 1.25        # m/s, corridors/doors, free flow
STAIR_DOWN_SPEED = 0.65       # m/s along the stair slope (descending)
# Floor-to-floor ramp (hospital). Drawings: two 8 m runs + 2 m landings each way per floor (switchback)
RAMP_LENGTH_PER_FLOOR_M = 40.0
RAMP_SPEED = 1.0              # m/s walking down a ~1:12 ramp (ASSUMPTION)
MIN_SPEED_FRACTION = 0.10     # agents never fully freeze while on an edge
JAM_DENSITY = 5.4             # persons/m^2 (Weidmann)
FD_GAMMA = 1.913              # shape of the speed-density curve (Weidmann 1993)
AGENT_SPEED_SPREAD = 0.15     # +/- 15% individual speed variation

# ---------------------------------------------------------------- widths (m)
# Effective clear widths by edge type. ASSUMPTIONS until measured.
WIDTH_M = {
    "corridor": 2.0,
    "room_door": 1.0,
    "exit_door": 1.8,
    "stair": 1.2,
    "lift": 1.0,
    "ramp": 2.1,      # hospital switchback ramp: 2125 mm clear width on the drawings
}

# Width used for CROWD DENSITY on door edges. A door edge runs from the middle of a room to the corridor:
# most of it is open floor, the door itself is a short bottleneck at the end. People queue in front of the
# door (node queue), so density on the edge is measured over a corridor-like width, while the DOOR limits
# how fast people get on (flow_cap). None = use the door width (the pre-calibration behaviour).
DOOR_APPROACH_WIDTH_M = None

# ---------------------------------------------------------------- flow
# Specific flow (persons / second / metre of width).
SPECIFIC_FLOW = {
    "corridor": 1.3,
    "room_door": 1.3,
    "exit_door": 1.3,
    "stair": 1.1,
    "lift": 0.3,
    "ramp": 1.3,      # treated like a level corridor (gentle 1:12 slope)
}

# ---------------------------------------------------------------- hazards
HAZARD_IMPASSABLE = 0.92      # edges at/above this hazard are removed (matches ml_routing_model)
HAZARD_EXPOSURE = 0.30        # crossing an edge at/above this counts as "exposed"
SMOKE_STAIR_DECAY = 0.80      # smoke in a stair shaft keeps 80% intensity per floor going UP
SMOKE_STAIR_MAX_FLOORS = 6

# Smoke slows walking (Phase 0, Improving ML). Frantzich & Nilsson (2003) measured walking speed falling
# linearly with the smoke extinction coefficient Ks (1/m); FDS+Evac uses v = v0 * max(0.1, 1 + (beta/alpha) Ks)
# with alpha = 0.706, beta = -0.057 (see Korhonen, FDS+Evac guide; Kasperski et al. A-Evac 2017).
# Our hazard is a 0-1 index, so we ASSUME hazard 1.0 ~ dense smoke with Ks = 10 /m (visibility ~0.2-0.3 m).
SMOKE_SLOWS_WALKING = True
SMOKE_KS_AT_FULL_HAZARD = 10.0   # ASSUMPTION: extinction coefficient (1/m) at hazard 1.0
SMOKE_SPEED_SLOPE = -0.057 / 0.706   # per 1/m of extinction coefficient (Frantzich & Nilsson 2003 via FDS+Evac)
SMOKE_MIN_FRACTION = 0.10        # people never stop completely in smoke (FDS+Evac)

# Lift rules (common building practice):
#  - in a FIRE, ordinary lifts are out of service for everyone
#  - in any other evacuation, lifts are reserved for step-free (mobility-impaired) users
LIFTS_DISABLED_IN_FIRE = True
LIFTS_ONLY_FOR_STEP_FREE = True

# ---------------------------------------------------------------- reactive rerouting
# The 'reactive' method reroutes people only AFTER congestion is visible:
# every REROUTE_INTERVAL_S, anyone standing at a node whose remaining route passes a
# queue of >= REACTIVE_QUEUE_THRESHOLD people is re-planned with current queue waits.
REROUTE_INTERVAL_S = 5.0
REACTIVE_QUEUE_THRESHOLD = 8

# ---------------------------------------------------------------- capacity-aware planning
SLOT_S = 4.0                  # capacity booking slot length (s): an edge admits flow_cap*SLOT_S per slot
REPLAN_S = 5.0                # 'lookahead' re-plans every 5 s
HORIZON_S = 300.0             # bookings predicted this far ahead (s). Short horizons (<=120 s) performed WORSE than no planning in tests.
SWITCH_GAIN = 0.15            # switch route only if predicted >= 15% faster (anti-oscillation)
SAFETY_WEIGHT = 1.0           # Phase 2: safety-vs-speed dial for the capacity planner (ccrp / lookahead). 0 = only the
                              # hard rules (never walk into fire); 1 = default; >1 = avoid smoke even at a big time cost.
                              # Scales both the smoke multiplier and HAZARD_DETOUR_PENALTY_S (evac/safety_tradeoff.py)
PRIORITY_VULNERABLE = False   # Phase 2: the capacity planner books slow walkers, wheelchair users, beds and other
                              # step-free people FIRST, so their routes get the earliest door/stair slots
DOOR_PRIORITY_VULNERABLE = False  # ... and at a door/stair queue they are let through first (staff / marshals)
HAZARD_DETOUR_PENALTY_S = 300.0  # planner adds this per edge with hazard >= HAZARD_EXPOSURE, so it only
                                 # trades safety for speed when waiting would exceed ~5 min

# Ablation switches (Session 6): turn one ingredient of the planner off to measure its contribution
PLANNER_USE_SUSTAINABLE_CAP = True   # False: book nominal door/stair flow instead of what the edge can sustain
PLANNER_BOOK_NONCOMPLIANT = True     # False: planner ignores where non-compliant people are heading

# ---------------------------------------------------------------- behaviour (Session 4)
# These are SWEPT in experiments, not claimed to be measured human behaviour.
COMPLIANCE = 1.0              # share of people who follow the system's route
HERDING = 0.0                 # chance a non-compliant person copies the majority exit of the queue they join
FAMILIAR_EXIT = "north main entrance"   # where non-compliant people head by default (the way they came in)

# ---------------------------------------------------------------- spreading fire (Session 4)
JAM_QUEUE = 10                # a 'jam' = at least this many people waiting at one node ...
JAM_MIN_S = 10.0              # ... for at least this many seconds in total (secondary-jam study, evac/jam_study.py)
HAZARD_UPDATE_S = 10.0        # recompute the hazard field this often when a fire spreads
STAIR_SMOKE_DELAY_S = 30.0    # in a spreading fire, smoke needs ~30 s per floor to climb a stair shaft

# ---------------------------------------------------------------- simulation
DT = 0.5                      # seconds per simulation step
MAX_TIME_S = 1800.0           # stop after 30 minutes of simulated time
PRE_MOVEMENT_MAX_S = 30.0     # each agent starts after U(0, this) seconds


# ---------------------------------------------------------------- occupant profiles (hospital)
# Used by scenarios with "spawn": "groups". ASSUMPTIONS for comparison purposes, in the range reported
# for hospital evacuation studies (wheelchair ~0.7-0.9 m/s, assisted bed movement ~0.5 m/s, bed occupies
# roughly 4 person-equivalents of corridor/door capacity, beds need preparation before moving).
#   speed      multiplier on walking speed
#   step_free  cannot use stairs (ramp / lifts only)
#   size       person-equivalents of door/corridor capacity used
#   pre_s      (min, max) pre-movement time in seconds
PROFILES = {
    "walk":       {"speed": 1.0, "step_free": False, "size": 1.0, "pre_s": (0, 30)},    # staff, visitors
    "slow":       {"speed": 0.6, "step_free": False, "size": 1.0, "pre_s": (10, 60)},   # walking patients, elderly
    "wheelchair": {"speed": 0.7, "step_free": True,  "size": 2.0, "pre_s": (20, 90)},
    "bed":        {"speed": 0.5, "step_free": True,  "size": 4.0, "pre_s": (60, 240)},  # bed pushed by staff
}

# ---------------------------------------------------------------- hospital procedures (Session 7)
COMPARTMENT_LEAK = 0.15       # hazard reaching the next fire compartment through closed fire doors (light smoke)
STAFF_PER_BED = 2             # staff needed to move one bed
STAFF_WALK_SPEED = 1.25       # m/s, staff walking back for the next bed

# ---------------------------------------------------------------- learned delay model version (Session 7)
# v1 = trained on the campus only (Session 6); v2 = campus + hospital, with ramp and mover (speed, size) features
# (evac/ml_transfer.py); v3 = Improving ML: trained on the CALIBRATED simulator (campus + hospital + random
# geometries, smoke slows walking), tested on real Juelich crossings, with 5 %/95 % quantile models (evac/ml_v3.py)
DELAY_MODEL = "v3"


# ---------------------------------------------------------------- apply the parameter set
_SFPE = {"FREE_WALK_SPEED": FREE_WALK_SPEED, "JAM_DENSITY": JAM_DENSITY, "FD_GAMMA": FD_GAMMA,
         "SPECIFIC_FLOW": dict(SPECIFIC_FLOW), "DOOR_APPROACH_WIDTH_M": DOOR_APPROACH_WIDTH_M}


def use_params(name):
    """Switch every module to a parameter set ('juelich' or 'sfpe'). Graphs built BEFORE the switch keep the
    old edge physics - rebuild them (load_building) after calling this."""
    import json
    import os
    g = globals()
    for k, v in _SFPE.items():
        g[k] = dict(v) if isinstance(v, dict) else v
    if name == "juelich":
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrated_params.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cal = json.load(f)["params"]
            for k, v in cal.items():
                if k == "SPECIFIC_FLOW":
                    g["SPECIFIC_FLOW"].update(v)
                else:
                    g[k] = v
        else:
            name = "sfpe"          # not calibrated yet (run: python -m evac.calibration)
    g["PARAM_SET"] = name
    from . import physics
    physics.refresh()
    return name


use_params(PARAM_SET)
