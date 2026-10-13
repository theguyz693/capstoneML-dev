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

# ---------------------------------------------------------------- speeds
FREE_WALK_SPEED = 1.25        # m/s, corridors/doors, free flow
STAIR_DOWN_SPEED = 0.65       # m/s along the stair slope (descending)
# Floor-to-floor ramp (hospital). Drawings: two 8 m runs + 2 m landings each way per floor (switchback)
RAMP_LENGTH_PER_FLOOR_M = 40.0
RAMP_SPEED = 1.0              # m/s walking down a ~1:12 ramp (ASSUMPTION)
MIN_SPEED_FRACTION = 0.10     # agents never fully freeze while on an edge
JAM_DENSITY = 5.4             # persons/m^2 (Weidmann)
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
