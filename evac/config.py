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
}

# ---------------------------------------------------------------- flow
# Specific flow (persons / second / metre of width).
SPECIFIC_FLOW = {
    "corridor": 1.3,
    "room_door": 1.3,
    "exit_door": 1.3,
    "stair": 1.1,
    "lift": 0.3,
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

# ---------------------------------------------------------------- simulation
DT = 0.5                      # seconds per simulation step
MAX_TIME_S = 1800.0           # stop after 30 minutes of simulated time
PRE_MOVEMENT_MAX_S = 30.0     # each agent starts after U(0, this) seconds
