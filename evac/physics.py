"""
evac/physics.py
Speed-density relation, smoke slowdown and edge capacities shared by the simulator, the planners and the
learned model. Parameters live in evac/config.py (calibrated on Juelich experiments, see evac/calibration.py).
"""
import math
from . import config as C


def weidmann_speed_factor(density):
    """Fraction of free speed at a given density (persons/m^2): Weidmann (1993) form,
    v/v0 = 1 - exp(-gamma (1/rho - 1/rho_max)); gamma and rho_max are fitted to Juelich data."""
    if density <= 0.05:
        return 1.0
    if density >= C.JAM_DENSITY:
        return C.MIN_SPEED_FRACTION
    f = 1.0 - math.exp(-C.FD_GAMMA * (1.0 / density - 1.0 / C.JAM_DENSITY))
    return max(C.MIN_SPEED_FRACTION, min(1.0, f))


def smoke_speed_factor(hazard):
    """Fraction of normal walking speed left in smoke (Frantzich & Nilsson 2003, as used in FDS+Evac)."""
    if not C.SMOKE_SLOWS_WALKING or hazard <= 0.0:
        return 1.0
    ks = C.SMOKE_KS_AT_FULL_HAZARD * min(1.0, hazard)
    return max(C.SMOKE_MIN_FRACTION, 1.0 + C.SMOKE_SPEED_SLOPE * ks)


def _peak_flow_fraction():
    """max over density of density x speed factor (persons/s per metre per (m/s) of free speed)."""
    return max((r / 100) * weidmann_speed_factor(r / 100) for r in range(1, int(C.JAM_DENSITY * 100)))


PEAK_FLOW_PER_SPEED = _peak_flow_fraction()


def refresh():
    """Recompute derived constants after the parameter set changes (config.use_params)."""
    global PEAK_FLOW_PER_SPEED
    PEAK_FLOW_PER_SPEED = _peak_flow_fraction()


def sustainable_flow(width_m, free_speed):
    """Highest steady flow (persons/s) an edge can carry under the speed-density relation."""
    return width_m * free_speed * PEAK_FLOW_PER_SPEED


def density_width(etype, width_m):
    """Width over which crowd density is measured on an edge (see config.DOOR_APPROACH_WIDTH_M)."""
    if etype in ("room_door", "exit_door") and C.DOOR_APPROACH_WIDTH_M:
        return max(width_m, C.DOOR_APPROACH_WIDTH_M)
    return width_m


def edge_density(d, occupants):
    return occupants / (d["length_m"] * d.get("dens_width_m", d["width_m"]))
