"""
evac/physics.py
Speed-density relation shared by the simulator and the planner.
"""
import math
from . import config as C


def weidmann_speed_factor(density):
    """Fraction of free speed at a given density (persons/m^2) - Weidmann (1993)."""
    if density <= 0.05:
        return 1.0
    if density >= C.JAM_DENSITY:
        return C.MIN_SPEED_FRACTION
    f = 1.0 - math.exp(-1.913 * (1.0 / density - 1.0 / C.JAM_DENSITY))
    return max(C.MIN_SPEED_FRACTION, min(1.0, f))


def _peak_flow_fraction():
    """max over density of density x speed factor (persons/s per metre per (m/s) of free speed)."""
    return max((r / 100) * weidmann_speed_factor(r / 100) for r in range(1, int(C.JAM_DENSITY * 100)))


PEAK_FLOW_PER_SPEED = _peak_flow_fraction()   # ~0.91 persons/m^2 at ~1.75 persons/m^2


def sustainable_flow(width_m, free_speed):
    """Highest steady flow (persons/s) an edge can carry under the speed-density relation."""
    return width_m * free_speed * PEAK_FLOW_PER_SPEED
