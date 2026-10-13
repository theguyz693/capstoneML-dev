"""
evac/knowledge.py
What the planner KNOWS about where people are (Session 7).

So far the planner was handed everyone's exact position for free. A real building only has
estimates: without IoT, the timetable / bed list ("60 students in room 301"); with IoT, camera
or Wi-Fi head-counts that are noisy and sometimes missing. Two ways of modelling that:

1. Imperfect live counts (IoT-style), scenario keys:
     sense_miss        share of real people the system does not see at each update   (undercount)
     sense_false       phantom people per real person seen, at each update            (overcount)
     sense_dead_floor  a floor whose sensors are down: nobody on it is seen
   The planner books capacity for what it sees (including phantoms) and re-plans only the real
   people it sees; unseen people keep walking their current route and still use real capacity.

2. No live data at all (no IoT): see evac/fixed_plan.py - plan ONCE from the timetable estimate and
   print it on the wall.

Both leave the physical simulation untouched: the real crowd always behaves as it really is.
"""

import random


def _clone(a, new_id):
    """A phantom copy of agent `a` (same place, route and timing) - never simulated, only planned."""
    g = object.__new__(type(a))
    for s in type(a).__slots__:
        setattr(g, s, getattr(a, s))
    g.id = new_id
    g.path = list(a.path) if a.path is not None else None
    return g


class Sensor:
    def __init__(self, scenario, seed, G):
        self.miss = float(scenario.get("sense_miss", 0.0))
        self.false = float(scenario.get("sense_false", 0.0))
        dead = scenario.get("sense_dead_floor")
        self.dead = None if dead is None else int(dead)
        self.rng = random.Random(f"sense-{seed}")
        self.G = G
        self._next_id = -1

    @property
    def active(self):
        return self.miss > 0 or self.false > 0 or self.dead is not None

    def _floor(self, a):
        node = a.path[a.idx] if a.path is not None else a.origin
        return self.G.nodes[node]["floor"]

    def sense(self, people):
        """The planner's view of `people`: the ones it detects plus phantom copies."""
        if not self.active:
            return list(people)
        seen = [a for a in people
                if (self.dead is None or self._floor(a) != self.dead) and self.rng.random() >= self.miss]
        out = list(seen)
        if self.false > 0 and seen:
            k = sum(1 for _ in seen if self.rng.random() < self.false)
            for _ in range(k):
                out.append(_clone(seen[self.rng.randrange(len(seen))], self._next_id))
                self._next_id -= 1
        return out
