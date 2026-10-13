# Field drill protocol (stopwatch validation)

Goal: compare the simulator with a real, timed evacuation of one room - the cheapest real-world validation.

**Before:** pick a room the model knows (e.g. `f1_sem hall 1`), get permission (it is a normal fire drill /
end-of-class exit), measure the door width with a tape, note the exit the class normally uses.

**During (3 people):**
1. Counter - counts the people in the room just before the signal.
2. Timer A at the room door - starts the stopwatch at the signal, records when the FIRST and LAST person
   pass the room door.
3. Timer B at the building exit - records when the last person from the room leaves the building.
Optional: phone video of the door (gives the flow through the door, people per second).

**After:** fill one row per drill in `data/field_drills.csv` (template: `docs/field_drill_template.csv`) and run

    python -m evac.validation --drill data/field_drills.csv

The script simulates the same room with the same head-count (10 seeds, people start moving at the
signal plus 0-10 s) and prints simulated vs measured times and the percentage error.

**How to read it:** within +-20% is good agreement for an engineering model; a consistent bias (e.g. the
model always 30% fast) tells which assumption to adjust (door flow or walking speed) - and the sensitivity
analysis already shows the method ranking does not change across those ranges.
