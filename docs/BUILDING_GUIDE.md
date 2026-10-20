# Adding your own building (building pack)

The planner, simulator, ML model and dashboards work on **any** building described as a graph. A building is one
JSON file in `buildings/`. Two real buildings ship as examples (`campus.json`, `hospital.json`) plus a small
`example_office.json` to copy.

## 1. Draw the graph (on top of the floor plan)

Put a point (node) on:
- every **room** (its middle) - people start here;
- every **corridor junction / bend** - name it `j1`, `j2`, ... or `<floor>_j1` (anything starting with `j` or containing `_j`);
- every **stair** landing on every floor - name contains `stairs` (e.g. `f1_stairs 2`); the same stair on two floors
  must share the number so the shaft is recognised;
- every **lift** (`lift` in the name) and **ramp** (kind `ramp`);
- every **exit** to the outside - name contains `exit` or `entrance`.

Join two points with an edge where a person can walk directly between them (room -> its door's corridor junction,
junction -> junction along a corridor, stair -> the same stair one floor down, junction -> exit).

## 2. Write the file (metres)

```json
{
  "name": "my_building",
  "units": "m",
  "nodes": {"g_office 1": [5, 6, 0], "g_j1": [5, 0, 0], "g_stairs 1": [-1, 2, 0], "f1_stairs 1": [-1, 2, 1],
            "west exit": [-4, 0, 0]},
  "edges": [["g_office 1", "g_j1"], ["g_j1", "g_stairs 1"], ["g_stairs 1", "f1_stairs 1"], ["g_j1", "west exit"]],
  "kinds":  {"g_ramp": "ramp"},
  "widths": {"g_j1|west exit": 1.2}
}
```

- `nodes`: x, y in metres (any origin), floor number (0 = ground). Measure from the drawings or count tiles.
- `kinds` (optional): only for names the rules above get wrong (`room`, `junction`, `stair`, `lift`, `ramp`, `exit`).
- `widths` (optional): clear width in metres for edges that differ from the defaults
  (corridor 2.0, room door 1.0, exit door 1.8, stair 1.2, ramp 2.1 - `evac/config.py`, `WIDTH_M`).
- Edge lengths come from the coordinates; a stair flight is 8 m, a ramp 40 m per floor (config).

## 3. Check and run it

```bash
python -m evac.building_pack check buildings/my_building.json       # errors (unusable) and warnings (risky)
python -m evac.building_pack run   buildings/my_building.json --people 200 --fire g_j1
python research_server.py          # Research Lab -> building menu lists every file in buildings/
```

`check` reports: no exits, nodes that cannot reach an exit, floors without a way down, floors with a single
stair (no alternative if it is lost), rooms with no step-free way out in a fire, overlapping nodes.
`run` compares shortest path with the look-ahead planner on that building (people spread over all rooms).

In Python: `from evac.building import load_building; G = load_building("my_building")` - every study and method
(`evac.simulator.run(G, scenario, "lookahead", seed)`) then works unchanged.

## What a pack file does not carry

Fire compartments (hospital procedures), hand-made scenarios and the 3D dashboard model (`index.html` uses its own
`*_data.js`). The Research Lab and every `evac` study work from the JSON alone.
