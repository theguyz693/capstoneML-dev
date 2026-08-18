import json

# 1. Import all 7 levels of data from your master file
from campus_data import (
    nodes_g_raw, edges_g,
    nodes_f1_raw, edges_f1,
    nodes_f2, edges_f2,
    nodes_f3, edges_f3,
    nodes_f4, edges_f4,
    nodes_f5, edges_f5,
    nodes_f6, edges_f6,
    bridges
)

# --- SETTINGS ---
SCALE_RATIO = 1.61  # Shrinks upper floors to match ground floor
FLOOR_HEIGHT = 350  # The vertical gap between each floor

js_data = {
    "nodes": {},
    "edges": []
}

# 2. Process Ground Floor (Z = 0)
for name, (x, y) in nodes_g_raw.items():
    js_data["nodes"][name] = {"x": x, "y": y, "z": 0}

# Define the anchor on the ground floor to align everything
ground_anchor_name = 'stairs 1'
if ground_anchor_name not in nodes_g_raw:
    print(f"❌ CRITICAL ERROR: Could not find '{ground_anchor_name}' on the ground floor to align the building!")
    exit()

ground_anchor = nodes_g_raw[ground_anchor_name]

# 3. Create a configuration for the upper floors
# Now every floor has its own specific 'scale' and 'anchor'
upper_floors = [
    {"nodes": nodes_f1_raw, "edges": edges_f1, "anchor": "f1_stairs 1", "level": 1, "scale": 1.5},
    {"nodes": nodes_f2, "edges": edges_f2, "anchor": "f2_stairs 1", "level": 2, "scale": 1.8},
    {"nodes": nodes_f3, "edges": edges_f3, "anchor": "f3_stairs 1", "level": 3, "scale": 1.8},
    
    # 🚨 FIXING FLOORS 4, 5, AND 6 🚨
    # Change the scale to 1.0 as a baseline (tweak it if it's still too small or big)
    # Ensure "f4_stairs 1" is the EXACT same physical staircase as Ground Floor "stairs 1"
    {"nodes": nodes_f4, "edges": edges_f4, "anchor": "f4_stairs 1", "level": 4, "scale": 0.4},
    {"nodes": nodes_f5, "edges": edges_f5, "anchor": "f5_stairs 1", "level": 5, "scale": 0.4},
    {"nodes": nodes_f6, "edges": edges_f6, "anchor": "f6_stairs 1", "level": 6, "scale": 0.4},
]

# 4. Loop through and stack each upper floor
for floor in upper_floors:
    if floor["anchor"] in floor["nodes"]:
        floor_anchor = floor["nodes"][floor["anchor"]]
        current_z = FLOOR_HEIGHT * floor["level"]
        
        for name, (x, y) in floor["nodes"].items():
            dist_x = x - floor_anchor[0]
            dist_y = y - floor_anchor[1]
            
            # Divide by this specific floor's scale
            scaled_x = dist_x / floor["scale"]
            scaled_y = dist_y / floor["scale"]
            
            final_x = ground_anchor[0] + scaled_x
            final_y = ground_anchor[1] + scaled_y
            
            js_data["nodes"][name] = {"x": final_x, "y": final_y, "z": current_z}
    else:
        print(f"⚠️ Warning: Anchor '{floor['anchor']}' not found. Floor {floor['level']} alignment may fail.")


# 5. Combine ALL edges
all_edges = edges_g + bridges
for floor in upper_floors:
    all_edges.extend(floor["edges"])

js_data["edges"] = all_edges

# 6. Save as a JavaScript file for the Three.js viewer
try:
    with open('campus_data.js', 'w') as f:
        f.write("const campusData = " + json.dumps(js_data, indent=4) + ";\n")
    print("✅ Successfully exported your entire 7-story map to campus_data.js!")
    print(f"Total Nodes: {len(js_data['nodes'])}")
    print(f"Total Edges: {len(js_data['edges'])}")
except Exception as e:
    print(f"❌ Error saving file: {e}")