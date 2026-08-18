import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import numpy as np

# ==========================================
# 1. THE MAGIC IMPORT (No more giant lists!)
# ==========================================
from campus_data import nodes_g_raw, nodes_f1_raw, edges_g, edges_f1, bridges
print("Starting 3D Map script...")
# ==========================================
# 2. SETTINGS
# ==========================================
FLOOR_HEIGHT = 1500  
DOWNSCALE_FACTOR = 2 # Change to 2 for higher quality, 6 for faster performance

GROUND_IMG_PATH = 'floor_plan_hd.jpg' 
FIRST_IMG_PATH = 'first_floor_hd.jpg'

# ==========================================
# 3. ALIGNMENT & COMBINING NODES/EDGES
# ==========================================
all_nodes_3d = {}

# Ground Floor (Z = 0)
for name, (x, y) in nodes_g_raw.items():
    all_nodes_3d[name] = (x, y, 0)

# Calculate the shift using 'stairs 1' so the floors stack perfectly
if 'stairs 1' in nodes_g_raw and 'f1_stairs 1' in nodes_f1_raw:
    shift_x = nodes_f1_raw['f1_stairs 1'][0] - nodes_g_raw['stairs 1'][0]
    shift_y = nodes_f1_raw['f1_stairs 1'][1] - nodes_g_raw['stairs 1'][1]
else:
    shift_x, shift_y = 0, 0 

# First Floor (Apply shift and set Z = FLOOR_HEIGHT)
for name, (x, y) in nodes_f1_raw.items():
    all_nodes_3d[name] = (x - shift_x, y - shift_y, FLOOR_HEIGHT)

# Combine all edges
all_edges = edges_g + edges_f1 + bridges


# ==========================================
# 4. IMAGE RENDERER FUNCTION
# ==========================================
def add_floor_plan_3d(ax, image_path, z_height, offset_x=0, offset_y=0):
    try:
        img = Image.open(image_path)
        w, h = img.size
        new_w, new_h = w // DOWNSCALE_FACTOR, h // DOWNSCALE_FACTOR
        img = img.resize((new_w, new_h))
        
        img_data = np.array(img) / 255.0
        
        x_grid = np.linspace(0 - offset_x, w - offset_x, new_w)
        y_grid = np.linspace(h - offset_y, 0 - offset_y, new_h) 
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.full_like(X, z_height)
        
        ax.plot_surface(X, Y, Z, facecolors=img_data, rstride=1, cstride=1, shade=False, alpha=0.6)
    except Exception as e:
        print(f"Could not load image {image_path}. Error: {e}")

# ==========================================
# 5. DRAW THE 3D SCENE
# ==========================================
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Draw Blueprints slightly below the nodes to prevent overlap
add_floor_plan_3d(ax, GROUND_IMG_PATH, z_height=-20)
add_floor_plan_3d(ax, FIRST_IMG_PATH, z_height=FLOOR_HEIGHT - 20, offset_x=shift_x, offset_y=shift_y)

# Draw Nodes
for name, (x, y, z) in all_nodes_3d.items():
    color = '#ff4d4d' if z == 0 else '#4d94ff'
    ax.scatter(x, y, z, c=color, s=20, depthshade=True, edgecolors='black', linewidths=0.5)
    
    # Hide messy junction text, only show room names
    if not name.startswith('j') and not name.startswith('f1_j'):
        ax.text(x, y, z + 40, f"{name}", size=6, color='black', alpha=0.8)

# Draw Edges
for start, end in all_edges:
    if start in all_nodes_3d and end in all_nodes_3d:
        x_coords = [all_nodes_3d[start][0], all_nodes_3d[end][0]]
        y_coords = [all_nodes_3d[start][1], all_nodes_3d[end][1]]
        z_coords = [all_nodes_3d[start][2], all_nodes_3d[end][2]]
        
        if all_nodes_3d[start][2] != all_nodes_3d[end][2]:
            # Bridges (Thick Green Dashed)
            ax.plot(x_coords, y_coords, z_coords, color='#2ca02c', linestyle='--', linewidth=3)
        else:
            # Hallways (Red for Ground, Blue for F1)
            line_color = '#ff4d4d' if all_nodes_3d[start][2] == 0 else '#4d94ff'
            ax.plot(x_coords, y_coords, z_coords, color=line_color, linewidth=1.5, alpha=0.7)

# Final Polish
ax.invert_yaxis() # Match image coordinates
ax.set_axis_off()
fig.patch.set_facecolor('#f4f4f4') 
ax.set_title("Interactive 3D Campus Map", fontsize=18, fontweight='bold', pad=20)
ax.view_init(elev=35, azim=-60) # Best default viewing angle

plt.tight_layout()
plt.show()
print("Map closed!")
