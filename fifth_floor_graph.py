import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# 1. PASTE YOUR FIFTH FLOOR DATA HERE
# ==========================================
# (Copy your F4 data, do a Find/Replace changing 'f4_' to 'f5_', and paste here)
from fifth_floor_data import nodes_f5, edges_f5

# ==========================================
# 2. LOAD IMAGE AND MAXIMIZE
# ==========================================
img_path = 'fourth_floor_hd.jpg' # Make sure this image name is correct!
try:
    img = Image.open(img_path) 
except FileNotFoundError:
    print(f"❌ Error: {img_path} not found. Make sure the image is in the same folder!")
    exit()

fig, ax = plt.subplots(figsize=(20, 12))

mng = plt.get_current_fig_manager()
try:
    mng.window.state('zoomed') 
except:
    pass

ax.imshow(img)

# ==========================================
# 3. DRAW THE FULL GRAPH
# ==========================================
# Draw the Nodes (Red dots + Text)
for name, (nx, ny) in nodes_f5.items():
    ax.plot(nx, ny, 'ro', markersize=4, zorder=5) 
    ax.text(nx, ny, f" {name}", color='red', fontsize=5, alpha=0.9, zorder=6)

# Draw the Edges (Blue lines)
for start_node, end_node in edges_f5:
    if start_node in nodes_f5 and end_node in nodes_f5:
        x_vals = [nodes_f5[start_node][0], nodes_f5[end_node][0]]
        y_vals = [nodes_f5[start_node][1], nodes_f5[end_node][1]]
        
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, alpha=0.6, zorder=4)
    else:
        print(f"⚠️ Warning: Edge {start_node} <-> {end_node} contains a node that doesn't exist!")

# ==========================================
# 4. DISPLAY
# ==========================================
plt.title("FIFTH FLOOR - FINAL GRAPH VERIFICATION", fontsize=16, fontweight='bold', pad=15)
plt.axis('off') 
plt.show()