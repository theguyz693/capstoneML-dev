import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# 1. PASTE YOUR FOURTH FLOOR DATA HERE
# ==========================================
# Paste the nodes dictionary you generated from fourth_floor_nodes.py
from fourth_floor_data import nodes_f4, edges_f4 

# ==========================================
# 2. LOAD IMAGE AND MAXIMIZE
# ==========================================
img_path = 'fourth_floor_hd.jpg' # Make sure this matches your filename!
try:
    img = Image.open(img_path) 
except FileNotFoundError:
    print(f"❌ Error: {img_path} not found. Make sure the image is in the same folder!")
    exit()

fig, ax = plt.subplots(figsize=(20, 12))

mng = plt.get_current_fig_manager()
try:
    mng.window.state('zoomed') # Full screen for Windows
except:
    pass

ax.imshow(img)

# ==========================================
# 3. DRAW THE FULL GRAPH
# ==========================================
# Draw the Nodes (Red dots + Text)
for name, (nx, ny) in nodes_f4.items():
    ax.plot(nx, ny, 'ro', markersize=4, zorder=5) # zorder ensures dots draw on top of lines
    ax.text(nx, ny, f" {name}", color='red', fontsize=5, alpha=0.9, zorder=6)

# Draw the Edges (Blue lines)
for start_node, end_node in edges_f4:
    if start_node in nodes_f4 and end_node in nodes_f4:
        x_vals = [nodes_f4[start_node][0], nodes_f4[end_node][0]]
        y_vals = [nodes_f4[start_node][1], nodes_f4[end_node][1]]
        
        # Plot a thick, slightly transparent blue line
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, alpha=0.6, zorder=4)
    else:
        print(f"⚠️ Warning: Edge {start_node} <-> {end_node} contains a node that doesn't exist!")

# ==========================================
# 4. DISPLAY
# ==========================================
plt.title("FOURTH FLOOR - FINAL GRAPH VERIFICATION", fontsize=16, fontweight='bold', pad=15)
plt.axis('off') # Hide the axis numbers for a clean look
plt.show()