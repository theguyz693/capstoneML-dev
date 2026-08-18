import matplotlib.pyplot as plt
from PIL import Image
import math

# ==========================================
# 1. PASTE YOUR SIXTH FLOOR NODES HERE
# ==========================================
# Replace this placeholder with the dictionary you generated in the last step!
from sixth_floor_data import nodes_f6

# This list will store your new connections
edges_f6 = []
temp_node = None 

# ==========================================
# 2. LOAD IMAGE AND MAXIMIZE
# ==========================================
img_path = 'sixth_floor_hd.jpg'
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
# 3. DRAW EXISTING NODES
# ==========================================
# This draws all the red dots so you know where to click
for name, (nx, ny) in nodes_f6.items():
    ax.plot(nx, ny, 'ro', markersize=4)
    ax.text(nx, ny, f" {name}", color='red', fontsize=4, alpha=0.7)

# ==========================================
# 4. THE CLICKING LOGIC
# ==========================================
def find_nearest_node(x, y):
    min_dist = 100 # Snap radius
    closest_name = None
    for name, (nx, ny) in nodes_f6.items():
        dist = math.sqrt((x-nx)**2 + (y-ny)**2)
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def onclick(event):
    global temp_node
    if event.xdata is not None and event.ydata is not None:
        clicked_node = find_nearest_node(event.xdata, event.ydata)
        
        if clicked_node:
            if temp_node is None:
                # First click: Highlight yellow (Start of hallway)
                temp_node = clicked_node
                print(f"Start: {temp_node} -> Click destination node...")
                ax.plot(nodes_f6[temp_node][0], nodes_f6[temp_node][1], 'yo', markersize=10) 
                plt.draw()
            else:
                # Second click: Connect (End of hallway)
                if temp_node != clicked_node:
                    edges_f6.append((temp_node, clicked_node))
                    
                    x_vals = [nodes_f6[temp_node][0], nodes_f6[clicked_node][0]]
                    y_vals = [nodes_f6[temp_node][1], nodes_f6[clicked_node][1]]
                    
                    ax.plot(x_vals, y_vals, 'b-', linewidth=2)
                    print(f"Connected: {temp_node} <--> {clicked_node}")
                
                # Reset for the next connection
                temp_node = None
                plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.title("F6 EDGE CONNECTOR: Click a start node, then an end node to connect them.", fontsize=12)
plt.axis('off')
plt.show()

# ==========================================
# 5. FINAL OUTPUT
# ==========================================
print("\n" + "="*40)
print("COPY THESE SIXTH FLOOR EDGES:")
print("="*40)
print("edges_f6 = [")
for edge in edges_f6:
    print(f"    {edge},")
print("]")