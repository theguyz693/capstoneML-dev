import matplotlib.pyplot as plt
from PIL import Image
import math


# Pull your ground floor data from the central file
from ground_floor_data import nodes_g_raw, edges_g

# Rename them locally so the rest of the script works without changes
nodes_data = nodes_g_raw
edges = edges_g
temp_node = None # Stores the first click of a pair


# 3. Load Image
img = Image.open('floor_plan_hd.jpg')
fig, ax = plt.subplots(figsize=(16, 12))
ax.imshow(img)

# Draw all nodes so you know where to click
for name, (nx, ny) in nodes_data.items():
    ax.plot(nx, ny, 'ro', markersize=4)
    ax.text(nx, ny, f" {name}", color='red', fontsize=5)

# --- NEW: DRAW EXISTING DATA SO IT REFLECTS ---
# Draw Dots
for name, (nx, ny) in nodes_data.items():
    ax.plot(nx, ny, 'ro', markersize=4)
    ax.text(nx, ny, f" {name}", color='red', fontsize=5)

# Draw Lines from the 'edges' list
for start, end in edges:
    if start in nodes_data and end in nodes_data:
        x_vals = [nodes_data[start][0], nodes_data[end][0]]
        y_vals = [nodes_data[start][1], nodes_data[end][1]]
        # Use alpha=0.5 so we can still see the blueprint underneath
        ax.plot(x_vals, y_vals, 'b-', linewidth=1, alpha=0.5)
        

def find_nearest_node(x, y):
    min_dist = 50 # Ignore clicks too far from a node
    closest_name = None
    for name, (nx, ny) in nodes_data.items():
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
                # First click of the pair
                temp_node = clicked_node
                print(f"Start: {temp_node} ... click next node to connect.")
                ax.plot(nodes_data[temp_node][0], nodes_data[temp_node][1], 'yo', markersize=6) # Highlight yellow
                plt.draw()
            else:
                # Second click of the pair
                if temp_node != clicked_node:
                    edges.append((temp_node, clicked_node))
                    # Draw the line
                    x_values = [nodes_data[temp_node][0], nodes_data[clicked_node][0]]
                    y_values = [nodes_data[temp_node][1], nodes_data[clicked_node][1]]
                    ax.plot(x_values, y_values, 'b-', linewidth=1)
                    print(f"Connected: {temp_node} <--> {clicked_node}")
                    
                # Reset for next pair
                temp_node = None
                plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.title("Click two nodes to connect them with an edge")
plt.show()

# 4. Print results
print("\n--- COPY THESE EDGES ---")
print("connections = [")
for edge in edges:
    print(f"    {edge},")
print("]")