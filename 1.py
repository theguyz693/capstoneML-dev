import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import math

# 1. Initialize the Graph
G = nx.Graph()

# 2. Define Nodes and their Pixel Coordinates
# Use your coordinate picker to find these exact (x, y) values
nodes_data = {
    # Key Rooms (identified by colors in ground floor.jpg)
    "Seminar_Hall_002": (166, 172),    # Top Left (Gray)
    "Class_Room_004": (200, 250),      # Left (Green)
    "Class_Room_003": (198, 302),      # Left (Green)
    "Computer_Center": (293, 163),     # Top (Light Gray)
    "Principal_Room": (530, 287),
    "chemistry_Room": (484, 366),      # Right (Blue)
    
    # Hallway Decision Points (Crucial for Granular Navigation)
    # These are points in the ORANGE area where paths intersect
    "Junction_North_Exit": (260, 310),
    "Junction_West_Hall": (260, 450),
    "Junction_Quad_Entry": (350, 450),
    "Junction_East_Wing": (500, 450)
}

# Add nodes to the graph
for name, pos in nodes_data.items():
    G.add_node(name, pos=pos)

# 3. Define Edges (The walkable paths)
# RULE: Connect rooms to junctions, then junctions to each other.
connections = [
    # Rooms to Junctions
    ("Seminar_Hall_002", "Junction_North_Exit"),
    ("Computer_Center", "Junction_North_Exit"),
    ("Class_Room_004", "Junction_West_Hall"),
    ("Class_Room_003", "Junction_West_Hall"),
    ("Principal_Room", "Junction_East_Wing"),
    
    # Hallway Network (The spine of your navigation)
    ("Junction_North_Exit", "Junction_West_Hall"),
    ("Junction_West_Hall", "Junction_Quad_Entry"),
    ("Junction_Quad_Entry", "Junction_East_Wing")
]

# 4. Add Edges with Automatic Weights (Pixel Distance)
for u, v in connections:
    p1 = nodes_data[u]
    p2 = nodes_data[v]
    # Calculate Euclidean distance as the weight
    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    G.add_edge(u, v, weight=round(dist, 2))

# 5. Overlay on ground floor.jpg
img = Image.open('ground floor.jpg')
fig, ax = plt.subplots(figsize=(15, 10))
ax.imshow(img)

# Get positions for plotting
pos_map = nx.get_node_attributes(G, 'pos')

# Draw the nodes and edges
nx.draw_networkx_nodes(G, pos_map, ax=ax, node_size=100, node_color='red')
nx.draw_networkx_edges(G, pos_map, ax=ax, edge_color='blue', width=2)
nx.draw_networkx_labels(G, pos_map, ax=ax, font_size=8, font_color='black')

plt.title("Granular Navigation Graph: Ground Floor")
plt.axis('off')
plt.show()