import networkx as nx
import matplotlib.pyplot as plt

# Create a test graph
G = nx.Graph()
G.add_edge("Start", "End", weight=1.0)

# Check if NetworkX can find a path
path = nx.shortest_path(G, "Start", "End")
print(f"Installation Successful! Path found: {path}")