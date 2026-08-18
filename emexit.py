import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import math

# --- 1. THE NODES (Coordinates) ---
nodes_data = {
    'sem hall 1': (1039, 944), 'computer lab': (1586, 919), 'stairs 1': (1179, 1174), 'classroom g24': (1230, 1354), 'classroom g25': (1222, 1585), 'lift': (2364, 1959), 'staff room 1': (1320, 1823), 'staff room 2': (1320, 1895), 'staff room 3': (1323, 1977), 'staff room 4': (1327, 2049), 'staff room 5': (1248, 2136), 'staff room 6': (1190, 2139), 'staff room hod': (873, 2179), 'staff room 7': (877, 2294), 'staff room 8': (880, 2377), 'staff room 9': (891, 2460), 'conference room 1': (920, 2600), 'staff room 10': (1050, 2618), 'staff room 11': (1132, 2626), 'staff room 12': (1215, 2615), 'staff room 13': (1341, 2626), 'staff room 14': (1370, 2323), 'staff room 15': (1370, 2255), 'staff room 16': (1449, 2183), 'staff room 17': (1536, 2168), 'staff room 18': (1608, 2175), 'staff room 19': (1687, 2175), 'stairs 2': (1338, 2446), 'male wr': (1158, 1718), 'female wr': (1791, 2320), 'classroom g18': (1932, 2262), 'classroom g17': (2151, 2266), 'stairs 3': (2331, 2302), 'sem hall 2': (2566, 2453), 'chemistry lab': (2602, 1898), 'east entrance/exit': (2666, 2129), 'north entrance/exit': (1348, 861), 'stairs 4': (1993, 1038), 'office': (2425, 1347), 'bank': (2159, 1088), 'cabin 1': (2753, 1427), 'cabin 2': (2756, 1297), 'cabin 3': (2674, 1106), 'cabin 4': (2407, 847), 'cabin 5': (2220, 760), 'cabin 6': (2069, 764), 'east main entrance': (3016, 825), 'north main entrance': (2684, 547), 'atm': (1929, 1013), 'lift 2': (2295, 1956), 'lift 3': (1523, 1134), 'lift 4': (1527, 1211), 'staff fem wr': (1470, 2372), 'staff mens wr': (1134, 2074), 'electrical room 1': (2335, 1869), 'electrical room 2': (1617, 1154), 'j1': (2711, 812), 'j2': (2453, 1080), 'j3': (2205, 1305), 'j4': (2429, 1510), 'j5': (2429, 1986), 'j6': (2443, 2178), 'j7': (2265, 2094), 'j8': (1782, 2094), 'j9': (1617, 2097), 'j10': (1409, 2090), 'j11': (1255, 2238), 'j12': (1309, 2537), 'j13': (1017, 2503), 'j14': (973, 2191), 'j15': (1396, 1842), 'j16': (1396, 1711), 'j17': (1389, 1211), 'j18': (1312, 1067), 'j19': (1513, 1067), 'j20': (1986, 1077)
}

# --- 2. THE EDGES (Connections) ---
connections = [
    ('north main entrance', 'j1'), ('east main entrance', 'j1'), ('j1', 'j2'),
    ('j2', 'cabin 3'), ('j2', 'cabin 2'), ('j2', 'cabin 1'), ('cabin 4', 'j2'),
    ('cabin 5', 'j2'), ('cabin 6', 'j2'), ('bank', 'j2'), ('office', 'j2'),
    ('j2', 'j3'), ('office', 'j3'), ('bank', 'j3'), ('j20', 'j3'), ('j3', 'j4'),
    ('j4', 'j5'), ('j5', 'j6'), ('j5', 'j7'), ('j6', 'j7'), ('j7', 'j8'),
    ('j8', 'j9'), ('j9', 'j10'), ('j10', 'j11'), ('j11', 'j12'), ('j12', 'j13'),
    ('j13', 'j14'), ('j14', 'j11'), ('j10', 'j15'), ('j15', 'j16'), ('j16', 'j17'),
    ('j17', 'j18'), ('j18', 'j19'), ('j19', 'j17'), ('j18', 'north entrance/exit'),
    ('sem hall 1', 'j18'), ('stairs 1', 'j18'), ('stairs 1', 'j17'), ('classroom g25', 'j16'),
    ('classroom g24', 'j17'), ('lift 4', 'j17'), ('lift 3', 'j17'), ('computer lab', 'j18'),
    ('j19', 'j20'), ('atm', 'j20'), ('j20', 'stairs 4'), ('lift', 'j5'), ('lift 2', 'j5'),
    ('lift', 'j7'), ('lift 2', 'j7'), ('lift 3', 'j19'), ('lift 4', 'j19'),
    ('sem hall 2', 'j6'), ('j6', 'east entrance/exit'), ('chemistry lab', 'j6'),
    ('stairs 3', 'j6'), ('stairs 3', 'j7'), ('classroom g17', 'j7'), ('classroom g18', 'j8'),
    ('female wr', 'j8'), ('male wr', 'j16'), ('staff room 6', 'j11'), ('staff room 5', 'j11'),
    ('staff mens wr', 'j11'), ('staff mens wr', 'j14'), ('staff room hod', 'j14'),
    ('staff room 7', 'j14'), ('staff room 8', 'j14'), ('staff room 9', 'j13'),
    ('staff room 8', 'j13'), ('staff room 7', 'j13'), ('staff room 9', 'j14'),
    ('conference room 1', 'j13'), ('staff room 10', 'j13'), ('staff room 11', 'j13'),
    ('staff room 12', 'j13'), ('staff room 13', 'j13'), ('staff room 10', 'j12'),
    ('staff room 11', 'j12'), ('staff room 12', 'j12'), ('staff room 13', 'j12'),
    ('j12', 'stairs 2'), ('stairs 2', 'j11'), ('staff room 14', 'j11'), ('staff room 15', 'j11'),
    ('staff room 15', 'j12'), ('staff room 14', 'j12'), ('j11', 'staff fem wr'),
    ('j12', 'staff fem wr'), ('staff room 6', 'j14'), ('staff room 5', 'j14'),
    ('staff room 1', 'j15'), ('staff room 2', 'j15'), ('staff room 3', 'j15'),
    ('staff room 4', 'j15'), ('staff room 4', 'j10'), ('staff room 3', 'j10'),
    ('staff room 16', 'j9'), ('staff room 17', 'j9'), ('staff room 16', 'j10'),
    ('staff room 17', 'j10'), ('staff room 18', 'j9'), ('staff room 19', 'j9'),
    ('staff room 19', 'j8'), ('staff room 18', 'j8'), ('electrical room 1', 'j5')
]

# --- 3. BUILD THE GRAPH ---
G = nx.Graph()
for u, v in connections:
    if u in nodes_data and v in nodes_data:
        p1, p2 = nodes_data[u], nodes_data[v]
        dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        G.add_edge(u, v, weight=dist)

# --- NEW ADDITION: EMERGENCY EXIT FUNCTION ---
def find_all_exits(start_node):
    all_exits = [n for n in nodes_data.keys() if 'exit' in n.lower() or 'entrance' in n.lower()]
    routes = []
    
    for exit_node in all_exits:
        try:
            path = nx.shortest_path(G, source=start_node, target=exit_node, weight='weight')
            length = nx.shortest_path_length(G, source=start_node, target=exit_node, weight='weight')
            routes.append({'path': path, 'dist': length, 'name': exit_node})
        except nx.NetworkXNoPath:
            continue

    if not routes:
        print("❌ No reachable exits found!")
        return

    routes.sort(key=lambda x: x['dist'])

    img = Image.open('floor_plan_hd.jpg')
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img)

    for i, route in enumerate(routes):
        p_nodes = route['path']
        px = [nodes_data[n][0] for n in p_nodes]
        py = [nodes_data[n][1] for n in p_nodes]
        
        color = 'lime' if i == 0 else 'orange'
        linewidth = 5 if i == 0 else 2
        label = f"FASTEST: {route['name']}" if i == 0 else f"Alt: {route['name']}"
        
        ax.plot(px, py, color=color, linewidth=linewidth, alpha=0.7, label=label)
        ax.scatter(px[-1], py[-1], color=color, s=100, edgecolors='black')

    start_pos = nodes_data[start_node]
    ax.scatter(start_pos[0], start_pos[1], color='blue', s=150, marker='X', label='YOU ARE HERE', zorder=5)

    plt.title(f"Emergency Routes from {start_node}")
    plt.legend()
    plt.axis('off')
    plt.show()

    print(f"\n✅ Closest Exit is: {routes[0]['name']}")

# --- 4. MAIN MENU & USER INPUT ---
print("\n--- CAMPUS NAVIGATION SYSTEM ---")
print("1. Standard Navigation (Room to Room)")
print("2. Emergency Routing (Find closest exits)")

mode = input("\nSelect Mode (1 or 2): ").strip()
print("\nAvailable Rooms:", ", ".join([n for n in nodes_data.keys() if not n.startswith('j')]))

if mode == '1':
    # YOUR ORIGINAL CODE FOR STANDARD NAVIGATION
    start_node = input("\nEnter your current location: ").strip()
    end_node = input("Enter your destination: ").strip()

    try:
        path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
        
        img = Image.open('floor_plan_hd.jpg')
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(img)

        path_x = [nodes_data[node][0] for node in path]
        path_y = [nodes_data[node][1] for node in path]

        ax.plot(path_x, path_y, color='lime', linewidth=5, alpha=0.6, label='Recommended Path')
        ax.scatter(path_x, path_y, color='red', s=20) 
        
        ax.text(path_x[0], path_y[0], ' START', color='blue', fontweight='bold', backgroundcolor='white')
        ax.text(path_x[-1], path_y[-1], ' DESTINATION', color='red', fontweight='bold', backgroundcolor='white')

        plt.title(f"Route: {start_node} to {end_node}")
        plt.axis('off')
        plt.legend()
        plt.show()

        print(f"\n✅ Directions: {' -> '.join(path)}")

    except nx.NetworkXNoPath:
        print(f"❌ Error: No path found between {start_node} and {end_node}. Check your edges!")
    except KeyError as e:
        print(f"❌ Error: Room '{e}' is not recognized. Check your spelling.")

elif mode == '2':
    # NEW EMERGENCY EXIT CODE
    start_node = input("\nEnter your current location to find exits: ").strip()
    try:
        if start_node in nodes_data:
            find_all_exits(start_node)
        else:
            print(f"❌ Error: Room '{start_node}' is not recognized.")
    except Exception as e:
        print(f"Error occurred: {e}")

else:
    print("❌ Invalid selection. Please run the code again and type 1 or 2.")