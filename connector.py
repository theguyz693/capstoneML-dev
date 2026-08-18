import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. SETTINGS
# ==========================================
FLOOR_HEIGHT = 500  # How high the first floor sits above the ground floor

# ==========================================
# 2. RAW NODE DATA (X, Y)
# ==========================================
# --- PASTE ALL GROUND FLOOR NODES HERE ---
nodes_g_raw = {'sem hall 1': (1039, 944), 'computer lab': (1586, 919), 'stairs 1': (1179, 1174), 'classroom g24': (1230, 1354), 'classroom g25': (1222, 1585), 'lift': (2364, 1959), 'staff room 1': (1320, 1823), 'staff room 2': (1320, 1895), 'staff room 3': (1323, 1977), 'staff room 4': (1327, 2049), 'staff room 5': (1248, 2136), 'staff room 6': (1190, 2139), 'staff room hod': (873, 2179), 'staff room 7': (877, 2294), 'staff room 8': (880, 2377), 'staff room 9': (891, 2460), 'conference room 1': (920, 2600), 'staff room 10': (1050, 2618), 'staff room 11': (1132, 2626), 'staff room 12': (1215, 2615), 'staff room 13': (1341, 2626), 'staff room 14': (1370, 2323), 'staff room 15': (1370, 2255), 'staff room 16': (1449, 2183), 'staff room 17': (1536, 2168), 'staff room 18': (1608, 2175), 'staff room 19': (1687, 2175), 'stairs 2': (1338, 2446), 'male wr': (1158, 1718), 'female wr': (1791, 2320), 'classroom g18': (1932, 2262), 'classroom g17': (2151, 2266), 'stairs 3': (2331, 2302), 'sem hall 2': (2566, 2453), 'chemistry lab': (2602, 1898), 'east entrance/exit': (2666, 2129), 'north entrance/exit': (1348, 861), 'stairs 4': (1993, 1038), 'office': (2425, 1347), 'bank': (2159, 1088), 'cabin 1': (2753, 1427), 'cabin 2': (2756, 1297), 'cabin 3': (2674, 1106), 'cabin 4': (2407, 847), 'cabin 5': (2220, 760), 'cabin 6': (2069, 764), 'east main entrance': (3016, 825), 'north main entrance': (2684, 547), 'atm': (1929, 1013), 'lift 2': (2295, 1956), 'lift 3': (1523, 1134), 'lift 4': (1527, 1211), 'staff fem wr': (1470, 2372), 'staff mens wr': (1134, 2074), 'electrical room 1': (2335, 1869), 'electrical room 2': (1617, 1154), 'j1': (2711, 812), 'j2': (2453, 1080), 'j3': (2205, 1305), 'j4': (2429, 1510), 'j5': (2429, 1986), 'j6': (2443, 2178), 'j7': (2265, 2094), 'j8': (1782, 2094), 'j9': (1617, 2097), 'j10': (1409, 2090), 'j11': (1255, 2238), 'j12': (1309, 2537), 'j13': (1017, 2503), 'j14': (973, 2191), 'j15': (1396, 1842), 'j16': (1396, 1711), 'j17': (1389, 1211), 'j18': (1312, 1067), 'j19': (1513, 1067), 'j20': (1986, 1077)}

# --- PASTE ALL FIRST FLOOR NODES HERE ---
nodes_f1_raw = {
    'f1_sem hall 1': (2119, 1244),
    'f1_stairs 1': (2376, 1689),
    'f1_stairs 3': (4535, 3882),
    'f1_stairs 2': (2632, 4071),
    'f1_stairs 4': (3928, 1460),
    'f1_classroom 124': (2490, 2027),
    'f1_classroom 125': (2463, 2465),
    'f1_lab': (3327, 1203),
    'f1_staff room 1': (2632, 2890),
    'f1_staff room 2': (2645, 3059),
    'f1_staff room 3': (2639, 3187),
    'f1_staff room 4': (2639, 3349),
    'f1_staff room 5': (2490, 3511),
    'f1_staff room 6': (2382, 3505),
    'f1_staff room hod': (1802, 3565),
    'f1_staff room 7': (1795, 3781),
    'f1_staff room 8': (1782, 3923),
    'f1_staff room 9': (1782, 4092),
    'f1_conference room 1': (1829, 4348),
    'f1_staff room 10': (2092, 4395),
    'f1_staff room 11': (2254, 4409),
    'f1_staff room 12': (2423, 4416),
    'f1_staff room 13': (2652, 4409),
    'f1_staff room 14': (2720, 3829),
    'f1_staff room 15': (2720, 3707),
    'f1_staff room 16': (2868, 3572),
    'f1_staff room 17': (3017, 3586),
    'f1_staff room 18': (3165, 3592),
    'f1_staff room 19': (3327, 3599),
    'f1_male wr': (2328, 2708),
    'f1_female wr': (3496, 3849),
    'f1_staff female wr': (2807, 3950),
    'f1_staff mens wr': (2268, 3430),
    'f1_classroom 117': (3766, 3748),
    'f1_classroom 118': (4238, 3768),
    'f1_sem hall 2': (4981, 4139),
    'f1_lift 1': (4609, 3194),
    'f1_lift 2': (4501, 3194),
    'f1_lift 3': (3044, 1635),
    'f1_lift 4': (3044, 1763),
    'f1_physics lab': (5041, 3154),
    'f1_library': (4549, 1730),
    'f1_storeroom': (4981, 2357),
    'f1_j1': (4198, 1858),
    'f1_j2': (4407, 2067),
    'f1_j3': (4738, 2391),
    'f1_j4': (4724, 3295),
    'f1_j5': (4751, 3532),
    'f1_j6': (4522, 3457),
    'f1_j7': (3509, 3451),
    'f1_j8': (3273, 3437),
    'f1_j9': (2834, 3410),
    'f1_j10': (2490, 3667),
    'f1_j11': (2578, 4240),
    'f1_j12': (2045, 4166),
    'f1_j13': (1957, 3613),
    'f1_j14': (2780, 3032),
    'f1_j15': (2767, 2715),
    'f1_j16': (2767, 1730),
    'f1_j17': (2706, 1480),
    'f1_j18': (2976, 1534),
    'f1_j19': (3860, 1520),
}

# ==========================================
# 3. ALIGNMENT & COMBINING NODES IN 3D
# ==========================================
all_nodes_3d = {}

# Add Ground Floor nodes (Z = 0)
for name, (x, y) in nodes_g_raw.items():
    all_nodes_3d[name] = (x, y, 0)

# Calculate the shift using 'stairs 1' as the anchor point
if 'stairs 1' in nodes_g_raw and 'f1_stairs 1' in nodes_f1_raw:
    shift_x = nodes_f1_raw['f1_stairs 1'][0] - nodes_g_raw['stairs 1'][0]
    shift_y = nodes_f1_raw['f1_stairs 1'][1] - nodes_g_raw['stairs 1'][1]
else:
    shift_x, shift_y = 0, 0 # Fallback if anchor is missing

# Add First Floor nodes, apply the shift, and set Z = FLOOR_HEIGHT
for name, (x, y) in nodes_f1_raw.items():
    all_nodes_3d[name] = (x - shift_x, y - shift_y, FLOOR_HEIGHT)


# ==========================================
# 4. EDGE DATA (CONNECTIONS)
# ==========================================
# --- PASTE ALL GROUND FLOOR EDGES HERE ---
edges_g = [
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

# --- PASTE ALL FIRST FLOOR EDGES HERE ---
edges_f1 = [
    ('f1_library', 'f1_j1'),
    ('f1_library', 'f1_j2'),
    ('f1_j2', 'f1_j3'),
    ('f1_j1', 'f1_j2'),
    ('f1_j19', 'f1_stairs 4'),
    ('f1_j19', 'f1_j1'),
    ('f1_j18', 'f1_j19'),
    ('f1_j17', 'f1_j18'),
    ('f1_j16', 'f1_j17'),
    ('f1_j18', 'f1_j16'),
    ('f1_j16', 'f1_j15'),
    ('f1_j15', 'f1_j14'),
    ('f1_j14', 'f1_j9'),
    ('f1_j9', 'f1_j10'),
    ('f1_j10', 'f1_j13'),
    ('f1_j13', 'f1_j12'),
    ('f1_j12', 'f1_j11'),
    ('f1_j11', 'f1_j10'),
    ('f1_j9', 'f1_j8'),
    ('f1_j8', 'f1_j7'),
    ('f1_j7', 'f1_j6'),
    ('f1_j6', 'f1_j4'),
    ('f1_j4', 'f1_j5'),
    ('f1_j5', 'f1_j6'),
    ('f1_j4', 'f1_j3'),
    ('f1_storeroom', 'f1_j3'),
    ('f1_physics lab', 'f1_j4'),
    ('f1_lift 2', 'f1_j6'),
    ('f1_lift 1', 'f1_j6'),
    ('f1_lift 1', 'f1_j4'),
    ('f1_lift 2', 'f1_j4'),
    ('f1_lift 1', 'f1_j5'),
    ('f1_lift 2', 'f1_j5'),
    ('f1_physics lab', 'f1_j5'),
    ('f1_sem hall 2', 'f1_j5'),
    ('f1_stairs 3', 'f1_j5'),
    ('f1_stairs 3', 'f1_j6'),
    ('f1_classroom 118', 'f1_j6'),
    ('f1_classroom 118', 'f1_j5'),
    ('f1_stairs 3', 'f1_j4'),
    ('f1_classroom 117', 'f1_j7'),
    ('f1_female wr', 'f1_j7'),
    ('f1_staff room 19', 'f1_j8'),
    ('f1_staff room 18', 'f1_j8'),
    ('f1_staff room 17', 'f1_j9'),
    ('f1_staff room 16', 'f1_j9'),
    ('f1_staff room 4', 'f1_j9'),
    ('f1_staff room 3', 'f1_j9'),
    ('f1_staff room 3', 'f1_j14'),
    ('f1_staff room 2', 'f1_j14'),
    ('f1_staff room 5', 'f1_j10'),
    ('f1_staff room 6', 'f1_j10'),
    ('f1_staff mens wr', 'f1_j10'),
    ('f1_staff mens wr', 'f1_j13'),
    ('f1_staff room hod', 'f1_j13'),
    ('f1_staff room 7', 'f1_j13'),
    ('f1_staff room 8', 'f1_j13'),
    ('f1_staff room 9', 'f1_j12'),
    ('f1_conference room 1', 'f1_j12'),
    ('f1_staff room 10', 'f1_j12'),
    ('f1_staff room 11', 'f1_j12'),
    ('f1_staff room 8', 'f1_j12'),
    ('f1_staff room 12', 'f1_j11'),
    ('f1_staff room 13', 'f1_j11'),
    ('f1_staff room 11', 'f1_j11'),
    ('f1_stairs 2', 'f1_j11'),
    ('f1_stairs 2', 'f1_j10'),
    ('f1_staff female wr', 'f1_j10'),
    ('f1_staff female wr', 'f1_stairs 2'),
    ('f1_staff room 15', 'f1_j10'),
    ('f1_staff room 14', 'f1_j10'),
    ('f1_staff room 1', 'f1_j14'),
    ('f1_staff room 1', 'f1_j15'),
    ('f1_male wr', 'f1_j15'),
    ('f1_classroom 125', 'f1_j15'),
    ('f1_classroom 124', 'f1_j16'),
    ('f1_stairs 1', 'f1_j16'),
    ('f1_sem hall 1', 'f1_j17'),
    ('f1_lab', 'f1_j17'),
    ('f1_lift 4', 'f1_j16'),
    ('f1_lift 3', 'f1_j16'),
    ('f1_lift 4', 'f1_j18'),
    ('f1_lift 3', 'f1_j18'),
    ('f1_j16', 'f1_sem hall 1'),
    ('f1_stairs 1', 'f1_j17'),
    ('f1_classroom 124', 'f1_j17'),
    ('f1_lift 4', 'f1_j17'),
    ('f1_j17', 'f1_lift 3'),
]

# --- THE BRIDGES (Connecting Ground to First) ---
bridges = [
    ('stairs 1', 'f1_stairs 1'),
    ('stairs 2', 'f1_stairs 2'),
    ('stairs 3', 'f1_stairs 3'),
    ('stairs 4', 'f1_stairs 4'),
    ('lift', 'f1_lift 1'),    # Note: Double check if your F1 is 'f1_lift 1' or 2
    ('lift 3', 'f1_lift 3'),
    ('lift 4', 'f1_lift 4'),
    ('lift 2', 'f1_lift 2')
]

# COMBINE ALL EDGES
all_edges = edges_g + edges_f1 + bridges


# ==========================================
# 5. DRAW THE 3D MAP
# ==========================================
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw the Nodes (Dots and Text)
for name, (x, y, z) in all_nodes_3d.items():
    # Ground = Red, First Floor = Blue
    color = '#ff4d4d' if z == 0 else '#4d94ff'
    ax.scatter(x, y, z, c=color, s=25, depthshade=True)
    
    # Hide the text of junction nodes ('j1', 'f1_j2') to keep the map clean
    if not name.startswith('j') and not name.startswith('f1_j'):
        ax.text(x, y, z, f" {name}", size=6, color='black', zorder=10)

# Draw the Edges (Lines)
for start, end in all_edges:
    if start in all_nodes_3d and end in all_nodes_3d:
        x_coords = [all_nodes_3d[start][0], all_nodes_3d[end][0]]
        y_coords = [all_nodes_3d[start][1], all_nodes_3d[end][1]]
        z_coords = [all_nodes_3d[start][2], all_nodes_3d[end][2]]
        
        # Check if it's a bridge (Z-coordinates are different)
        if all_nodes_3d[start][2] != all_nodes_3d[end][2]:
            ax.plot(x_coords, y_coords, z_coords, color='#2ca02c', linestyle='--', linewidth=3, label='Stairs/Lift')
        else:
            # Floor hallways
            line_color = '#ff9999' if all_nodes_3d[start][2] == 0 else '#99c2ff'
            ax.plot(x_coords, y_coords, z_coords, color=line_color, linewidth=1.5, alpha=0.8)

# Remove the messy grid and axes numbers for a clean look
ax.set_axis_off()
ax.set_title("3D College Building Navigation Map", fontsize=16, fontweight='bold', pad=20)

# Adjust the default camera angle so you can see both floors easily
ax.view_init(elev=20, azim=-45)

# Show the plot
plt.tight_layout()
plt.show()