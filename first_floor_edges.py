import matplotlib.pyplot as plt
from PIL import Image
import math

# 1. YOUR SAVED FIRST FLOOR NODES
nodes_f1 = {
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
temp_node = None 

# 2. LOAD IMAGE AND MAXIMIZE
# Update this filename to your actual First Floor image!
img = Image.open('first_floor_hd.jpg') 
fig, ax = plt.subplots(figsize=(20, 12))

mng = plt.get_current_fig_manager()
try:
    mng.window.state('zoomed') # Full screen for Windows
except:
    pass

ax.imshow(img)

# 3. DRAW EXISTING NODES (So you know where to click)
for name, (nx, ny) in nodes_f1.items():
    ax.plot(nx, ny, 'ro', markersize=4)
    ax.text(nx, ny, f" {name}", color='red', fontsize=4, alpha=0.7)

def find_nearest_node(x, y):
    min_dist = 100 # Adjusted for high-res first floor image
    closest_name = None
    for name, (nx, ny) in nodes_f1.items():
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
                # First click: Highlight yellow
                temp_node = clicked_node
                print(f"Start: {temp_node} -> Click next node...")
                ax.plot(nodes_f1[temp_node][0], nodes_f1[temp_node][1], 'yo', markersize=10) 
                plt.draw()
            else:
                # Second click: Connect
                if temp_node != clicked_node:
                    edges_f1.append((temp_node, clicked_node))
                    x_vals = [nodes_f1[temp_node][0], nodes_f1[clicked_node][0]]
                    y_vals = [nodes_f1[temp_node][1], nodes_f1[clicked_node][1]]
                    ax.plot(x_vals, y_vals, 'b-', linewidth=2)
                    print(f"Connected: {temp_node} <--> {clicked_node}")
                temp_node = None
                plt.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.title("F1 EDGE CONNECTOR: Connect your First Floor Nodes", fontsize=7)
plt.axis('off')
plt.show()

# 4. FINAL OUTPUT
print("\n" + "="*30)
print("COPY THESE FIRST FLOOR EDGES:")
print("="*30)
print("edges_f1 = [")
for edge in edges_f1:
    print(f"    {edge},")
print("]")