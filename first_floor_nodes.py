import matplotlib.pyplot as plt
from PIL import Image

# 1. Load your First Floor Blueprint
# Change 'first_floor_hd.jpg' to your actual filename
img_path = 'first_floor_hd.jpg' 
try:
    img = Image.open(img_path)
except FileNotFoundError:
    print(f"❌ Error: {img_path} not found. Make sure the image is in the same folder!")
    exit()

fig, ax = plt.subplots(figsize=(16, 12))
mng = plt.get_current_fig_manager()
try:
    mng.window.state('zoomed') # Maximize for better clicking
except:
    pass

ax.imshow(img)

# This dictionary will store your FIRST FLOOR nodes
f1_captured_nodes = {
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

# --- ADD THIS BLOCK TO RENDER EXISTING NODES ---
for name, (ix, iy) in f1_captured_nodes.items():
    ax.plot(ix, iy, 'ro', markersize=2)
    ax.text(ix, iy, f"  {name}", color='red', fontsize=4 ,
            fontweight='bold', alpha=0.7)
    
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        ix, iy = int(event.xdata), int(event.ydata)
        
        print(f"\nCaptured Coordinates: ({ix}, {iy})")
        raw_name = input("Enter room name (e.g., classroom 101): ").strip().lower()
        
        if raw_name:
            # We automatically add 'f1_' so you don't have to type it every time
            full_name = f"f1_{raw_name}"
            
            f1_captured_nodes[full_name] = (ix, iy)
            
            # Draw on screen
            ax.plot(ix, iy, 'ro', markersize=2)
            ax.text(ix, iy, f"  {full_name}", color='red', fontsize=4, 
                    fontweight='bold', alpha=0.7)
            plt.draw()
            
            print(f"Added: {full_name} at {f1_captured_nodes[full_name]}")

cid = fig.canvas.mpl_connect('button_press_event', onclick)

print("--- FIRST FLOOR NODE PICKER ---")
print("1. Click a room or junction on the FIRST floor.")
print("2. Type the name in the terminal (the script adds 'f1_' automatically).")
print("3. IMPORTANT: Map the STAIRS and LIFT exactly where they are on this floor!")

plt.show()

# 4. Print the final dictionary for you to copy
print("\n" + "="*40)
print("COPY THIS INTO YOUR FIRST FLOOR DATA SECTION:")
print("="*40)
print("nodes_f1 = {")
for name, coords in f1_captured_nodes.items():
    print(f"    '{name}': {coords},")
print("}")