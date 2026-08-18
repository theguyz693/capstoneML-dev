import matplotlib.pyplot as plt
from PIL import Image

# 1. Load your Second Floor Blueprint
img_path = 'second_floor_hd.jpg' 
img = Image.open(img_path)

fig, ax = plt.subplots(figsize=(16, 12))
mng = plt.get_current_fig_manager()
try:
    mng.window.state('zoomed')
except:
    pass

ax.imshow(img)

# This dictionary will store your SECOND FLOOR nodes
f2_captured_nodes = {}

# Render existing nodes (empty at first, but good if you have to restart)
for name, (ix, iy) in f2_captured_nodes.items():
    ax.plot(ix, iy, 'ro', markersize=2)
    ax.text(ix, iy, f"  {name}", color='red', fontsize=4, fontweight='bold', alpha=0.7)
    
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        ix, iy = int(event.xdata), int(event.ydata)
        
        print(f"\nCaptured Coordinates: ({ix}, {iy})")
        raw_name = input("Enter room name (e.g., classroom 201): ").strip().lower()
        
        if raw_name:
            # Automatically adds 'f2_' 
            full_name = f"f2_{raw_name}"
            
            f2_captured_nodes[full_name] = (ix, iy)
            
            ax.plot(ix, iy, 'ro', markersize=2)
            ax.text(ix, iy, f"  {full_name}", color='red', fontsize=4, fontweight='bold', alpha=0.7)
            plt.draw()
            
            print(f"Added: {full_name} at {f2_captured_nodes[full_name]}")

cid = fig.canvas.mpl_connect('button_press_event', onclick)

print("--- SECOND FLOOR NODE PICKER ---")
print("1. Click a room or junction on the SECOND floor.")
print("2. Type the name in the terminal (the script adds 'f2_' automatically).")
print("3. IMPORTANT: Map the STAIRS and LIFT exactly where they are on this floor!")

plt.show()

# Print the final dictionary
print("\n" + "="*40)
print("COPY THIS INTO YOUR SECOND FLOOR DATA SECTION:")
print("="*40)
print("nodes_f2 = {")
for name, coords in f2_captured_nodes.items():
    print(f"    '{name}': {coords},")
print("}")