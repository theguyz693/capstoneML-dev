import matplotlib.pyplot as plt
from PIL import Image

# 1. Load your HD Blueprint
img_path = 'floor_plan_hd.jpg' 
img = Image.open(img_path)

fig, ax = plt.subplots(figsize=(16, 12))
ax.imshow(img)

# --- PASTE YOUR DICTIONARY HERE ---
captured_nodes = {'sem hall 1': (1039, 944), 'computer lab': (1586, 919), 'stairs 1': (1179, 1174), 'classroom g24': (1230, 1354), 'classroom g25': (1222, 1585), 'lift': (2364, 1959), 'staff room 1': (1320, 1823), 'staff room 2': (1320, 1895), 'staff room 3': (1323, 1977), 'staff room 4': (1327, 2049), 'staff room 5': (1248, 2136), 'staff room 6': (1190, 2139), 'staff room hod': (873, 2179), 'staff room 7': (877, 2294), 'staff room 8': (880, 2377), 'staff room 9': (891, 2460), 'conference room 1': (920, 2600), 'staff room 10': (1050, 2618), 'staff room 11': (1132, 2626), 'staff room 12': (1215, 2615), 'staff room 13': (1341, 2626), 'staff room 14': (1370, 2323), 'staff room 15': (1370, 2255), 'staff room 16': (1449, 2183), 'staff room 17': (1536, 2168), 'staff room 18': (1608, 2175), 'staff room 19': (1687, 2175), 'stairs 2': (1338, 2446), 'male wr': (1158, 1718), 'female wr': (1791, 2320), 'classroom g18': (1932, 2262), 'classroom g17': (2151, 2266), 'stairs 3': (2331, 2302), 'sem hall 2': (2566, 2453), 'chemistry lab': (2602, 1898), 'east entrance/exit': (2666, 2129), 'north entrance/exit': (1348, 861), 'stairs 4': (1993, 1038), 'office': (2425, 1347), 'bank': (2159, 1088), 'cabin 1': (2753, 1427), 'cabin 2': (2756, 1297), 'cabin 3': (2674, 1106), 'cabin 4': (2407, 847), 'cabin 5': (2220, 760), 'cabin 6': (2069, 764), 'east main entrance': (3016, 825), 'north main entrance': (2684, 547), 'atm': (1929, 1013), 'lift 2': (2295, 1956), 'lift 3': (1523, 1134), 'lift 4': (1527, 1211), 'staff fem wr': (1470, 2372), 'staff mens wr': (1134, 2074), 'electrical room 1': (2335, 1869), 'electrical room 2': (1617, 1154), 'j1': (2711, 812), 'j2': (2453, 1080), 'j3': (2205, 1305), 'j4': (2429, 1510), 'j5': (2429, 1986), 'j6': (2443, 2178), 'j7': (2265, 2094), 'j8': (1782, 2094), 'j9': (1617, 2097), 'j10': (1409, 2090), 'j11': (1255, 2238), 'j12': (1309, 2537), 'j13': (1017, 2503), 'j14': (973, 2191), 'j15': (1396, 1842), 'j16': (1396, 1711), 'j17': (1389, 1211), 'j18': (1312, 1067), 'j19': (1513, 1067), 'j20': (1986, 1077)}

# --- ADD THIS: DRAW PREVIOUS NODES SO YOU SEE THEM ---
for name, (ix, iy) in captured_nodes.items():
    ax.plot(ix, iy, 'ro', markersize=4) # Increased markersize slightly for HD
    ax.text(ix, iy, f"  {name}", color='red', fontsize=6, fontweight='bold')

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        ix, iy = int(event.xdata), int(event.ydata)
        
        print(f"\nCaptured Coordinates: ({ix}, {iy})")
        name = input("Enter the name for this room: ").strip()
        
        if name:
            captured_nodes[name] = (ix, iy)
            ax.plot(ix, iy, 'ro', markersize=4)
            ax.text(ix, iy, f"  {name}", color='red', fontsize=6, fontweight='bold')
            plt.draw()
            print(f"Current Dictionary: {captured_nodes}")

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
