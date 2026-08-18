"""
visualize_ml_routing.py
Visualizes static vs ML dynamic evacuation paths over the high-definition building floor plan image.
"""

import sys
import os
import math
import matplotlib.pyplot as plt
from PIL import Image

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from campus_data import nodes_g_raw
from ml_edge_updater import build_ground_floor_graph, apply_environmental_scenario
from ml_evacuation_router import compare_static_vs_dynamic_routing

def plot_evacuation_comparison(start_node='classroom g17', fire_node='j6', save_path='evacuation_comparison.png'):
    """
    Renders side-by-side or overlaid static vs ML dynamic paths on the floor plan image.
    """
    g_clean = build_ground_floor_graph()
    g_fire = build_ground_floor_graph()
    apply_environmental_scenario(g_fire, hazard_sources=[{'node': fire_node, 'radius': 280, 'intensity': 1.0}])
    
    comp = compare_static_vs_dynamic_routing(g_clean, g_fire, start_node)
    if not comp:
        print("Could not compute route comparison.")
        return
        
    static_route = comp['static_route_with_current_hazards']
    dynamic_route = comp['dynamic_ml_route']
    
    img_path = 'floor_plan_hd.jpg'
    if not os.path.exists(img_path):
        print(f"Floor plan image '{img_path}' not found. Skipping plot.")
        return
        
    img = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.imshow(img)
    
    # 1. Plot Fire Hazard Zone
    if fire_node in nodes_g_raw:
        fx, fy = nodes_g_raw[fire_node]
        fire_circle = plt.Circle((fx, fy), 280, color='red', alpha=0.35, label='Fire Danger Zone (j6)')
        ax.add_patch(fire_circle)
        ax.scatter([fx], [fy], color='darkred', s=250, marker='*', zorder=10, label='Fire Origin')
        ax.text(fx + 20, fy - 20, 'FIRE OUTBREAK', color='red', fontweight='bold', fontsize=12, backgroundcolor='white')
        
    # 2. Plot Static Path (Danger / Red dashed)
    sp_nodes = static_route['path']
    sp_x = [nodes_g_raw[n][0] for n in sp_nodes if n in nodes_g_raw]
    sp_y = [nodes_g_raw[n][1] for n in sp_nodes if n in nodes_g_raw]
    ax.plot(sp_x, sp_y, color='red', linestyle='--', linewidth=3.5, alpha=0.85, 
            label=f"Static Shortest Path (Enters Fire! Exit: {static_route['destination_exit']})")
    
    # 3. Plot ML Dynamic Path (Safe / Bright Lime green solid)
    dp_nodes = dynamic_route['path']
    dp_x = [nodes_g_raw[n][0] for n in dp_nodes if n in nodes_g_raw]
    dp_y = [nodes_g_raw[n][1] for n in dp_nodes if n in nodes_g_raw]
    ax.plot(dp_x, dp_y, color='lime', linewidth=5, alpha=0.9, 
            label=f"ML Dynamic Safe Path (Avoids Fire -> {dynamic_route['destination_exit']})")
    ax.scatter(dp_x, dp_y, color='lime', s=35, edgecolors='black', zorder=5)
    
    # 4. Mark Start & Exits
    start_pos = nodes_g_raw[start_node]
    ax.scatter(start_pos[0], start_pos[1], color='blue', s=200, marker='X', zorder=10, label=f"Start: {start_node}")
    ax.text(start_pos[0], start_pos[1] - 40, f"START: {start_node}", color='blue', fontweight='bold', backgroundcolor='white')
    
    # Mark target exits
    if static_route['destination_exit'] in nodes_g_raw:
        ep = nodes_g_raw[static_route['destination_exit']]
        ax.scatter(ep[0], ep[1], color='red', s=150, marker='s', zorder=8)
        
    if dynamic_route['destination_exit'] in nodes_g_raw:
        ep = nodes_g_raw[dynamic_route['destination_exit']]
        ax.scatter(ep[0], ep[1], color='green', s=180, marker='s', zorder=8)
        ax.text(ep[0], ep[1] + 40, f"SAFE EXIT: {dynamic_route['destination_exit']}", color='darkgreen', fontweight='bold', backgroundcolor='white')

    ax.set_title(f"Dynamic ML Evacuation Routing vs Static Baseline (Origin: {start_node})", fontsize=15, fontweight='bold', pad=15)
    ax.axis('off')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Evacuation comparison plot saved successfully to '{save_path}'.")
    plt.close()

if __name__ == '__main__':
    plot_evacuation_comparison()
