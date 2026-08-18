"""
demo_scenarios.py
Interactive CLI demonstration of Dynamic Crowd + Hazard ML Evacuation Routing.
Demonstrates 5 emergency scenarios showing how ML-predicted costs change evacuation paths.
"""

import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from ml_routing_model import get_predictor
from ml_edge_updater import build_ground_floor_graph, build_full_campus_graph, apply_environmental_scenario
from ml_evacuation_router import find_optimal_evacuation_routes, compare_static_vs_dynamic_routing

def print_banner(title):
    print("\n" + "="*70)
    print(f" {title.upper()}")
    print("="*70)

def print_route_card(label, route_info):
    if not route_info:
        print(f"  [{label}] No route available!")
        return
    print(f"\n  --- {label} ---")
    print(f"  Target Exit    : {route_info.get('destination_exit', 'N/A')}")
    print(f"  Physical Dist  : {route_info.get('physical_distance', 0)} px units")
    print(f"  ML Edge Cost   : {route_info.get('predicted_cost', 0):.2f}")
    print(f"  Peak Hazard    : {route_info.get('max_hazard', 0.0):.2f} / 1.00")
    print(f"  Avg Congestion : {route_info.get('avg_crowd', 0.0):.2f} / 1.00")
    print(f"  Safety Status  : {'[SAFE]' if route_info.get('is_safe', True) else '[DANGEROUS]'}")
    path = route_info.get('path', [])
    print(f"  Path Sequence  : {' -> '.join(path)}")

def run_scenario_1_baseline():
    print_banner("Scenario 1: Baseline Normal Conditions (Ground Floor)")
    print("Context: Building is safe. Pathfinding finds optimal shortest physical distance.")
    
    g_clean = build_ground_floor_graph()
    start_node = 'classroom g17'
    routes = find_optimal_evacuation_routes(g_clean, start_node)
    
    print(f"\nOrigin: '{start_node}'")
    print_route_card("Standard Evacuation Route", routes[0])

def run_scenario_2_fire_outbreak():
    print_banner("Scenario 2: Active Fire Outbreak at East Junction (j6)")
    print("Context: Fire breaks out at j6 (intensity 1.0). Static shortest path would lead straight")
    print("into the fire. ML routing raises edge weights dynamically to reroute to North Main Entrance.")
    
    start_node = 'classroom g17'
    g_clean = build_ground_floor_graph()
    g_fire = build_ground_floor_graph()
    apply_environmental_scenario(g_fire, hazard_sources=[{'node': 'j6', 'radius': 280, 'intensity': 1.0}])
    
    comp = compare_static_vs_dynamic_routing(g_clean, g_fire, start_node)
    
    print_route_card("Static Shortest Path (Without ML - Blunders into Fire!)", comp['static_route_with_current_hazards'])
    print_route_card("ML Dynamic Evacuation Path (Avoids Fire Zone)", comp['dynamic_ml_route'])
    print(f"\n[DECISION]: Path changed: {comp['path_changed']} | Exit rerouted: {comp['exit_changed']}")

def run_scenario_3_crowd_congestion():
    print_banner("Scenario 3: Severe Lobby Corridor Crowd Crush (j2)")
    print("Context: High crowd density (0.95) creates a severe crush bottleneck at junction j2.")
    print("Static routing blindly follows the shortest Euclidean path through the bottleneck (North Main Exit).")
    print("ML dynamic routing detects the congestion delay and reroutes occupants through the clear southern/eastern corridor (East Exit).")
    
    start_node = 'office'
    g_clean = build_ground_floor_graph()
    g_crowd = build_ground_floor_graph()
    apply_environmental_scenario(g_crowd, crowd_sources=[{'node': 'j2', 'radius': 300, 'intensity': 0.95}])
    
    comp = compare_static_vs_dynamic_routing(g_clean, g_crowd, start_node)
    
    print_route_card("Static Path (Bottlenecked at j2)", comp['static_route_with_current_hazards'])
    print_route_card("ML Dynamic Path (Clear Alternative via East Exit)", comp['dynamic_ml_route'])
    print(f"\n[DECISION]: Path changed: {comp['path_changed']} | Exit rerouted: {comp['exit_changed']}")

def run_scenario_4_wheelchair_accessibility():
    print_banner("Scenario 4: Wheelchair & Mobility-Restricted Evacuation")
    print("Context: Person with mobility impairment requires step-free / elevator-accessible evacuation.")
    print("ML routing assigns severe accessibility penalty to staircases, ensuring 100% step-free routing.")
    
    start_node = 'f1_lab'
    campus_access = build_full_campus_graph()
    apply_environmental_scenario(campus_access, mobility_impaired=True)
    
    routes = find_optimal_evacuation_routes(campus_access, start_node)
    print_route_card("Accessible Multi-Floor Evacuation Route", routes[0])

def run_scenario_5_multifloor_fire():
    print_banner("Scenario 5: Multi-Floor Evacuation with Blocked Primary Staircase")
    print("Context: Evacuating from Floor 1 (f1_classroom 124). Primary Stair 1 is compromised by smoke.")
    print("ML routing diverts the evacuation path through elevator/secondary vertical connector to ground exit.")
    
    start_node = 'f1_classroom 124'
    campus_clean = build_full_campus_graph()
    campus_fire = build_full_campus_graph()
    apply_environmental_scenario(campus_fire, hazard_sources=[{'node': 'f1_stairs 1', 'radius': 300, 'intensity': 1.0}])
    
    comp = compare_static_vs_dynamic_routing(campus_clean, campus_fire, start_node)
    print_route_card("Static Multi-Floor Path (Trapped in Smoke at Stair 1)", comp['static_route_with_current_hazards'])
    print_route_card("ML Dynamic Multi-Floor Path (Safe Alternative Exit)", comp['dynamic_ml_route'])

def main():
    print_banner("EMERGENCY EVACUATION & INDOOR NAVIGATION")
    print("FEATURE 1: DYNAMIC CROWD + HAZARD ML ROUTING DEMONSTRATION")
    
    run_scenario_1_baseline()
    run_scenario_2_fire_outbreak()
    run_scenario_3_crowd_congestion()
    run_scenario_4_wheelchair_accessibility()
    run_scenario_5_multifloor_fire()
    
    print("\n" + "="*70)
    print(" [COMPLETE] All 5 Demonstration Scenarios Executed Successfully.")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
