"""
test_ml_routing.py
Automated test suite verifying the ML Dynamic Crowd + Hazard Routing pipeline.
"""

import sys
import unittest
import networkx as nx

# Ensure UTF-8 output encoding across Windows shells
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from ml_routing_model import get_predictor, MLEdgeCostPredictor
from ml_edge_updater import build_ground_floor_graph, build_full_campus_graph, apply_environmental_scenario
from ml_evacuation_router import find_optimal_evacuation_routes, compare_static_vs_dynamic_routing, GROUND_EXITS

class TestMLRoutingPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()
        cls.g_clean = build_ground_floor_graph()
        cls.campus_clean = build_full_campus_graph()
        
    def test_01_model_predictions(self):
        """Verify basic ML model prediction behavior and constraints."""
        p = self.predictor
        base_dist = 500.0
        
        cost_clean = p.predict_cost(distance=base_dist, hazard_level=0.0, crowd_density=0.0, accessibility=1.0)
        cost_crowded = p.predict_cost(distance=base_dist, hazard_level=0.0, crowd_density=0.8, accessibility=1.0)
        cost_hazard = p.predict_cost(distance=base_dist, hazard_level=0.7, crowd_density=0.0, accessibility=1.0)
        cost_impassable = p.predict_cost(distance=base_dist, hazard_level=0.95, crowd_density=0.0, accessibility=1.0)
        
        self.assertGreaterEqual(cost_clean, base_dist * 0.95, "Clean cost should match distance closely")
        self.assertGreater(cost_crowded, cost_clean, "Crowd should increase traversal cost")
        self.assertGreater(cost_hazard, cost_crowded, "Hazard should incur heavy risk penalty")
        self.assertGreaterEqual(cost_impassable, 1e6, "Severe hazard must trigger impassable barrier cost")
        print("[PASS] Test 1: ML Model prediction bounds and penalties verified.")

    def test_02_graph_structure(self):
        """Verify ground floor and multi-floor graphs build properly with attributes."""
        self.assertGreater(len(self.g_clean.nodes), 40, "Ground floor should contain >40 nodes")
        self.assertGreater(len(self.g_clean.edges), 40, "Ground floor should contain >40 edges")
        self.assertGreater(len(self.campus_clean.nodes), 200, "Full campus graph should contain >200 nodes")
        
        # Check edge attribute schema
        for u, v, d in self.g_clean.edges(data=True):
            self.assertIn('weight', d)
            self.assertIn('hazard_level', d)
            self.assertIn('crowd_density', d)
            self.assertIn('accessibility', d)
            self.assertIn('base_distance', d)
        print(f"[PASS] Test 2: Graph topology verified ({len(self.campus_clean.nodes)} campus nodes, {len(self.campus_clean.edges)} edges).")

    def test_03_fire_hazard_rerouting(self):
        """Verify that a fire on the shortest path triggers dynamic rerouting around danger."""
        start_node = 'classroom g17'
        
        # In clean conditions, classroom g17 exits via 'east entrance/exit' (via j7 -> j6)
        clean_routes = find_optimal_evacuation_routes(self.g_clean, start_node)
        self.assertTrue(len(clean_routes) > 0)
        default_exit = clean_routes[0]['destination_exit']
        
        # Fire outbreak at j6 / east entrance/exit
        g_hazard = build_ground_floor_graph()
        apply_environmental_scenario(
            g_hazard,
            hazard_sources=[{'node': 'j6', 'radius': 250, 'intensity': 1.0}],
            predictor=self.predictor
        )
        
        comp = compare_static_vs_dynamic_routing(self.g_clean, g_hazard, start_node)
        self.assertIsNotNone(comp)
        
        # The dynamic route must avoid the severe hazard at j6
        self.assertTrue(comp['path_changed'], "Path must dynamically change to avoid fire")
        self.assertNotIn('j6', comp['dynamic_ml_route']['path'], "Dynamic path must avoid fire junction j6")
        self.assertLess(comp['dynamic_ml_route']['max_hazard'], 0.70, "Dynamic path must remain safe")
        print(f"[PASS] Test 3: Fire at j6 safely rerouted from {default_exit} to {comp['dynamic_ml_route']['destination_exit']}.")

    def test_04_crowd_congestion_rerouting(self):
        """Verify that severe crowd congestion causes the algorithm to take a genuinely different, lower-cost route."""
        start_node = 'office'
        
        # In clean conditions, 'office' takes northern path via j2 -> j1 -> north main entrance (dist ~906)
        clean_routes = find_optimal_evacuation_routes(self.g_clean, start_node)
        self.assertTrue(len(clean_routes) > 0)
        self.assertEqual(clean_routes[0]['destination_exit'], 'north main entrance')
        
        # Inject crowd crush at junction j2
        g_crowded = build_ground_floor_graph()
        apply_environmental_scenario(
            g_crowded,
            crowd_sources=[{'node': 'j2', 'radius': 300, 'intensity': 0.95}],
            predictor=self.predictor
        )
        
        comp = compare_static_vs_dynamic_routing(self.g_clean, g_crowd_graph:=g_crowded, start_node)
        self.assertIsNotNone(comp)
        
        # Assert that the dynamic path changes away from the crowded j2 route
        self.assertTrue(comp['path_changed'], "Crowd congestion at j2 must cause Dijkstra to select an alternative corridor")
        self.assertTrue(comp['exit_changed'], "Exit should switch from north main entrance to east entrance/exit")
        self.assertNotIn('j2', comp['dynamic_ml_route']['path'], "Dynamic route must completely bypass congested junction j2")
        self.assertLess(comp['dynamic_ml_route']['predicted_cost'], comp['static_route_with_current_hazards']['predicted_cost'],
                        "Dynamic route ML cost must be strictly lower than navigating through the crowd bottleneck")
        print(f"[PASS] Test 4: Severe crowd at j2 successfully rerouted path from {comp['static_route']['destination_exit']} to {comp['dynamic_ml_route']['destination_exit']} (Path: {' -> '.join(comp['dynamic_ml_route']['path'])}).")

    def test_05_multi_floor_stairwell_blocking(self):
        """Verify multi-floor evacuation adapts when a primary stairwell is compromised."""
        start_node = 'f1_classroom 124'
        
        # Block f1_stairs 1 with high fire hazard
        campus_hazard = build_full_campus_graph()
        apply_environmental_scenario(
            campus_hazard,
            hazard_sources=[{'node': 'f1_stairs 1', 'radius': 300, 'intensity': 1.0}],
            predictor=self.predictor
        )
        
        comp = compare_static_vs_dynamic_routing(self.campus_clean, campus_hazard, start_node)
        self.assertIsNotNone(comp)
        self.assertTrue(comp['path_changed'], "Multi-floor path must reroute when primary stairwell is blocked")
        self.assertNotIn('f1_stairs 1', comp['dynamic_ml_route']['path'], "Dynamic route must not use blocked stair 1")
        print(f"[PASS] Test 5: Multi-floor routing rerouted around blocked stairwell to safe ground exit: {comp['dynamic_ml_route']['destination_exit']}.")

    def test_06_wheelchair_accessibility_routing(self):
        """Verify that accessibility-restricted routing avoids staircases in favor of accessible paths."""
        start_node = 'f1_lab'
        
        # Build campus graph with mobility_impaired=True (stairs accessibility = 0.0)
        campus_accessible = build_full_campus_graph()
        apply_environmental_scenario(
            campus_accessible,
            mobility_impaired=True,
            predictor=self.predictor
        )
        
        routes = find_optimal_evacuation_routes(campus_accessible, start_node)
        self.assertTrue(len(routes) > 0)
        chosen_path = routes[0]['path']
        has_stairs = any('stairs' in n.lower() for n in chosen_path)
        self.assertFalse(has_stairs, f"Accessible route must avoid stairs! Route taken: {chosen_path}")
        print(f"[PASS] Test 6: Wheelchair accessibility routing verified (Path: {' -> '.join(chosen_path)}).")

if __name__ == '__main__':
    unittest.main(verbosity=2)
