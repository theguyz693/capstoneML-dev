"""
generate_dataset.py
Generates a physics-grounded synthetic training dataset for ML-based edge cost estimation.

Features:
- distance (float, pixels / distance units)
- hazard_level (float, 0.0 to 1.0)
- crowd_density (float, 0.0 to 1.0)
- accessibility (float, 1.0 = accessible, 0.0 = stairs/non-accessible)

Target:
- edge_cost (float, effective traversal cost used by shortest path algorithms)
"""

import os
import numpy as np
import pandas as pd

def calculate_ground_truth_cost(distance, hazard_level, crowd_density, accessibility, add_noise=True):
    """
    Computes grounded effective edge traversal cost based on evacuation dynamics.
    """
    # Speed degradation from crowd congestion (Weidmann crowd speed decay)
    speed_factor = np.maximum(0.15, 1.0 - 0.75 * (crowd_density ** 1.3))
    
    # Hazard danger penalty: exponential scaling to strongly deter entering hazardous zones
    hazard_penalty = 1.0 + 12.0 * (hazard_level ** 2) + 60.0 * np.maximum(0.0, hazard_level - 0.7) ** 3
    
    # Accessibility penalty: significant extra impedance on non-accessible paths
    access_penalty = 1.0 + 4.0 * (1.0 - accessibility)
    
    # Base effective cost (proportional to traversal time & perceived risk)
    cost = (distance / speed_factor) * hazard_penalty * access_penalty
    
    # Severe hazard barrier (> 0.90 is effectively impassable)
    if isinstance(hazard_level, np.ndarray):
        impassable_mask = hazard_level >= 0.92
        cost[impassable_mask] = np.maximum(cost[impassable_mask], 50000.0 + distance[impassable_mask] * 20.0)
    else:
        if hazard_level >= 0.92:
            cost = max(cost, 50000.0 + distance * 20.0)
            
    if add_noise:
        noise = np.random.normal(1.0, 0.025, size=np.shape(cost))
        cost = cost * np.clip(noise, 0.92, 1.08)
        
    return np.round(cost, 2)

def generate_evacuation_dataset(num_samples=12000, random_seed=42, output_path="data/evacuation_edge_data.csv"):
    """
    Generates a diverse set of edge condition scenarios.
    """
    np.random.seed(random_seed)
    
    # Distance distribution matching actual building floor plan edge lengths (30 to 2500 units)
    distances = np.random.uniform(30.0, 2500.0, size=num_samples)
    
    # Hazard levels: combination of clean conditions (0.0), low risk, and hot emergencies
    hazard_probs = np.random.choice(['clean', 'low', 'medium', 'high', 'critical'], size=num_samples, p=[0.45, 0.20, 0.15, 0.12, 0.08])
    hazard_levels = np.zeros(num_samples)
    for i, h_type in enumerate(hazard_probs):
        if h_type == 'clean':
            hazard_levels[i] = np.random.uniform(0.0, 0.05)
        elif h_type == 'low':
            hazard_levels[i] = np.random.uniform(0.05, 0.30)
        elif h_type == 'medium':
            hazard_levels[i] = np.random.uniform(0.30, 0.65)
        elif h_type == 'high':
            hazard_levels[i] = np.random.uniform(0.65, 0.90)
        else: # critical
            hazard_levels[i] = np.random.uniform(0.90, 1.00)
            
    # Crowd densities: mixture of empty, light, moderate, and jammed corridors
    crowd_probs = np.random.choice(['empty', 'light', 'moderate', 'dense', 'jammed'], size=num_samples, p=[0.35, 0.25, 0.20, 0.12, 0.08])
    crowd_densities = np.zeros(num_samples)
    for i, c_type in enumerate(crowd_probs):
        if c_type == 'empty':
            crowd_densities[i] = np.random.uniform(0.0, 0.10)
        elif c_type == 'light':
            crowd_densities[i] = np.random.uniform(0.10, 0.35)
        elif c_type == 'moderate':
            crowd_densities[i] = np.random.uniform(0.35, 0.65)
        elif c_type == 'dense':
            crowd_densities[i] = np.random.uniform(0.65, 0.85)
        else: # jammed
            crowd_densities[i] = np.random.uniform(0.85, 1.00)
            
    # Accessibility: ~75% accessible corridors/ramps/elevators, ~25% stairs/restricted paths
    accessibility = np.random.choice([1.0, 0.0], size=num_samples, p=[0.75, 0.25])
    
    # Calculate target edge costs
    edge_costs = calculate_ground_truth_cost(distances, hazard_levels, crowd_densities, accessibility, add_noise=True)
    
    df = pd.DataFrame({
        'distance': np.round(distances, 2),
        'hazard_level': np.round(hazard_levels, 4),
        'crowd_density': np.round(crowd_densities, 4),
        'accessibility': accessibility.astype(float),
        'edge_cost': edge_costs
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at '{output_path}' with {len(df)} samples.")
    print("\nDataset Summary Statistics:")
    print(df.describe())
    return df

if __name__ == '__main__':
    generate_evacuation_dataset()
