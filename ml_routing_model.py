"""
ml_routing_model.py
Inference module for predicting dynamic edge traversal costs using the trained Random Forest model.
"""

import os
import joblib
import numpy as np
import pandas as pd

class MLEdgeCostPredictor:
    """
    ML predictor wrapper for graph edge weights.
    """
    DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "edge_cost_rf.joblib")
    IMPASSABLE_COST = 1e7  # Cost representing physically blocked or lethal hazard
    
    def __init__(self, model_path=None):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.model = None
        self.feature_cols = ['distance', 'hazard_level', 'crowd_density', 'accessibility']
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"[MLEdgeCostPredictor] Loaded model from: {self.model_path}")
            except Exception as e:
                print(f"[MLEdgeCostPredictor] Warning: Error loading model ({e}). Using heuristic fallback.")
                self.model = None
        else:
            print(f"[MLEdgeCostPredictor] Warning: Model file not found at {self.model_path}. Using heuristic fallback.")
            self.model = None

    def predict_cost(self, distance: float, hazard_level: float = 0.0, 
                     crowd_density: float = 0.0, accessibility: float = 1.0) -> float:
        """
        Predicts traversal cost for a single edge.
        """
        # Distance zero check
        if distance <= 0:
            return 0.0
            
        # Hard safety barrier for severe hazards
        if hazard_level >= 0.92:
            return self.IMPASSABLE_COST
            
        # Bound inputs
        hazard = float(np.clip(hazard_level, 0.0, 1.0))
        crowd = float(np.clip(crowd_density, 0.0, 1.0))
        access = float(np.clip(accessibility, 0.0, 1.0))
        dist = float(max(1.0, distance))
        
        if self.model is not None:
            features = pd.DataFrame([{
                'distance': dist,
                'hazard_level': hazard,
                'crowd_density': crowd,
                'accessibility': access
            }])
            cost = float(self.model.predict(features)[0])
            return max(dist, cost)  # Traversal cost cannot be less than physical distance
        else:
            # Heuristic fallback if model not loaded
            speed_factor = max(0.15, 1.0 - 0.75 * (crowd ** 1.3))
            hazard_penalty = 1.0 + 12.0 * (hazard ** 2) + 60.0 * max(0.0, hazard - 0.7) ** 3
            access_penalty = 1.0 + 4.0 * (1.0 - access)
            return float((dist / speed_factor) * hazard_penalty * access_penalty)

    def batch_predict(self, edge_features_list) -> list:
        """
        Predicts traversal costs for a list of dictionaries with keys:
        'distance', 'hazard_level', 'crowd_density', 'accessibility'
        """
        if not edge_features_list:
            return []
            
        df = pd.DataFrame(edge_features_list)
        for col in self.feature_cols:
            if col not in df.columns:
                if col == 'hazard_level' or col == 'crowd_density':
                    df[col] = 0.0
                elif col == 'accessibility':
                    df[col] = 1.0
                elif col == 'distance':
                    df[col] = 100.0
                    
        df['distance'] = df['distance'].clip(lower=0.0)
        df['hazard_level'] = df['hazard_level'].clip(0.0, 1.0)
        df['crowd_density'] = df['crowd_density'].clip(0.0, 1.0)
        df['accessibility'] = df['accessibility'].clip(0.0, 1.0)
        
        if self.model is not None:
            preds = self.model.predict(df[self.feature_cols])
        else:
            speed = np.maximum(0.15, 1.0 - 0.75 * (df['crowd_density'] ** 1.3))
            haz = 1.0 + 12.0 * (df['hazard_level'] ** 2) + 60.0 * np.maximum(0.0, df['hazard_level'] - 0.7) ** 3
            acc = 1.0 + 4.0 * (1.0 - df['accessibility'])
            preds = (df['distance'] / speed) * haz * acc
            
        # Apply safety threshold and minimum bounds
        results = []
        for i, row in df.iterrows():
            if row['hazard_level'] >= 0.92:
                results.append(self.IMPASSABLE_COST)
            elif row['distance'] <= 0:
                results.append(0.0)
            else:
                results.append(float(max(row['distance'], preds[i])))
                
        return results

# Singleton instance for quick module-level access
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = MLEdgeCostPredictor()
    return _predictor

def predict_edge_cost(distance: float, hazard_level: float = 0.0, 
                      crowd_density: float = 0.0, accessibility: float = 1.0) -> float:
    return get_predictor().predict_cost(distance, hazard_level, crowd_density, accessibility)
