"""
train_model.py
Trains and evaluates a Random Forest Regressor to predict edge traversal cost
from environmental and structural features (distance, hazard_level, crowd_density, accessibility).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_edge_cost_model(data_path="data/evacuation_edge_data.csv", model_dir="models"):
    print("==================================================")
    print("PHASE C: RANDOM FOREST REGRESSOR TRAINING")
    print("==================================================")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}. Run generate_dataset.py first.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} training samples from '{data_path}'.")
    
    feature_cols = ['distance', 'hazard_level', 'crowd_density', 'accessibility']
    target_col = 'edge_cost'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 80/20 Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # Configure Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest Regressor...")
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Mean Absolute Percentage Error (excluding edge_cost < 1 to prevent div by zero)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / np.maximum(y_test, 1.0))) * 100.0
    
    print("\n--- MODEL PERFORMANCE EVALUATION ---")
    print(f"Train R² Score : {train_r2:.4f}")
    print(f"Test R² Score  : {test_r2:.4f}")
    print(f"Test MAE       : {test_mae:.2f}")
    print(f"Test RMSE      : {test_rmse:.2f}")
    print(f"Test MAPE      : {test_mape:.2f}%")
    
    # Feature Importances
    importances = rf_model.feature_importances_
    print("\n--- FEATURE IMPORTANCES ---")
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"  {col:<15}: {imp * 100:.2f}%")
        
    # Save Model & Metadata
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "edge_cost_rf.joblib")
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    
    joblib.dump(rf_model, model_path)
    print(f"\nModel saved successfully to '{model_path}'.")
    
    metadata = {
        "model_type": "RandomForestRegressor",
        "feature_cols": feature_cols,
        "target_col": target_col,
        "n_samples": len(df),
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 18,
            "min_samples_split": 4,
            "min_samples_leaf": 2
        },
        "metrics": {
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "test_mape_percent": float(test_mape)
        },
        "feature_importances": {col: float(imp) for col, imp in zip(feature_cols, importances)}
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved successfully to '{metadata_path}'.")
    
    return rf_model, metadata

if __name__ == '__main__':
    train_edge_cost_model()
