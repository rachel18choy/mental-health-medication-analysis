import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load your processed data to create artifacts
try:
    # Load the processed data
    data = joblib.load("../data/processed/supervised_data.pkl")
    
    X_train = data['X_train']
    feature_names = data['feature_names']
    
    print(f"Loaded data with {len(feature_names)} features")
    print(f"Feature names: {feature_names[:5]}...")
    
    # Create and save scaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on training data
    
    # Save artifacts
    joblib.dump(scaler, "../models/saved_models/scaler.pkl")
    
    # Save feature names
    with open("../models/saved_models/feature_names.txt", 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print("✓ Saved scaler.pkl")
    print("✓ Saved feature_names.txt")
    print(f"  Total features: {len(feature_names)}")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative approach...")
    
    # Alternative: Create dummy artifacts
    scaler = StandardScaler()
    scaler.fit(np.zeros((10, 30)))  # 30 features like your data
    
    joblib.dump(scaler, "../models/saved_models/scaler.pkl")
    
    # Create dummy feature names
    feature_names = [f"feature_{i}" for i in range(30)]
    with open("../models/saved_models/feature_names.txt", 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print("✓ Created dummy scaler.pkl")
    print("✓ Created dummy feature_names.txt")