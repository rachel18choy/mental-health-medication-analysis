import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

class InferenceHelper:
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        
    def load_artifacts(self):
        """Load preprocessing artifacts"""
        try:
            self.scaler = joblib.load("../models/saved_models/scaler.pkl")
            
            # Load feature names
            with open("../models/saved_models/feature_names.txt", 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            return True
        except Exception as e:
            print(f"Warning: Could not load artifacts: {e}")
            return False
    
    def preprocess_input(self, input_dict: dict) -> np.ndarray:
        """Preprocess a single input for prediction"""
        # This is a placeholder - you should implement the exact
        # preprocessing you used during training
        df = pd.DataFrame([input_dict])
        
        # TODO: Implement your actual preprocessing logic here
        # This should match what you did in scripts/preprocessing.py
        
        # For now, return a dummy array
        return np.zeros((1, 30))  # Assuming 30 features

# Create global instance
inference_helper = InferenceHelper()