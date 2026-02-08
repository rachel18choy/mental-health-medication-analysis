import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime
import os
from sklearn.preprocessing import LabelEncoder

class ModelInference:
    def __init__(self, model_path: str = "../models/saved_models/supervised/"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.model_name = None
        self.loaded = False
        
        # Load the best model (default is Random Forest)
        self.load_model()
    
    def load_model(self, model_name: str = "random_forest"):
        """Load trained model and preprocessing artifacts"""
        try:
            # Determine which model to load
            model_files = {
                'random_forest': 'random_forest.pkl',
                'logistic_regression': 'logistic_regression.pkl',
                'gradient_boosting': 'gradient_boosting.pkl',
                'svm': 'support_vector_machine.pkl'
            }
            
            # Check which models are available
            available_models = []
            for name, filename in model_files.items():
                filepath = os.path.join(self.model_path, filename)
                if os.path.exists(filepath):
                    available_models.append((name, filepath))
            
            if not available_models:
                raise FileNotFoundError(f"No trained models found in {self.model_path}")
            
            # Load the specified model or default to first available
            model_to_load = None
            for name, filepath in available_models:
                if name == model_name:
                    model_to_load = (name, filepath)
                    break
            
            if not model_to_load:
                model_to_load = available_models[0]  # Default to first available
            
            self.model_name, model_filepath = model_to_load
            self.model = joblib.load(model_filepath)
            
            # Load preprocessing artifacts
            artifacts_path = "../models/saved_models/"
            self.scaler = joblib.load(os.path.join(artifacts_path, "scaler.pkl"))
            
            # Load feature names
            feature_names_file = os.path.join(artifacts_path, "feature_names.txt")
            if os.path.exists(feature_names_file):
                with open(feature_names_file, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
            else:
                # If feature names file doesn't exist, create default names
                self.feature_names = [f"feature_{i}" for i in range(30)]  # Based on your data
            
            # Load label encoders if they exist
            label_encoders_file = os.path.join(artifacts_path, "label_encoders.pkl")
            if os.path.exists(label_encoders_file):
                self.label_encoders = joblib.load(label_encoders_file)
            else:
                self.label_encoders = {}
            
            self.loaded = True
            print(f"Model loaded successfully: {self.model_name}")
            print(f"Number of features expected: {len(self.feature_names)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.loaded = False
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess input data for prediction"""
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Apply label encoding for categorical variables
        if self.label_encoders:
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    # Handle unseen categories
                    try:
                        df[col] = encoder.transform(df[col].astype(str))
                    except ValueError:
                        # If category is unseen, use most frequent
                        df[col] = encoder.transform([encoder.classes_[0]] * len(df))
        
        # Convert study years to numeric
        if 'study_years' in df.columns:
            study_years_mapping = {
                '0 years': 0,
                '0-1 years': 0.5,
                '1-2 years': 1.5,
                '2-3 years': 2.5,
                '3-4 years': 3.5,
                '4-5 years': 4.5,
                '5-6 years': 5.5,
                '6-7 years': 6.5
            }
            df['study_years'] = df['study_years'].map(study_years_mapping)
            df['study_years'] = df['study_years'].fillna(df['study_years'].median())
        
        # Ensure all columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Ensure we have the right number of features
        if len(df.columns) != len(self.feature_names):
            # If feature count doesn't match, try to align
            if len(df.columns) > len(self.feature_names):
                df = df.iloc[:, :len(self.feature_names)]
            else:
                # Pad with zeros if needed
                missing = len(self.feature_names) - len(df.columns)
                for i in range(missing):
                    df[f'pad_{i}'] = 0
        
        # Scale features
        if self.scaler:
            df_scaled = self.scaler.transform(df)
        else:
            df_scaled = df.values
        
        return df_scaled
    
    def predict_single(self, input_data: Dict[str, Any]) -> Tuple[int, float, str]:
        """Make prediction for a single input"""
        if not self.loaded:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Preprocess input
        processed_input = self.preprocess_input(input_data)
        
        # Make prediction
        try:
            prediction = self.model.predict(processed_input)[0]
            
            # Get prediction probability if available
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(processed_input)[0][1]
            else:
                # For models without predict_proba, use decision function or default
                if hasattr(self.model, 'decision_function'):
                    prob = self.model.decision_function(processed_input)[0]
                    probability = 1 / (1 + np.exp(-prob))  # Sigmoid transformation
                else:
                    probability = 0.5 if prediction == 1 else 0.5
            
            # Determine confidence level
            confidence = self._get_confidence_level(probability)
            
            return int(prediction), float(probability), confidence
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise
    
    def predict_batch(self, input_data_list: List[Dict[str, Any]]) -> List[Tuple[int, float, str]]:
        """Make predictions for multiple inputs"""
        results = []
        for input_data in input_data_list:
            try:
                prediction, probability, confidence = self.predict_single(input_data)
                results.append((prediction, probability, confidence))
            except Exception as e:
                print(f"Error predicting item: {e}")
                results.append((0, 0.5, "low"))  # Default values on error
        
        return results
    
    def _get_confidence_level(self, probability: float) -> str:
        """Convert probability to confidence level"""
        if probability >= 0.8 or probability <= 0.2:
            return "high"
        elif probability >= 0.7 or probability <= 0.3:
            return "medium"
        else:
            return "low"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.loaded:
            return {"status": "Model not loaded"}
        
        info = {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_names),
            "loaded": self.loaded,
            "supports_probability": hasattr(self.model, 'predict_proba')
        }
        
        # Add model-specific information
        if hasattr(self.model, 'feature_importances_'):
            info["has_feature_importance"] = True
        else:
            info["has_feature_importance"] = False
        
        return info

# Create global inference instance
inference_model = ModelInference()