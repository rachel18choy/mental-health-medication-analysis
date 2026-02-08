from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
import os  # <-- THIS WAS MISSING!

# Import schemas
from app.schemas import (
    HealthCheck, DataResponse, 
    PredictionInput, PredictionOutput,
    BatchPredictionInput, BatchPredictionOutput
)

# Import your original data loader
from app.data_loader import load_data

# Create FastAPI app
app = FastAPI(
    title="Psychosis First Aid API",
    description="API for serving survey data and making predictions",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
survey_data = None
inference_model = None
scaler = None
feature_names = []

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load data and models when the API starts"""
    global survey_data, inference_model, scaler, feature_names
    
    print("="*50)
    print("Starting Psychosis First Aid API...")
    print("="*50)
    
    # 1. Load survey data
    try:
        survey_data = load_data()
        print(f"✓ Survey data loaded: {len(survey_data)} records")
    except Exception as e:
        print(f"✗ Error loading survey data: {e}")
        survey_data = pd.DataFrame()
    
    # 2. Try to load inference model
    try:
        # Check models directory
        model_dir = "models/saved_models/supervised"
        if not os.path.exists(model_dir):
            print(f"⚠️  Models directory not found: {model_dir}")
            print("   Run training scripts to enable predictions")
            return
        
        # Try to load any model
        model_files = [
            "random_forest.pkl",
            "logistic_regression.pkl",
            "gradient_boosting.pkl",
            "support_vector_machine.pkl"
        ]
        
        loaded = False
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                try:
                    inference_model = joblib.load(model_path)
                    print(f"✓ Model loaded: {model_file.replace('.pkl', '')}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"⚠️  Could not load {model_file}: {e}")
        
        if not loaded:
            print("✗ No models could be loaded")
            print("  Run: python models/train_supervised.py")
            return
        
        # Try to load preprocessing artifacts
        try:
            scaler = joblib.load("models/saved_models/scaler.pkl")
            print("✓ Scaler loaded")
        except:
            print("⚠️  Scaler not found")
            scaler = None
        
        # Try to load feature names
        try:
            with open("models/saved_models/feature_names.txt", 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            print(f"✓ Feature names loaded ({len(feature_names)} features)")
        except:
            print("⚠️  Feature names not found")
            feature_names = []
            
    except Exception as e:
        print(f"✗ Error during model loading: {e}")

# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthCheck, tags=["Health"])
async def root():
    """Root endpoint - health check"""
    return {
        "status": "healthy",
        "data_loaded": survey_data is not None and not survey_data.empty,
        "model_loaded": inference_model is not None,
        "data_records": len(survey_data) if survey_data is not None else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return await root()

@app.get("/status", tags=["Health"])
async def status():
    """Detailed status endpoint"""
    model_info = {}
    if inference_model is not None:
        model_info = {
            "type": type(inference_model).__name__,
            "features": len(feature_names) if feature_names else 0,
            "has_scaler": scaler is not None
        }
    
    return {
        "api": "running",
        "data": {
            "loaded": survey_data is not None and not survey_data.empty,
            "records": len(survey_data) if survey_data is not None else 0,
            "columns": list(survey_data.columns) if survey_data is not None else []
        },
        "model": model_info,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# DATA ENDPOINTS (Your Original API)
# ============================================================================

@app.get("/data", response_model=DataResponse, tags=["Data"])
async def get_all_data(
    limit: Optional[int] = Query(None, description="Limit number of records"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get all survey data with optional pagination"""
    if survey_data is None or survey_data.empty:
        raise HTTPException(status_code=503, detail="Survey data not loaded")
    
    try:
        # Apply pagination if requested
        if limit is not None:
            end_idx = min(offset + limit, len(survey_data))
            data_slice = survey_data.iloc[offset:end_idx]
        else:
            data_slice = survey_data
        
        # Convert to dictionary
        data_dict = data_slice.to_dict(orient="records")
        
        return {
            "data": data_dict,
            "total_records": len(survey_data),
            "returned_records": len(data_dict),
            "offset": offset,
            "limit": limit  # Return None instead of 'none'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.get("/data/stats", tags=["Data"])
async def get_data_statistics():
    """Get statistics about the survey data"""
    if survey_data is None or survey_data.empty:
        raise HTTPException(status_code=503, detail="Survey data not loaded")
    
    try:
        stats = {
            "total_records": len(survey_data),
            "total_columns": len(survey_data.columns),
            "columns": list(survey_data.columns),
            "missing_values": int(survey_data.isnull().sum().sum())
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = survey_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["numeric_columns"] = list(numeric_cols)
            stats["numeric_stats"] = survey_data[numeric_cols].describe().to_dict()
        
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating statistics: {str(e)}")

# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

def check_model_loaded():
    """Check if model is available"""
    if inference_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first."
        )

@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    check_model_loaded()
    
    return {
        "model_type": type(inference_model).__name__,
        "feature_count": len(feature_names) if feature_names else 0,
        "scaler_loaded": scaler is not None,
        "supports_probability": hasattr(inference_model, 'predict_proba')
    }

def preprocess_input(input_data: Dict[str, Any]) -> np.ndarray:
    """Preprocess input for prediction with feature mapping"""
    from app.feature_mapper import map_api_to_model, get_expected_feature_order
    
    # Map API field names to model field names
    model_input = map_api_to_model(input_data)
    
    # Convert to DataFrame
    df = pd.DataFrame([model_input])
    
    # Get the expected feature order from training
    expected_features = get_expected_feature_order()
    
    # Reorder columns to match training order
    # Keep only columns that exist in both
    available_cols = [col for col in expected_features if col in df.columns]
    
    # Create a new DataFrame with correct column order
    df_ordered = pd.DataFrame(columns=expected_features)
    
    # Fill with values from input
    for col in expected_features:
        if col in df.columns:
            df_ordered[col] = df[col]
        else:
            # Fill missing columns with 0 (or appropriate default)
            df_ordered[col] = 0
    
    # Convert all to numeric
    for col in df_ordered.columns:
        df_ordered[col] = pd.to_numeric(df_ordered[col], errors='coerce')
    
    # Fill missing values
    df_ordered = df_ordered.fillna(df_ordered.median(numeric_only=True))
    
    # Scale if scaler available
    if scaler is not None:
        return scaler.transform(df_ordered)
    else:
        return df_ordered.values

@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict(input_data: PredictionInput):
    """
    Make a prediction for a single respondent
    
    Predicts whether antipsychotics are helpful (1) or not (0)
    """
    check_model_loaded()
    
    try:
        # Convert to dict
        input_dict = input_data.dict()
        
        # Preprocess
        processed_input = preprocess_input(input_dict)
        
        # Make prediction
        prediction = inference_model.predict(processed_input)[0]
        
        # Get probability
        if hasattr(inference_model, 'predict_proba'):
            probability = inference_model.predict_proba(processed_input)[0][1]
        else:
            probability = 0.5
        
        # Confidence level
        if probability >= 0.8 or probability <= 0.2:
            confidence = "high"
        elif probability >= 0.7 or probability <= 0.3:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "model_used": type(inference_model).__name__,
            "confidence": confidence
        }
    
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Predictions"])
async def predict_batch(batch_input: BatchPredictionInput):
    """
    Make predictions for multiple respondents
    """
    check_model_loaded()
    
    try:
        results = []
        helpful_count = 0
        
        for item in batch_input.items:
            try:
                # Make prediction for each item
                input_dict = item.dict()
                processed_input = preprocess_input(input_dict)
                prediction = inference_model.predict(processed_input)[0]
                
                # Get probability
                if hasattr(inference_model, 'predict_proba'):
                    probability = inference_model.predict_proba(processed_input)[0][1]
                else:
                    probability = 0.5
                
                # Confidence
                if probability >= 0.8 or probability <= 0.2:
                    confidence = "high"
                elif probability >= 0.7 or probability <= 0.3:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                results.append({
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "model_used": type(inference_model).__name__,
                    "confidence": confidence
                })
                
                if prediction == 1:
                    helpful_count += 1
                    
            except Exception as e:
                # Default prediction on error
                results.append({
                    "prediction": 0,
                    "probability": 0.5,
                    "model_used": type(inference_model).__name__,
                    "confidence": "low"
                })
        
        total = len(results)
        helpful_percentage = (helpful_count / total * 100) if total > 0 else 0
        
        return {
            "predictions": results,
            "total_items": total,
            "helpful_count": helpful_count,
            "not_helpful_count": total - helpful_count,
            "helpful_percentage": helpful_percentage
        }
    
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Batch prediction failed: {str(e)}")

@app.get("/predict/example", tags=["Predictions"])
async def get_prediction_example():
    """Get example input for testing predictions"""
    # Return exactly 30 fields (no extra fields)
    return {
        "age": 25,
        "gender": "Woman",
        "study_area": "Psychology",
        "study_years": "3-4 years",
        "recognize_psychosis": 1,
        "professional_help_helpful": 1,
        "john_could_snap_out": 2,
        "john_weakness": 2,
        "john_not_real_illness": 2,
        "john_dangerous": 2,
        "avoid_john": 2,
        "john_unpredictable": 2,
        "not_tell_anyone": 2,
        "go_out_weekend": 4,
        "work_on_project": 4,
        "invite_to_house": 4,
        "go_to_johns_house": 4,
        "develop_friendship": 4,
        "ask_harm_thoughts": 5,
        "listen_restate": 5,
        "convey_hope": 5,
        "discuss_professional_options": 5,
        "ask_supportive_people": 5,
        "ask_suicide_thoughts": 5,
        "ask_suicide_plan": 5,
        "encourage_professional_help": 5,
        "acknowledge_frightened": 5,
        "convince_false_beliefs": 1,
        "listen_unreal_experiences": 4,
        "find_reasons_no_help": 5
    }
    
    # Only return the first 30 fields (to match model)
    # Remove the last 4 extra fields
    keys_to_keep = list(example.keys())[:30]
    return {k: example[k] for k in keys_to_keep}

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/endpoints", tags=["Utility"])
async def list_endpoints():
    """List all available API endpoints"""
    endpoints = []
    
    for route in app.routes:
        if hasattr(route, "methods"):
            endpoints.append({
                "path": route.path,
                "methods": list(route.methods),
                "tags": route.tags if hasattr(route, "tags") else []
            })
    
    return {
        "endpoints": endpoints,
        "total": len(endpoints)
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )