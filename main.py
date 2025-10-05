from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import io
import pickle
import joblib
import numpy as np
from enum import Enum

app = FastAPI(
    title="Exoplanet Prediction API",
    description="API for predicting exoplanet candidates using KOI features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define available models
class ModelType(str, Enum):
    random_forest = "random_forest"
    xgboost = "xgboost"
    ensemble = "ensemble"
    # logistic_regression = "logistic_regression"
    # svm = "svm"
    # neural_network = "neural_network"
    # gradient_boosting = "gradient_boosting"

# Dictionary to store loaded models
models = {}

def load_models():
    """Load all available models on startup"""
    # models['random_forest'] = pickle.load(open('models/xgb_rf.pkl', 'rb'))
    models['xgboost'] = joblib.load('models/xgboost.pkl')
    models['ensemble'] = joblib.load('models/xgb_rf.pkl')
    # models['logistic_regression'] = pickle.load(open('models/logistic_regression.pkl', 'rb'))
    # models['svm'] = pickle.load(open('models/svm.pkl', 'rb'))
    # models['neural_network'] = pickle.load(open('models/neural_network.pkl', 'rb'))
    # models['gradient_boosting'] = pickle.load(open('models/gradient_boosting.pkl', 'rb'))

# Load models when app starts
@app.on_event("startup")
async def startup_event():
    load_models()
    print("Models loaded successfully")

class PredictionInput(BaseModel):
    koi_model_snr: float = Field(..., description="Transit signal-to-noise ratio")
    koi_prad: float = Field(..., description="Planetary radius in Earth radii")
    koi_fpflag_ss: int = Field(..., ge=0, le=1, description="Stellar eclipse flag")
    koi_fpflag_co: int = Field(..., ge=0, le=1, description="Centroid offset flag")
    koi_period: float = Field(..., description="Orbital period in days")
    koi_depth: float = Field(..., description="Transit depth in parts per million")
    koi_fpflag_nt: int = Field(..., ge=0, le=1, description="Not transit-like flag")
    koi_insol: float = Field(..., description="Insolation flux in Earth units")

    class Config:
        json_schema_extra = {
            "example": {
                "koi_model_snr": 15.5,
                "koi_prad": 2.3,
                "koi_fpflag_ss": 0,
                "koi_fpflag_co": 0,
                "koi_period": 10.5,
                "koi_depth": 500.0,
                "koi_fpflag_nt": 0,
                "koi_insol": 1.2
            }
        }

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    classification: str

class BatchPredictionOutput(BaseModel):
    predictions: List[dict]
    total_processed: int

def prepare_features(data: dict) -> np.ndarray:
    """Convert input dictionary to feature array in correct order"""
    feature_order = [
        'koi_model_snr', 'koi_prad', 'koi_fpflag_ss', 'koi_fpflag_co',
        'koi_period', 'koi_depth', 'koi_fpflag_nt', 'koi_insol'
    ]
    return np.array([[data[f] for f in feature_order]])

def preprocess(data):
    for col in data.columns:
        if data[col].isnull().sum() > 0 and data[col].dtype != 'O':
            data.fillna({col: data[col].fillna(0)}, inplace=True)
    return data

def make_prediction(features: np.ndarray, model_name: str):
    """Make prediction using the selected model"""
    if model_name in models:
        model = models[model_name]
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else 0.5
    else:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not found")
    
    # Placeholder for demonstration - replace with actual model prediction
    prediction = np.random.choice([0, 1, 2])  # 0=false positive, 1=confirmed, 2=candidate
    probability = np.random.random()
    
    return prediction, probability

@app.get("/")
def read_root():
    return {
        "message": "Exoplanet Prediction API",
        "available_models": [model.value for model in ModelType],
        "endpoints": {
            "/predict": "Single prediction (POST)",
            "/predict/batch": "Batch prediction from CSV (POST)",
            "/models": "List available models (GET)",
            "/health": "Health check (GET)",
            "/docs": "API documentation"
        }
    }

@app.get("/models")
def list_models():
    """List all available models"""
    return {
        "available_models": [model.value for model in ModelType],
        "loaded_models": list(models.keys()) if models else []
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "models_loaded": len(models),
        "available_models": [model.value for model in ModelType]
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_single(
    input_data: PredictionInput,
    model: ModelType = Query(ModelType.ensemble, description="Model to use for prediction")
):
    """
    Make a single prediction for exoplanet classification using the specified model
    """
    try:
        features = prepare_features(input_data.dict())
        prediction, probability = make_prediction(features, model.value)
        
        if prediction == 1:
            classification = "Confirmed Exoplanet"
        elif prediction == 2:
            classification = "Exoplanet Candidate"
        else:
            classification = "False Positive"
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            classification=classification
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
    model: ModelType = Query(ModelType.ensemble, description="Model to use for predictions")
):
    """
    Make batch predictions from CSV file using the specified model
    Returns a CSV file with predictions
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate required columns
        required_cols = [
            'koi_model_snr', 'koi_prad', 'koi_fpflag_ss', 'koi_fpflag_co',
            'koi_period', 'koi_depth', 'koi_fpflag_nt', 'koi_insol'
        ]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        cleaned_df = preprocess(df)

        # Make predictions
        predictions = []
        probabilities = []
        
        for _, row in cleaned_df.iterrows():
            features = prepare_features(row[required_cols].to_dict())
            pred, prob = make_prediction(features, model.value)
            predictions.append(pred)
            probabilities.append(prob)
        
        # Add predictions to dataframe
        cleaned_df['prediction'] = predictions
        cleaned_df['probability'] = probabilities
        cleaned_df['classification'] = cleaned_df['prediction'].map({
            1: 'Confirmed Exoplanet',
            2: 'Exoplanet Candidate',
            0: 'False Positive'
        })
        
        # Convert to CSV for download
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{model.value}_{file.filename}"}
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/predict/batch/json", response_model=BatchPredictionOutput)
async def predict_batch_json(
    file: UploadFile = File(...),
    model: ModelType = Query(ModelType.ensemble, description="Model to use for predictions")
):
    """
    Make batch predictions from CSV file using the specified model
    Returns JSON response with predictions
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        print("file received and read")

        required_cols = [
            'koi_model_snr', 'koi_prad', 'koi_fpflag_ss', 'koi_fpflag_co',
            'koi_period', 'koi_depth', 'koi_fpflag_nt', 'koi_insol'
        ]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        cleaned_df = preprocess(df)

        print("data cleaned")

        results = []
        for idx, row in cleaned_df.iterrows():
            features = prepare_features(row[required_cols].to_dict())
            pred, prob = make_prediction(features, model.value)
            
            if pred == 1:
                classification = "Confirmed Exoplanet"
            elif pred == 2:
                classification = "Exoplanet Candidate"
            else:
                classification = "False Positive"
            
            results.append({
                "row_index": int(idx),
                "prediction": int(pred),
                "probability": float(prob),
                "classification": classification
            })
        
        return BatchPredictionOutput(
            predictions=results,
            total_processed=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)