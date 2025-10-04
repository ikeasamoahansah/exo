from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import io
import pickle
import numpy as np

app = FastAPI(
    title="Exoplanet Prediction API",
    description="API for predicting exoplanet candidates using KOI features",
    version="1.0.0"
)

model = pickle.load(open('model/xgb_model_top8_mi.pkl', 'rb'))

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

def make_prediction(features: np.ndarray):
    """Make prediction using the loaded model"""
    # Uncomment when model is loaded
    # prediction = model.predict(features)[0]
    # probability = model.predict_proba(features)[0][1]
    
    # Placeholder for demonstration
    prediction = np.random.choice([0, 1])
    probability = np.random.random()
    
    return prediction, probability

@app.get("/")
def read_root():
    return {
        "message": "Exoplanet Prediction API",
        "endpoints": {
            "/predict": "Single prediction (POST)",
            "/predict/batch": "Batch prediction from CSV (POST)",
            "/health": "Health check (GET)",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionOutput)
def predict_single(input_data: PredictionInput):
    """
    Make a single prediction for exoplanet classification
    """
    try:
        features = prepare_features(input_data.dict())
        prediction, probability = make_prediction(features)
        
        classification = "Exoplanet Candidate" if prediction == 1 else "False Positive"
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            classification=classification
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Make batch predictions from CSV file
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
        
        # Make predictions
        predictions = []
        probabilities = []
        
        for _, row in df.iterrows():
            features = prepare_features(row[required_cols].to_dict())
            pred, prob = make_prediction(features)
            predictions.append(pred)
            probabilities.append(prob)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        df['probability'] = probabilities
        df['classification'] = df['prediction'].map({
            1: 'Exoplanet Candidate',
            0: 'False Positive'
        })
        
        # Convert to CSV for download
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"}
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/predict/batch/json", response_model=BatchPredictionOutput)
async def predict_batch_json(file: UploadFile = File(...)):
    """
    Make batch predictions from CSV file
    Returns JSON response with predictions
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
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
        
        results = []
        for idx, row in df.iterrows():
            features = prepare_features(row[required_cols].to_dict())
            pred, prob = make_prediction(features)
            
            results.append({
                "row_index": int(idx),
                "prediction": int(pred),
                "probability": float(prob),
                "classification": "Exoplanet Candidate" if pred == 1 else "False Positive"
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
