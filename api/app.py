from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / 'models' / 'attrition_platform_model.joblib')
EXPECTED = list(MODEL.feature_names_in_)
app = FastAPI(title='Employee Attrition Scoring API', version='1.0.0')

class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(description='One employee record using the training column names')

class PredictionResponse(BaseModel):
    attrition_probability: float
    predicted_attrition: bool
    risk_category: str

@app.get('/health')
def health(): return {'status': 'ok', 'model_loaded': True}

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    missing = sorted(set(EXPECTED) - set(request.features))
    if missing: raise HTTPException(status_code=422, detail={'missing_features': missing})
    frame = pd.DataFrame([{name: request.features[name] for name in EXPECTED}])
    probability = float(MODEL.predict_proba(frame)[0, 1])
    category = 'High' if probability >= .66 else 'Medium' if probability >= .33 else 'Low'
    return PredictionResponse(attrition_probability=round(probability, 6), predicted_attrition=probability >= .5, risk_category=category)
