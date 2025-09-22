import os
import logging
import random
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import uvicorn
from typing import List, Any

app = FastAPI(title="ML Model Serving API")

# Environment variables
MODEL_NAME = os.getenv("MODEL_NAME", None)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)
NEXT_MODEL_VERSION = os.getenv("NEXT_MODEL_VERSION", None)
P_CANARY = float(os.getenv("CANARY_PROB", 0.1))  # probability to use next model

if not MODEL_NAME or not MLFLOW_TRACKING_URI or not MODEL_VERSION or not NEXT_MODEL_VERSION:
    raise ValueError("MODEL_NAME, MLFLOW_TRACKING_URI, MODEL_VERSION and NEXT_MODEL_VERSION must be set")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global models
current_model = None
next_model = None
current_model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
next_model_uri = f"models:/{MODEL_NAME}/{NEXT_MODEL_VERSION}"

class PredictRequest(BaseModel):
    data: List[Any]

class UpdateModelRequest(BaseModel):
    model_uri: str

@app.on_event("startup")
def load_models():
    global current_model, next_model
    try:
        current_model = mlflow.pyfunc.load_model(current_model_uri)
        next_model = current_model  # initially, next model = current
        logging.info(f"✅ Loaded initial model from {current_model_uri}")
    except Exception as e:
        logging.error(f"❌ Failed to load model from MLflow ({current_model_uri}): {e}")
        raise RuntimeError(f"Failed to load model: {e}")

@app.post("/predict")
def predict(request: PredictRequest):
    global current_model, next_model
    if current_model is None or next_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    data_array = np.array(request.data).reshape(1, -1)

    # Choose which model to use based on canary probability
    if random.random() < P_CANARY:
        model_to_use = next_model
        model_name = "next_model"
    else:
        model_to_use = current_model
        model_name = "current_model"

    try:
        predictions = model_to_use.predict(data_array)
        return {"predictions": predictions.tolist(), "used_model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-model")
def update_model(request: UpdateModelRequest):
    global next_model, next_model_uri
    try:
        next_model = mlflow.pyfunc.load_model(request.model_uri)
        next_model_uri = request.model_uri
        logging.info(f"✅ Next model updated to {request.model_uri}")
        return {"status": "success", "next_model_uri": next_model_uri}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update next model: {e}")

@app.post("/accept-next-model")
def accept_next_model():
    global current_model, next_model, current_model_uri, next_model_uri
    if next_model is None:
        raise HTTPException(status_code=500, detail="Next model not loaded")
    current_model = next_model
    current_model_uri = next_model_uri
    logging.info(f"✅ Promoted next model to current: {current_model_uri}")
    return {"status": "success", "current_model_uri": current_model_uri}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)