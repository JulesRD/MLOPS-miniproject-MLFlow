import os
import logging

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import uvicorn
from typing import List, Any

app = FastAPI(title="ML Model Serving API")

# Global model
MODEL_NAME = os.getenv("MODEL_NAME", None)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)

if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable not set")
if not MLFLOW_TRACKING_URI:
    raise ValueError("MLFLOW_TRACKING_URI environment variable not set")
if not MODEL_VERSION:
    raise ValueError("MODEL_VERSION environment variable not set")

current_model_uri = f"{MLFLOW_TRACKING_URI}/models:/{MODEL_NAME}/{MODEL_VERSION}"  # default model URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # Set your MLflow tracking URI here

class PredictRequest(BaseModel):
    data: List[Any]


class UpdateModelRequest(BaseModel):
    model_uri: str


@app.on_event("startup")
def load_initial_model():
    """
    Load the model from MLflow when the service starts.
    """
    logging.info(f"✅ Entered function load_initial_model")
    global model, current_model_uri
    default_uri = "models:/Iris-model/2"
    try:
        model = mlflow.pyfunc.load_model(default_uri)
        current_model_uri = default_uri
        print(f"Loaded initial model from {default_uri}")
        logging.info(f"✅ Loaded initial model from {default_uri}")
    except Exception as e:
        logging.error(f"❌ Failed to load model from MLflow ({default_uri}): {e}")
        raise RuntimeError(f"Failed to load model from MLflow: {str(e)}")


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Return predictions from the loaded model.
    """
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        data_array = np.array(request.data).reshape(1, -1)
        predictions = model.predict(data_array)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-model")
def update_model(request: UpdateModelRequest):
    """
    Update the model by loading a new version from MLflow.
    """
    global model, current_model_uri
    try:
        model = mlflow.pyfunc.load_model(request.model_uri)
        current_model_uri = request.model_uri
        return {"status": "success", "model_uri": current_model_uri}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
