"""FastAPI backend app"""

from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras  # pylint: disable=no-name-in-module

from backend.app.model import train_model as tm

app = FastAPI(title="MassBank Spectrum Predictor")

MODEL_PATH = "backend/app/model/spectrum_predictor.keras"
model = keras.models.load_model(MODEL_PATH)


class PredictBatchRequest(BaseModel):
    """Request for batch predict"""

    smiles_list: List[str]


class PredictBatchResponse(BaseModel):
    """Response for batch predict"""

    pred_vectors: List[List[float]]


class PredictRequest(BaseModel):
    """Request for predict"""

    smiles: str


class PredictResponse(BaseModel):
    """Response for predict"""

    pred_vector: list[float]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict spectrum based on smiles"""
    if not req.smiles.strip():
        return {"pred_vector": []}

    fp = tm.smiles_to_fp(req.smiles)
    if fp is None:
        return {"pred_vector": []}

    fp = np.expand_dims(fp, axis=0)
    pred = model.predict(fp)
    return {"pred_vector": pred[0].tolist()}


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    """Predict spectrums based on smiles batch"""
    preds: List = []
    for smi in req.smiles_list:
        if not smi.strip():
            preds.append([])
            continue
        fp = tm.smiles_to_fp(smi)
        if fp is None:
            preds.append([])
            continue
        fp = np.expand_dims(fp, axis=0)
        pred = model.predict(fp)
        preds.append(pred[0].tolist())
    return {"pred_vectors": preds}
