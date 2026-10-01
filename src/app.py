import os

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
labels = artifact["labels"]

app = FastAPI(title="MLOps Lab Classifier")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    probs = model.predict_proba([req.text])[0]
    idx = int(probs.argmax())
    return PredictResponse(label=labels[idx], confidence=float(probs[idx]))
