import os

import joblib
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

MODEL_URI = os.getenv("MODEL_URI")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

if MODEL_URI:
    import mlflow.sklearn

    model = mlflow.sklearn.load_model(MODEL_URI)
    labels = list(model.classes_)
    print(f"Loaded model from registry: {MODEL_URI}", flush=True)
else:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    labels = artifact["labels"]
    print(f"Loaded model from file: {MODEL_PATH}", flush=True)

PREDICTIONS = Counter("predictions_total", "Total predictions", ["label"])
LATENCY = Histogram("predict_latency_seconds", "Latency of /predict")

app = FastAPI(title="MLOps Lab Classifier")
app.mount("/metrics", make_asgi_app())


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
    with LATENCY.time():
        probs = model.predict_proba([req.text])[0]
        idx = int(probs.argmax())
    PREDICTIONS.labels(label=labels[idx]).inc()
    return PredictResponse(label=labels[idx], confidence=float(probs[idx]))
