# MLOps Lab

End-to-end MLOps pipeline for a text classifier (20 Newsgroups: sci.space, rec.autos, talk.politics.misc).

## Stack

- **Model**: scikit-learn (TF-IDF + Logistic Regression), accuracy ~0.86
- **Serving**: FastAPI (`/health`, `/predict`, `/metrics`)
- **Experiment tracking**: MLflow
- **Data/model versioning**: DVC
- **Containerization**: Docker
- **CI**: GitHub Actions (ruff + pytest)
- **Deployment**: Kubernetes (kind), 2 replicas with readiness/liveness probes
- **Monitoring**: Prometheus + Grafana (kube-prometheus-stack via Helm), custom metrics `predictions_total` and `predict_latency_seconds`

## Run locally

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python src/train.py
    uvicorn src.app:app --port 8000

## Docker

    docker build -t mlops-lab:v3 .
    docker run --rm -p 8001:8000 mlops-lab:v3

## Kubernetes

    kind create cluster --name mlops
    kind load docker-image mlops-lab:v3 --name mlops
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/servicemonitor.yaml

## Tests

    python -m pytest -v
    ruff check .
