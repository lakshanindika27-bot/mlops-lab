![CI](https://github.com/lakshanindika27-bot/mlops-lab/actions/workflows/ci.yml/badge.svg)

# MLOps Lab

End-to-end MLOps pipeline for a text classifier (20 Newsgroups: sci.space, rec.autos, talk.politics.misc).

## Stack

- **Model**: scikit-learn (TF-IDF + Logistic Regression), accuracy ~0.86
- **Serving**: FastAPI (`/health`, `/predict`, `/metrics`)
- **Experiment tracking & model registry**: MLflow (DagsHub), `production` alias
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

    docker build -t mlops-lab:v5 .
    docker run --rm -p 8001:8000 mlops-lab:v5

## Kubernetes (kind)

    kind create cluster --name mlops-lab
    kind load docker-image mlops-lab:v5 --name mlops-lab

    # DagsHub credentials live in a Secret, never in git
    kubectl create secret generic mlflow-creds \
      --from-literal=MLFLOW_TRACKING_URI=https://dagshub.com/<user>/mlops-lab.mlflow \
      --from-literal=MLFLOW_TRACKING_USERNAME=<user> \
      --from-literal=MLFLOW_TRACKING_PASSWORD=<token>

    kubectl apply -f k8s/
    kubectl port-forward svc/mlops-api 8002:80
    curl -X POST localhost:8002/predict -H "Content-Type: application/json" \
      -d '{"text": "NASA launched a new satellite"}'

The Deployment (2 replicas, `/health` readiness and liveness probes) sets
`MODEL_URI=models:/newsgroups-classifier@production`, so each pod loads the
model from the MLflow registry at startup and logs
`Loaded model from registry: ...`.

`k8s-later/servicemonitor.yaml` needs the Prometheus Operator CRDs and is
kept aside until monitoring is installed.

## Tests

    python -m pytest -v
    ruff check .

## CI/CD

GitHub Actions runs on every push to `main`:

1. `ruff check` and `pytest`
2. `dvc pull` of the model from the DagsHub DVC remote (token stored as the `DAGSHUB_TOKEN` secret)
3. Docker image build
4. Smoke test of the running container (`/health` and `/predict`)

## Data versioning

The model file is tracked with DVC and stored on DagsHub storage (S3-compatible). To fetch it locally:

    dvc remote modify dagshub --local access_key_id YOUR_TOKEN
    dvc remote modify dagshub --local secret_access_key YOUR_TOKEN
    dvc pull

## Model registry

Training runs are logged to MLflow (hosted on DagsHub) and every trained model is registered as a new version of `newsgroups-classifier`. The version that should serve traffic carries the `production` alias.

    python src/train.py          # logs a run and registers a new model version
    python src/promote.py        # points the alias at the best version (by accuracy)
    python src/promote.py 3      # manual promotion / rollback to version 3
    MODEL_URI="models:/newsgroups-classifier@production" uvicorn src.app:app

Without `MODEL_URI` the API falls back to the local `models/model.joblib`.

## Rollback demo

The model is loaded once at pod startup, so after moving the alias the
pods must be restarted:

    python src/promote.py 2
    kubectl rollout restart deployment mlops-api
    kubectl rollout status deployment/mlops-api

Serving the same request (`"NASA launched a new satellite"`) from each
registry version:

| Alias points to | C / max_features | /predict confidence |
|---|---|---|
| v1 | 1.0 / 20000 | 0.817 |
| v2 | 10.0 / 20000 | 0.961 |
| v3 | 0.5 / 5000 | 0.732 |
