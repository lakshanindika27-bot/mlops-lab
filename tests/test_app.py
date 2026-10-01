import os

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LABELS = ["sci.space", "rec.autos", "talk.politics.misc"]
TEXTS = [
    "nasa rocket moon orbit satellite launch",
    "space shuttle astronaut mars telescope",
    "car engine brake wheel transmission",
    "truck tires oil change motor garage",
    "senate vote election policy president",
    "government law tax congress debate",
]
TARGETS = [0, 0, 1, 1, 2, 2]


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(TEXTS, TARGETS)
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump({"model": model, "labels": LABELS}, path)
    os.environ["MODEL_PATH"] = str(path)

    from src.app import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_valid_label(client):
    r = client.post("/predict", json={"text": "nasa rocket launch to the moon"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in LABELS
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_missing_text_is_rejected(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422
EOpython -m pytest -v
ruff check .
mkdir -p .github/workflows
cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
          cache: pip

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest httpx ruff

      - name: Lint
        run: ruff check .

      - name: Test
        run: python -m pytest -v
