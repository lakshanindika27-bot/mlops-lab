import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LABELS = ["rec.autos", "sci.space", "talk.politics.misc"]
TEXTS = [
    "car engine wheel drive",
    "automobile motor road",
    "rocket orbit nasa launch",
    "space shuttle moon planet",
    "senate vote election policy",
    "government law tax congress",
]
TARGETS = [0, 0, 1, 1, 2, 2]


@pytest.fixture(scope="module")
def saved_bundle(tmp_path_factory):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])
    model.fit(TEXTS, [LABELS[i] for i in TARGETS])
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump({"model": model, "labels": LABELS}, path)
    return joblib.load(path)


def test_bundle_has_expected_keys(saved_bundle):
    assert set(saved_bundle) == {"model", "labels"}


def test_predict_returns_known_label(saved_bundle):
    pred = saved_bundle["model"].predict(["nasa rocket launch"])[0]
    assert pred in saved_bundle["labels"]


def test_predict_proba_sums_to_one(saved_bundle):
    proba = saved_bundle["model"].predict_proba(["car engine"])[0]
    assert abs(proba.sum() - 1.0) < 1e-6
    assert len(proba) == len(saved_bundle["labels"])
