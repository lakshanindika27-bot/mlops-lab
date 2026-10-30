import joblib
import pytest

from src import train

LABELS = ["rec.autos", "sci.space", "talk.politics.misc"]
TEXTS = [
    "car engine wheel drive",
    "automobile motor road",
    "rocket orbit nasa launch",
    "space shuttle moon planet",
    "senate vote election policy",
    "government law tax congress",
]
Y = [LABELS[i] for i in [0, 0, 1, 1, 2, 2]]


def test_parse_args_defaults():
    args = train.parse_args([])
    assert args.max_features == 20000
    assert args.C == 1.0


def test_parse_args_override():
    args = train.parse_args(["--max-features", "100", "--C", "0.5"])
    assert args.max_features == 100
    assert args.C == 0.5


def test_build_model_uses_params():
    model = train.build_model(max_features=50, C=2.0)
    assert model.named_steps["tfidf"].max_features == 50
    assert model.named_steps["clf"].C == 2.0


def test_evaluate_scores_in_range():
    model = train.build_model(100, 1.0).fit(TEXTS, Y)
    acc, f1 = train.evaluate(model, TEXTS, Y)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_save_bundle_roundtrip(tmp_path):
    model = train.build_model(100, 1.0).fit(TEXTS, Y)
    path = tmp_path / "nested" / "model.joblib"
    train.save_bundle(model, LABELS, str(path))
    bundle = joblib.load(path)
    assert set(bundle) == {"model", "labels"}
    assert bundle["labels"] == LABELS


def test_main_end_to_end_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "load_data", lambda: (TEXTS, Y, TEXTS, Y, LABELS))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "model.joblib"

    acc, f1 = train.main(["--max-features", "50"], model_path=str(out))

    assert out.exists()
    assert joblib.load(out)["labels"] == LABELS
    assert acc > 0.5
