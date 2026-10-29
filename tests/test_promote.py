import runpy
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SCRIPT = "src/promote.py"


def make_client(accuracies):
    """Fake MlflowClient: one model version per accuracy value (versions 1..n)."""
    client = MagicMock()
    versions = [
        SimpleNamespace(version=str(i + 1), run_id=f"run{i + 1}")
        for i in range(len(accuracies))
    ]
    client.search_model_versions.return_value = versions

    def get_run(run_id):
        acc = accuracies[int(run_id.removeprefix("run")) - 1]
        metrics = {} if acc is None else {"accuracy": acc}
        return SimpleNamespace(data=SimpleNamespace(metrics=metrics))

    client.get_run.side_effect = get_run
    return client


@pytest.fixture
def run_promote(monkeypatch):
    def _run(accuracies, argv=None):
        client = make_client(accuracies)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///unused.db")
        monkeypatch.setattr("mlflow.set_tracking_uri", lambda uri: None)
        monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: client)
        monkeypatch.setattr(sys, "argv", ["promote.py"] + (argv or []))
        return client

    return _run


def test_auto_promotes_best_accuracy(run_promote, capsys):
    client = run_promote([0.80, 0.91, 0.85])
    runpy.run_path(SCRIPT, run_name="__main__")
    client.set_registered_model_alias.assert_called_once_with(
        "newsgroups-classifier", "production", "2"
    )
    assert "version 2" in capsys.readouterr().out


def test_manual_version_overrides_search(run_promote, capsys):
    client = run_promote([0.80, 0.91], argv=["1"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(SCRIPT, run_name="__main__")
    assert exc.value.code == 0
    client.set_registered_model_alias.assert_called_once_with(
        "newsgroups-classifier", "production", "1"
    )
    client.search_model_versions.assert_not_called()
    assert "(manual)" in capsys.readouterr().out


def test_version_without_accuracy_is_ignored(run_promote):
    client = run_promote([None, 0.70])
    runpy.run_path(SCRIPT, run_name="__main__")
    client.set_registered_model_alias.assert_called_once_with(
        "newsgroups-classifier", "production", "2"
    )


def test_missing_tracking_uri_fails(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(sys, "argv", ["promote.py"])
    with pytest.raises(KeyError):
        runpy.run_path(SCRIPT, run_name="__main__")
