import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

REGISTERED_MODEL_NAME = "newsgroups-classifier"
CATEGORIES = ["sci.space", "rec.autos", "talk.politics.misc"]
REMOVE = ("headers", "footers", "quotes")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--C", type=float, default=1.0)
    return parser.parse_args(argv)


def load_data():
    train = fetch_20newsgroups(subset="train", categories=CATEGORIES, remove=REMOVE)
    test = fetch_20newsgroups(subset="test", categories=CATEGORIES, remove=REMOVE)
    names = list(train.target_names)
    y_train = [names[i] for i in train.target]
    y_test = [names[i] for i in test.target]
    return train.data, y_train, test.data, y_test, names


def build_model(max_features, C):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, stop_words="english")),
        ("clf", LogisticRegression(C=C, max_iter=1000)),
    ])


def evaluate(model, X, y):
    preds = model.predict(X)
    return accuracy_score(y, preds), f1_score(y, preds, average="macro")


def save_bundle(model, names, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump({"model": model, "labels": names}, path)


def main(argv=None, model_path="models/model.joblib"):
    args = parse_args(argv)
    X_train, y_train, X_test, y_test, names = load_data()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("newsgroups-classifier")

    with mlflow.start_run():
        mlflow.log_params({"max_features": args.max_features, "C": args.C})
        model = build_model(args.max_features, args.C)
        model.fit(X_train, y_train)

        acc, f1 = evaluate(model, X_test, y_test)
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
        mlflow.sklearn.log_model(
            model, name="model", registered_model_name=REGISTERED_MODEL_NAME
        )
        print(f"accuracy: {acc:.3f}  f1_macro: {f1:.3f}")

    save_bundle(model, names, model_path)
    print(f"saved {model_path}")
    return acc, f1


if __name__ == "__main__":
    main()
