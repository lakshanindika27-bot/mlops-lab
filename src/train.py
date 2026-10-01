import argparse

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--max-features", type=int, default=20000)
parser.add_argument("--C", type=float, default=1.0)
args = parser.parse_args()

CATEGORIES = ["sci.space", "rec.autos", "talk.politics.misc"]
REMOVE = ("headers", "footers", "quotes")

train = fetch_20newsgroups(subset="train", categories=CATEGORIES, remove=REMOVE)
test = fetch_20newsgroups(subset="test", categories=CATEGORIES, remove=REMOVE)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("newsgroups-classifier")

with mlflow.start_run():
    mlflow.log_params({"max_features": args.max_features, "C": args.C})

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=args.max_features, stop_words="english")),
        ("clf", LogisticRegression(C=args.C, max_iter=1000)),
    ])
    model.fit(train.data, train.target)

    preds = model.predict(test.data)
    acc = accuracy_score(test.target, preds)
    f1 = f1_score(test.target, preds, average="macro")
    mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
    mlflow.sklearn.log_model(model, name="model")

    print(f"accuracy: {acc:.3f}  f1_macro: {f1:.3f}")

joblib.dump({"model": model, "labels": train.target_names}, "models/model.joblib")
print("saved models/model.joblib")
