import argparse
import os

import joblib
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn.datasets import fetch_20newsgroups

CATEGORIES = ["sci.space", "rec.autos", "talk.politics.misc"]
DRIFT_CATEGORIES = ["comp.graphics", "sci.med", "rec.sport.hockey"]
REMOVE = ("headers", "footers", "quotes")

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", choices=["none", "drift"], default="none")
args = parser.parse_args()

bundle = joblib.load("models/model.joblib")
model = bundle["model"]


def featurize(texts):
    proba = model.predict_proba(texts)
    return pd.DataFrame({
        "n_chars": [len(t) for t in texts],
        "n_words": [len(t.split()) for t in texts],
        "confidence": proba.max(axis=1),
        "prediction": model.predict(texts),
    })


# Reference = test data එකෙන් half එකක් (model එක නොදුටු data)
ref_raw = fetch_20newsgroups(subset="test", categories=CATEGORIES, remove=REMOVE)
all_ref = pd.Series(ref_raw.data).sample(frac=1, random_state=0).tolist()
half = len(all_ref) // 2
reference = featurize(all_ref[:half])

# Current = "production" data (simulate කරන්නේ)
if args.scenario == "none":
    cur_texts = all_ref[half:]
else:
    cur_raw = fetch_20newsgroups(subset="test", categories=DRIFT_CATEGORIES, remove=REMOVE)
    cur_texts = pd.Series(cur_raw.data).sample(500, random_state=42).tolist()

current = featurize(cur_texts)

definition = DataDefinition(
    numerical_columns=["n_chars", "n_words", "confidence"],
    categorical_columns=["prediction"],
)
ref_ds = Dataset.from_pandas(reference, data_definition=definition)
cur_ds = Dataset.from_pandas(current, data_definition=definition)

report = Report([DataDriftPreset()])
snapshot = report.run(cur_ds, ref_ds)

os.makedirs("reports", exist_ok=True)
out = f"reports/drift_{args.scenario}.html"
snapshot.save_html(out)
print(f"saved {out}")
