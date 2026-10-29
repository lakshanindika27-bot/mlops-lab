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
parser.add_argument("--push", action="store_true")
parser.add_argument("--gateway", default="localhost:9091")
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

if args.push:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    result = snapshot.dict()
    registry = CollectorRegistry()
    share = Gauge("drift_share", "Share of drifted columns", registry=registry)
    count = Gauge("drift_columns", "Number of drifted columns", registry=registry)
    pval = Gauge("drift_p_value", "Per-column drift p-value", ["column"], registry=registry)
    flag = Gauge("drift_detected", "1 if column drifted", ["column"], registry=registry)

    for m in result["metrics"]:
        name = m["metric_name"]
        if name.startswith("DriftedColumnsCount"):
            count.set(m["value"]["count"])
            share.set(m["value"]["share"])
        elif name.startswith("ValueDrift"):
            col = m["config"]["column"]
            pval.labels(column=col).set(m["value"])
            flag.labels(column=col).set(1 if m["value"] < m["config"]["threshold"] else 0)

    push_to_gateway(args.gateway, job="drift_check", grouping_key={"scenario": args.scenario}, registry=registry)
    print(f"pushed to {args.gateway} (scenario={args.scenario})")
