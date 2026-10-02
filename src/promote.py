import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

NAME = "newsgroups-classifier"
ALIAS = "production"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

if len(sys.argv) > 1:
    client.set_registered_model_alias(NAME, ALIAS, sys.argv[1])
    print(f"alias '{ALIAS}' -> version {sys.argv[1]} (manual)")
    sys.exit(0)

best_version, best_acc = None, -1.0
for mv in client.search_model_versions(f"name='{NAME}'"):
    acc = client.get_run(mv.run_id).data.metrics.get("accuracy", -1.0)
    print(f"version {mv.version}: accuracy={acc:.3f}")
    if acc > best_acc:
        best_version, best_acc = mv.version, acc

client.set_registered_model_alias(NAME, ALIAS, best_version)
print(f"alias '{ALIAS}' -> version {best_version} (accuracy={best_acc:.3f})")
