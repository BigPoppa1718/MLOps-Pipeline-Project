import json
import os
import yaml
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


def evaluate_and_save_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, config: dict
):
    """Calculates evaluation metrics and exports them to a JSON file."""
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_dict["roc_auc_score"] = float(roc_auc_score(y_true, y_proba))

    metrics_dir = config["paths"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_file_path = os.path.join(metrics_dir, config["paths"]["metrics_name"])
    with open(metrics_file_path, "w") as json_file:
        json.dump(report_dict, json_file, indent=4)

    print(f"Metrics written successfully to {metrics_file_path}")
