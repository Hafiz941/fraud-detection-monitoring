# evaluation/evaluate.py

import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)
from datetime import datetime


def evaluate_model(y_true, y_pred, output_path=None):
    """
    Evaluate model predictions and return metrics.
    Optionally save results to JSON.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "evaluated_at": datetime.utcnow().isoformat()
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics
