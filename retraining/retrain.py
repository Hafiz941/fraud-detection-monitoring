"""
Retraining Script (Day 2)
------------------------
- Loads versioned, processed training data
- Trains a candidate fraud detection model
- Saves the candidate model safely
- Stores retraining metadata for traceability
"""
import sys
import os
import json
import pandas as pd
import joblib
from datetime import datetime
from evaluation.evaluate import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef
# from sklearn.metrics import accuracy_score, f1_score

# -------------------
# Configuration
# -------------------
DATA_VERSION = "v1"
DATA_PATH = "data/processed/creditcard_v1_processed.csv"
RANDOM_STATE = 42

MODE = sys.argv[1] if len(sys.argv) > 1 else "candidate"

MODEL_PATH = (
    "model/model.pkl"
    if MODE == "production"
    else "model/candidate_model.pkl"
)

MODEL_METADATA_PATH = (
    "model/model_metadata.json"
    if MODE == "production"
    else "model/candidate_model_metadata.json"
)

METRICS_PATH = (
    "model/model_metrics.json"
    if MODE == "production"
    else "model/candidate_model_metrics.json"
)

# Versioning
VERSION_ID = datetime.utcnow().strftime("v%Y%m%d")
MODEL_REGISTRY_DIR = f"model/registry/{VERSION_ID}"


# -------------------
# Retraining Function
# -------------------
def retrain():
    print("Starting retraining pipeline...")
    print(f"Model version: {VERSION_ID}")

# Create registry directory
    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)

    # 1. Load processed data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")
    
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 2. Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Train candidate model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # 4. Evaluate candidate model
    y_pred = model.predict(X_val)

    metrics = evaluate_model(
        y_true=y_val,
        y_pred=y_pred,
        output_path=METRICS_PATH
    )
    
    print("Candidate model evaluation metrics:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v}")
    
     # 5. Save versioned artifacts
    versioned_model_path = f"{MODEL_REGISTRY_DIR}/model.pkl"
    versioned_metrics_path = f"{MODEL_REGISTRY_DIR}/metrics.json"
    versioned_metadata_path = f"{MODEL_REGISTRY_DIR}/metadata.json"

    joblib.dump(model, versioned_model_path)
    
    with open(versioned_metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # 6. Save versioned metadata
    metadata = {
        "model_version": VERSION_ID,
        "model_type": "LogisticRegression",
        "trained_on": datetime.utcnow().isoformat(),
        "dataset_version": DATA_VERSION,
        "validation_split": 0.2,
        "model_file": versioned_model_path,
        "metrics_file": versioned_metrics_path,
        "mode": MODE,
        "approved": MODE == "production"
    }
    
    with open(versioned_metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    # 5. Save candidate model
    joblib.dump(model, MODEL_PATH)
    print(f"Candidate model saved to {MODEL_PATH}")


    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Versioned model saved to {MODEL_REGISTRY_DIR}")
    print("Retraining completed successfully.")

# -------------------
# Entry point
# -------------------
if __name__ == "__main__":
    retrain()
