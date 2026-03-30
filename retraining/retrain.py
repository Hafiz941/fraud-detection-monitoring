"""
Retraining Script
------------------------
- Loads versioned, processed training data
- Trains a candidate fraud detection model
- Evaluates performance
- Compares with production model (MCC-based)
- Promotes to production if better
- Stores versioned artifacts and metadata
"""

import sys
import os
import json
import pandas as pd
import joblib
from datetime import datetime, UTC
from evaluation.evaluate import evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from retraining.config import DATA_PATH, DATA_VERSION, RANDOM_STATE, CANDIDATE_MODEL_PATH, CANDIDATE_METRICS_PATH, CANDIDATE_METADATA_PATH, PRODUCTION_MODEL_PATH, PRODUCTION_METRICS_PATH, MODEL_REGISTRY_BASE, TEST_SIZE


MODE = sys.argv[1] if len(sys.argv) > 1 else "candidate"

# Unique version id (prevents overwrite)
VERSION_ID = datetime.now(UTC).strftime("v%Y%m%d_%H%M%S")
MODEL_REGISTRY_DIR = f"{MODEL_REGISTRY_BASE}/{VERSION_ID}"

# -------------------
# Utility
# -------------------
def load_production_metrics():
    if os.path.exists(PRODUCTION_METRICS_PATH):
        with open(PRODUCTION_METRICS_PATH, "r") as f:
            return json.load(f)
    return None


# -------------------
# Retraining Pipeline
# -------------------
def retrain():
    print("Starting retraining pipeline...")
    print(f"Model version: {VERSION_ID}")

    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)

    # 1 Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 2️ Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 3️ Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 4️ Evaluate
    y_pred = model.predict(X_val)

    metrics = evaluate_model(
        y_true=y_val,
        y_pred=y_pred,
        output_path=CANDIDATE_METRICS_PATH
    )

    # Save candidate metrics
    with open(CANDIDATE_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Candidate model evaluation metrics:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v}")

    # 5️ Save versioned artifacts
    versioned_model_path = f"{MODEL_REGISTRY_DIR}/model.pkl"
    versioned_metrics_path = f"{MODEL_REGISTRY_DIR}/metrics.json"
    versioned_metadata_path = f"{MODEL_REGISTRY_DIR}/metadata.json"

    joblib.dump(model, versioned_model_path)

    with open(versioned_metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    metadata = {
        "model_version": VERSION_ID,
        "model_type": "LogisticRegression",
        "trained_on": datetime.now(UTC).isoformat(),
        "dataset_version": DATA_VERSION,
        "validation_split": TEST_SIZE,
        "registry_path": MODEL_REGISTRY_DIR,
        "mode": MODE
    }

    with open(versioned_metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    # 6️ Save candidate artifacts
    joblib.dump(model, CANDIDATE_MODEL_PATH)
    
    # Save candidate metadata
    with open(CANDIDATE_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Versioned model saved to {MODEL_REGISTRY_DIR}")

    # -------------------
    # Promotion Logic
    # -------------------
    prod_metrics = load_production_metrics()
    promote = False

    if MODE == "production":
        print("Production mode forced. Promoting model.")
        promote = True

    elif prod_metrics is None:
        print("No production model found. Promoting candidate.")
        promote = True

    elif metrics.get("mcc", 0) > prod_metrics.get("mcc", 0):
        print("Candidate model outperforms production (MCC improved). Promoting.")
        promote = True

    else:
        print("Candidate model did not outperform production.")

    # 7️ Promote if approved
    if promote:
        joblib.dump(model, PRODUCTION_MODEL_PATH)

        with open(PRODUCTION_METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        print("Production model updated successfully.")
    else:
        print("Production model unchanged.")

    print("Retraining pipeline completed.")

# -------------------
# Entry point
# -------------------
if __name__ == "__main__":
    retrain()