"""
Model Comparison & Promotion Logic
--------------------------------------------------
- Compares candidate and production models
- Uses stored evaluation metrics (no re-evaluation)
- Applies MCC-based promotion rules
- Logs explainable promotion decisions
"""

import json
import shutil
import os
from datetime import datetime

# -------------------
# Configuration
# -------------------
CANDIDATE_METRICS_PATH = "model/candidate_model_metrics.json"
PRODUCTION_METRICS_PATH = "model/model_metrics.json"

CANDIDATE_MODEL_PATH = "model/candidate_model.pkl"
PRODUCTION_MODEL_PATH = "model/model.pkl"

DECISION_PATH = "model/promotion_decision.json"

MCC_MARGIN = 0.005  # Safety margin

# -------------------
# Utilities
# -------------------
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

# -------------------
# Promotion Logic
# -------------------
def compare_models():
    print("Starting model comparison...")

    candidate_metrics = load_json(CANDIDATE_METRICS_PATH)
    production_metrics = load_json(PRODUCTION_METRICS_PATH)

    decision = {
        "evaluated_at": datetime.utcnow().isoformat(),
        "decision": None,
        "reason": "",
        "candidate_mcc": None,
        "production_mcc": None
    }

    # Case 1: No production model yet
    if production_metrics is None:
        print("No production model found. Promoting candidate.")

        shutil.copy(CANDIDATE_MODEL_PATH, PRODUCTION_MODEL_PATH)
        shutil.copy(CANDIDATE_METRICS_PATH, PRODUCTION_METRICS_PATH)

        decision.update({
            "decision": "PROMOTED",
            "reason": "No existing production model",
            "candidate_mcc": candidate_metrics["mcc"]
        })

    else:
        cand_mcc = candidate_metrics["mcc"]
        prod_mcc = production_metrics["mcc"]

        decision["candidate_mcc"] = cand_mcc
        decision["production_mcc"] = prod_mcc

        print(f"Candidate MCC: {cand_mcc}")
        print(f"Production MCC: {prod_mcc}")

        if cand_mcc > prod_mcc + MCC_MARGIN:
            print("Candidate outperforms production. Promoting model.")

            shutil.copy(CANDIDATE_MODEL_PATH, PRODUCTION_MODEL_PATH)
            shutil.copy(CANDIDATE_METRICS_PATH, PRODUCTION_METRICS_PATH)

            decision.update({
                "decision": "PROMOTED",
                "reason": "Candidate MCC significantly higher than production"
            })

        else:
            print("Candidate does not outperform production. Rejecting.")

            decision.update({
                "decision": "REJECTED",
                "reason": "Candidate MCC not sufficiently better than production"
            })

    # Save decision
    with open(DECISION_PATH, "w") as f:
        json.dump(decision, f, indent=4)

    print(f"Promotion decision saved to {DECISION_PATH}")
    print("Comparison completed.")

# -------------------
# Entry Point
# -------------------
if __name__ == "__main__":
    compare_models()
