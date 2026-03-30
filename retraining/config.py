# Data paths
RAW_PATH = "data/raw/creditcard_v1.csv"
PROCESSED_PATH = "data/processed/creditcard_v1_processed.csv"

# Model paths
MODEL_PATH = "model/model.pkl"

# Training params
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -------------------
# Retraining specific
# -------------------

DATA_PATH = PROCESSED_PATH
DATA_VERSION = "v1"

# Candidate model paths
CANDIDATE_MODEL_PATH = "model/candidate_model.pkl"
CANDIDATE_METRICS_PATH = "model/candidate_model_metrics.json"
CANDIDATE_METADATA_PATH = "model/candidate_model_metadata.json"

# Production model paths
PRODUCTION_MODEL_PATH = MODEL_PATH
PRODUCTION_METRICS_PATH = "model/model_metrics.json"

# Model registry
MODEL_REGISTRY_BASE = "model/registry"

