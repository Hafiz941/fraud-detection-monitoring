import gzip
import json
import pandas as pd
from pathlib import Path

# -----------------------
# Paths
# -----------------------
MANAGER_DATASET_PATH = Path("data/raw/fraudio.gz")
EXISTING_CSV_PATH = Path("data/raw/creditcard_v1.csv")

# -----------------------
# Load Manager Dataset (JSON.GZ)
# -----------------------
print("Loading manager-provided dataset...")

with gzip.open(MANAGER_DATASET_PATH, "rt", encoding="utf-8") as f:
    manager_data = json.load(f)

transactions = manager_data["transactions"]

print(f"Total transactions in manager dataset: {len(transactions)}")
print("Transaction keys:")
print(transactions[0].keys())

# -----------------------
# Load Existing CSV Dataset (READ ONLY)
# -----------------------
print("\nLoading existing CSV dataset (read-only)...")

df_existing = pd.read_csv(EXISTING_CSV_PATH)

print(f"CSV shape: {df_existing.shape}")
print("CSV columns:")
print(df_existing.columns.tolist())

# -----------------------
# Notes
# -----------------------
print("\nDay-1 analysis completed successfully.")
print("No data transformation performed.")
