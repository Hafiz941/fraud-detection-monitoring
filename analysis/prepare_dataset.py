import gzip
import json
import pandas as pd
from pathlib import Path

# -----------------------
# Paths
# -----------------------
RAW_PATH = Path("data/raw/fraudio.gz")
OUTPUT_PATH = Path("data/processed/fraudio_transactions.csv")

# -----------------------
# Load JSON.GZ
# -----------------------
print("Loading manager dataset...")

with gzip.open(RAW_PATH, "rt", encoding="utf-8") as f:
    data = json.load(f)

transactions = data["transactions"]

print(f"Total transactions loaded: {len(transactions)}")

# -----------------------
# Convert to DataFrame
# -----------------------
df = pd.json_normalize(transactions)

print("DataFrame created")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# -----------------------
# Select useful columns (safe subset)
# -----------------------
selected_columns = [
    "id",
    "transacted_at",
    "base_currency_amount",
    "base_currency",
    "successful",
    "transaction_type",
    "merchant_id",
    "card_bin",
    "card_country",
    "channel",
    "chargeback"
]

# keep only columns that exist
selected_columns = [c for c in selected_columns if c in df.columns]

df_clean = df[selected_columns]

print("Selected columns:")
print(df_clean.columns.tolist())

# -----------------------
# Save to CSV
# -----------------------
df_clean.to_csv(OUTPUT_PATH, index=False)

print(f"Processed CSV saved to: {OUTPUT_PATH}")
print("Day-2 Step-1 completed successfully.")

# -----------------------
# Timestamp cleaning
# -----------------------
print("Cleaning timestamps...")

df_clean["transacted_at"] = pd.to_datetime(
    df_clean["transacted_at"],
    errors="coerce",
    utc=True
)

print("Timestamp range:")
print(df_clean["transacted_at"].min(), "→", df_clean["transacted_at"].max())

# Total transactions
total_transactions = len(df_clean)

# Success vs failure
success_count = df_clean["successful"].sum()
failure_count = total_transactions - success_count

#Success Rate
success_rate = success_count / total_transactions * 100

#Chargeback count (Fraud signal)
chargeback_count = df_clean["chargeback"].notna().sum()

print("\n--- BASIC KPIs ---")
print(f"Total transactions: {total_transactions}")
print(f"Successful: {success_count}")
print(f"Failed: {failure_count}")
print(f"Success rate: {success_rate:.2f}%")
print(f"Chargebacks: {chargeback_count}")




