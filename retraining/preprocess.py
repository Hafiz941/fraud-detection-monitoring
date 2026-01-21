import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/raw/creditcard_v1.csv"
PROCESSED_PATH = "data/processed/creditcard_v1_processed.csv"

def preprocess():
    df = pd.read_csv(RAW_PATH)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df["Class"] = y.values

    processed_df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed dataset saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess()
