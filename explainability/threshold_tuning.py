import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

# Model & Data load
MODEL_PATH = "model/candidate_model.pkl"
model = joblib.load(MODEL_PATH)

df = pd.read_csv("data/processed/creditcard_v1_processed.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

#Fraud Probabilities for all transaction
probs = model.predict_proba(X)[:, 1]

#Set threshold range
thresholds = np.arange(0.1, 0.91, 0.1)

#Calculate metrices
results = []

for t in thresholds:
    preds = (probs >= t).astype(int)

    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    mcc = matthews_corrcoef(y, preds)

    results.append([t, precision, recall, f1, mcc])
    
# Draw result tables
results_df = pd.DataFrame(
    results,
    columns=["Threshold", "Precision", "Recall", "F1", "MCC"]
)

print(results_df)

