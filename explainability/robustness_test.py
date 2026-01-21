import joblib
import pandas as pd
import numpy as np

# Model & Data load
MODEL_PATH = "model/candidate_model.pkl"

model = joblib.load(MODEL_PATH)

df = pd.read_csv("data/processed/creditcard_v1_processed.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# small sample for testing
X_sample = X.sample(1000, random_state=42)

#Orignal Prediction
original_preds = model.predict_proba(X_sample)[:, 1]

#Adding noice
noise = np.random.normal(0, 0.01, X_sample.shape)
X_noisy = X_sample + noise

#Noice Prediction
noisy_preds = model.predict_proba(X_noisy)[:, 1]

#Difference measure of orignal and noice prediction
diff = np.abs(original_preds - noisy_preds)

print("Average prediction change:", diff.mean())
print("Max prediction change:", diff.max())

#Extreme input
X_extreme = X_sample.copy()

# increase amount feature heavily
if "Amount" in X_extreme.columns:
    X_extreme["Amount"] = X_extreme["Amount"] * 10
    
#Extreme Prediction
extreme_preds = model.predict_proba(X_extreme)[:, 1]

print("Mean prediction (normal):", original_preds.mean())
print("Mean prediction (extreme):", extreme_preds.mean())

#Flip rate check
original_labels = (original_preds > 0.5).astype(int)
noisy_labels = (noisy_preds > 0.5).astype(int)

flip_rate = (original_labels != noisy_labels).mean()
print("Label flip rate:", flip_rate)
    



