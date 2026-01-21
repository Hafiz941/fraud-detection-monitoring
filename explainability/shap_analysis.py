import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
MODEL_PATH = "model/candidate_model.pkl"

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully")

#Use validation data
DATA_PATH = "data/processed/creditcard_v1_processed.csv"

df = pd.read_csv(DATA_PATH)

# Separate features & label
X = df.drop(columns=["Class"])  
y = df["Class"]

print("Dataset shape:", X.shape)

# Reduce Data size
X_sample = X.sample(n=2000, random_state=42)

#Crreate SHAP explainer
explainer = shap.Explainer(model, X_sample)

# Compute SHAP values
shap_values = explainer(X_sample)
print("✅ SHAP values calculated")

# Plot summary
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("explainability/shap_summary.png", bbox_inches="tight")
plt.close()

print("📊 SHAP summary plot saved")

# Bar Plot
plt.figure()
shap.plots.bar(shap_values, max_display=10, show=False)
plt.savefig("explainability/shap_bar.png", bbox_inches="tight")
plt.close()

print("📊 SHAP bar plot saved")

#FInd one fraud and one non fraud row
fraud_idx = y[y == 1].index[0]
nonfraud_idx = y[y == 0].index[0]

print("Fraud index:", fraud_idx)
print("Non-fraud index:", nonfraud_idx)

# Explaination for this fraud row using SHAP
fraud_sample = X.loc[[fraud_idx]]

shap_values_fraud = explainer(fraud_sample)

#Plot waterfall
#which feature pushes prediction towards fraud
#which feature pushes prediction away

plt.figure()
shap.plots.waterfall(shap_values_fraud[0], show=False)
plt.savefig("explainability/fraud_case.png", bbox_inches="tight")
plt.close()

print("📌 Fraud case explanation saved")

# Explanation one non fraud row
nonfraud_sample = X.loc[[nonfraud_idx]]

shap_values_nonfraud = explainer(nonfraud_sample)

plt.figure()
shap.plots.waterfall(shap_values_nonfraud[0], show=False)
plt.savefig("explainability/nonfraud_case.png", bbox_inches="tight")
plt.close()

print("📌 Non-fraud case explanation saved")






