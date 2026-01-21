# prepare_model.py
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate dummy fraud dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    random_state=42
)

# Train simple model
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X, y)

# Save model inside /model
joblib.dump(clf, "model/model.pkl")

print("Saved model to model/model.pkl")