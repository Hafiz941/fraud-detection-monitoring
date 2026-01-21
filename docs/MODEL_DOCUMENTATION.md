## Model Overview

This project implements a machine learning–based fraud detection system designed to identify fraudulent financial transactions. The model outputs a fraud probability score, which is converted into a final decision using a business aligned threshold.

## Data Pipeline

- Raw transaction data is preprocessed and cleaned.
- Features are scaled and transformed before training.
- The target variable represents fraudulent (1) and non fraudulent (0) transactions.
- Data imbalance is handled implicitly through evaluation metrics and threshold tuning.

## Model Training

A Logistic Regression classifier was trained to estimate fraud probabilities. The model was selected due to its interpretability, stability, and suitability for imbalanced datasets.

## Model Explainability

SHAP (SHapley Additive exPlanations) was integrated to interpret model predictions.

- Global explainability identified the most influential features across all predictions.
- Local explainability provided transaction level insights for both fraud and non fraud cases.
- Explainability results confirmed that model decisions are driven by meaningful and consistent feature patterns.

## Robustness & Stability Testing

The model was evaluated under noisy and extreme input conditions.

- Small Gaussian noise introduced negligible changes in prediction probabilities.
- Extreme value testing resulted in controlled prediction shifts.
- No decision label flips were observed.

These results indicate strong model stability and production readiness.

## Threshold Tuning & Business Alignment

Multiple decision thresholds were evaluated using Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

- Lower thresholds improved fraud recall.
- Higher thresholds improved alert precision.
- A business aligned threshold was selected to minimize missed fraud while maintaining acceptable false positive rates.

## Final Decision Logic

The final fraud decision is computed as:

- Predict fraud probability using the trained model.
- Apply the selected threshold to classify transactions.
- Generate alerts for transactions exceeding the threshold.

## Monitoring & Future Work

- Performance metrics can be monitored over time.
- Model retraining can be triggered based on data drift.
- Explainability and robustness checks can be repeated periodically.









