### Local Explainability Observations

Fraud Case:
- High transaction amount strongly increased fraud probability
- V14 and V10 contributed positively toward fraud
- Model confidence is driven by abnormal feature patterns

Non-Fraud Case:
- Feature contributions were balanced
- No single feature strongly pushed toward fraud
- Prediction aligns with expected normal behavior

### Robustness Testing

- Model predictions showed minimal change under small Gaussian noise.
- Average prediction shift remained within acceptable bounds.
- Edge case testing (high transaction amounts) did not cause extreme prediction spikes.
- Label flip rate under noise was low, indicating stable decision boundaries.

### Threshold Tuning Results

- Multiple thresholds from 0.1 to 0.9 were evaluated.
- Lower thresholds improved recall, while higher thresholds increased precision.
- Threshold 0.1 achieved the highest F1-score (0.78), indicating optimal balance.
- This threshold was selected to minimize missed fraud cases while maintaining acceptable false positive rates.


