## Alert Threshold Tuning (Day 4)

### Prediction Latency
Histogram-based P95 prediction latency was analyzed under generated load.
The observed baseline latency was approximately 5 ms.
Based on this behavior, warning and critical thresholds were tuned to 200 ms
and 500 ms respectively to detect meaningful performance degradation while
avoiding false positives.

### Error Rate
The fraud API error rate was observed over extended monitoring windows and
remained close to zero under normal operation.
The existing 5% 5xx error-rate threshold was retained, as it provides a
reliable signal for application-level failures without triggering on
occasional transient errors.
