# Monitoring and Alerting Overview

This document describes the monitoring and alerting setup implemented for the fraud detection system, focusing on runtime performance, system health, and production readiness.

## Monitoring Stack

The monitoring stack consists of:
- Prometheus for metrics collection and alert evaluation
- Grafana for visualization and exploratory analysis
- Alertmanager for alert routing and notification delivery
- cAdvisor for container-level resource monitoring

## Alerting Strategy

Alerts were designed to detect both infrastructure-level failures and application-level issues.

Key alerts include:
- Fraud API availability alerts
- Histogram-based P95 prediction latency alerts
- HTTP 5xx error-rate alerts

## Performance Monitoring

Prediction latency is monitored using histogram metrics exposed by the fraud detection API.
The 95th percentile latency (P95) is evaluated to capture worst-case performance behavior.

During controlled traffic generation, the observed baseline P95 latency was approximately 5 ms.
Based on this baseline, alert thresholds were tuned to detect meaningful performance degradation while avoiding false positives.

## Error Rate Monitoring

Application error rates were monitored using HTTP status code metrics.
No 5xx errors were observed during normal operation, and existing error-rate thresholds were retained as reliable indicators of application failure.

## Alert Validation

Alert delivery was validated end-to-end by triggering controlled failure scenarios.
Slack notifications were successfully received, confirming correct integration between Prometheus, Alertmanager, and Slack.

## Summary

The monitoring and alerting system was implemented using a data-driven approach and validated under realistic conditions to ensure production readiness and operational reliability.
