# app.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
import os
import json
import joblib
import time
import numpy as np
import pandas as pd
import logging
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# --------------------------------------------------
# Logging (structured, docker-friendly)
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-api")

# --------------------------------------------------
# Prometheus Metrics
# --------------------------------------------------

# System-level
HTTP_ERRORS = Counter(
    "http_errors_total",
    "Total HTTP errors",
    ["path", "status"]
)

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["path", "method", "status"]
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether model is loaded (1=loaded, 0=not)"
)

# Prediction-level
PREDICTION_COUNT = Counter(
    "prediction_count_total",
    "Total number of predictions",
    ["model_name"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

# Business metrics

# 1. Total predictions
fraud_predictions_total = Counter(
    "fraud_predictions_total",
    "Total fraud predictions",
    ["prediction"]
)

# 2. Prediction confidence (average)
fraud_prediction_confidence = Gauge(
    "fraud_prediction_confidence",
    "Average confidence of fraud predictions"
)

# 3. Prediction latency
fraud_prediction_latency = Histogram(
    "fraud_prediction_latency_seconds",
    "Latency for fraud predictions"
)

# Input / drift monitoring
FEATURE_MEAN = Gauge(
    "feature_mean",
    "Mean value of input features",
    ["feature_index"]
)

DRIFT_EVENTS = Counter(
    "drift_events_total",
    "Number of detected drift events",
    ["drift_type"]
)

RETRAINING_REQUIRED = Gauge(
    "retraining_required",
    "Indicates whether retraining is required (1=yes, 0=no)"
)

# -----------------------
# Model Performance Metrics
# -----------------------
MODEL_MCC = Gauge(
    "model_mcc",
    "Matthews correlation coefficient of the current model"
)

MODEL_PRECISION = Gauge(
    "model_precision",
    "Model precision score"
)

MODEL_RECALL = Gauge(
    "model_recall",
    "Model recall score"
)

# -----------------------
# Offline / Business Risk Metrics
# -----------------------
MERCHANT_CHARGEBACK_RATE = Gauge(
    "fraud_merchant_chargeback_rate",
    "Chargeback rate per merchant",
    ["merchant_id"]
)

MERCHANT_TOTAL_TRANSACTIONS = Gauge(
    "fraud_merchant_total_transactions",
    "Total transactions per merchant",
    ["merchant_id"]
)

BIN_FAILURE_RATE = Gauge(
    "fraud_bin_failure_rate",
    "Failure rate per card BIN",
    ["card_bin"]
)

# BIN total transaction count (offline)
BIN_TOTAL_TRANSACTIONS = Gauge(
    "fraud_bin_total_transactions",
    "Total transactions per card BIN",
    ["card_bin"]
)

def load_offline_risk_metrics():
    try:
        logger.info('{"event":"offline_metrics_start"}')

        merchants = pd.read_csv("data/processed/merchant_kpis.csv")
        bins = pd.read_csv("data/processed/bin_kpis.csv")

        logger.info(
            f'{{"event":"offline_metrics_files_loaded",'
            f'"merchants_rows":{len(merchants)},'
            f'"bins_rows":{len(bins)}}}'
        )

        # DEBUG prints (very important)
        logger.info(f"Merchant columns: {merchants.columns.tolist()}")
        logger.info(f"BIN columns: {bins.columns.tolist()}")

        # for _, r in merchants.iterrows():
        #     MERCHANT_CHARGEBACK_RATE.labels(
        #         merchant_id=str(r["merchant_id"])
        #     ).set(float(r["chargeback_rate"]))
        
        for _, r in merchants.iterrows():
            merchant_id = str(r["merchant_id"])

            MERCHANT_TOTAL_TRANSACTIONS.labels(
                merchant_id=merchant_id
            ).set(float(r["total_transactions"]))

            MERCHANT_CHARGEBACK_RATE.labels(
                merchant_id=merchant_id
            ).set(float(r["chargeback_rate"]))
        
        # for _, r in bins.iterrows():
        #     BIN_FAILURE_RATE.labels(
        #         card_bin=str(r["card_bin"])
        #     ).set(float(r["failure_rate"]))
            
        for _, r in bins.iterrows():
            bin_id = str(r["card_bin"])

            # total volume
            BIN_TOTAL_TRANSACTIONS.labels(
                card_bin=bin_id
            ).set(float(r["total_transactions"]))

            # failure rate (already present)
            BIN_FAILURE_RATE.labels(
                card_bin=bin_id
            ).set(float(r["failure_rate"]))

        logger.info('{"event":"offline_metrics_loaded"}')

    except Exception as e:
        logger.exception(f'{{"event":"offline_metrics_load_failed","error":"{e}"}}')

def load_model_performance_metrics():
    try:
        metrics_path = "model/model_metrics.json"

        if not os.path.exists(metrics_path):
            metrics_path = "model/candidate_model_metrics.json"

        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            MODEL_MCC.set(metrics.get("mcc", 0))
            MODEL_PRECISION.set(metrics.get("precision", 0))
            MODEL_RECALL.set(metrics.get("recall", 0))

            logger.info('{"event":"model_metrics_loaded"}')

    except Exception as e:
        logger.exception(
            f'{{"event":"model_metrics_load_failed","error":"{e}"}}'
        )


# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(title="Fraud Detection API")

MODEL_PATH = "model/model.pkl"
model = None

# --------------------------------------------------
# Schemas
# --------------------------------------------------
class PredictRequest(BaseModel):
    features: list # list of numeric features

class PredictResponse(BaseModel):
    prediction: int
    probability: float

# --------------------------------------------------
# Startup
# --------------------------------------------------
@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        MODEL_LOADED.set(1)
        RETRAINING_REQUIRED.set(0)
        logger.info(f'{{"event":"startup","status":"model_loaded","path":"{MODEL_PATH}"}}')
        
        load_offline_risk_metrics() 
        load_model_performance_metrics()  
    except Exception:
        model = None
        MODEL_LOADED.set(0)
        RETRAINING_REQUIRED.set(1)
        logger.exception('{"event":"startup","status":"model_load_failed"}')

        
# --------------------------------------------------
# Request Logging Middleware
# --------------------------------------------------
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round(time.time() - start_time, 4)

    HTTP_REQUESTS.labels(
        path=request.url.path,
        method=request.method,
        status=str(response.status_code)
    ).inc()

    logger.info(
        f'{{"method":"{request.method}",'
        f'"path":"{request.url.path}",'
        f'"status":{response.status_code},'
        f'"duration":{duration}}}'
    )

    return response

# --------------------------------------------------
# Health
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# --------------------------------------------------
# Prediction
# --------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    if model is None:
        HTTP_ERRORS.labels(path="/predict", status="503").inc()
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not isinstance(req.features, list):
        HTTP_ERRORS.labels(path="/predict", status="400").inc()
        raise HTTPException(status_code=400, detail="`features` must be a list")

    arr = np.array(req.features).reshape(1, -1)
    
    expected_features = model.n_features_in_

    if arr.shape[1] != expected_features:
        HTTP_ERRORS.labels(path="/predict", status="400").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_features} features, got {arr.shape[1]}"
        )
    
    # Feature monitoring
    for i, val in enumerate(arr[0]):
        FEATURE_MEAN.labels(feature_index=str(i)).set(float(val))

    start = time.time()

    try:
        # Forced test error
        if req.features[0] == -999:
            raise ValueError("Forced test error")

        # Model Prediction
        pred = int(model.predict(arr)[0])

        prob = (
            float(max(model.predict_proba(arr)[0]))
            if hasattr(model, "predict_proba")
            else 0.0
        )

        # Drift Detection
        if prob < 0.4:
            DRIFT_EVENTS.labels(drift_type="low_confidence").inc()
            RETRAINING_REQUIRED.set(1)
        else:
            RETRAINING_REQUIRED.set(0)

        if len(arr[0]) > 3 and arr[0][3] > 10000:
            DRIFT_EVENTS.labels(drift_type="feature_outlier").inc()

        latency = time.time() - start

        # Model Performance Metrics
        label = "fraud" if pred == 1 else "non_fraud"
        fraud_predictions_total.labels(prediction=label).inc()
        fraud_prediction_confidence.set(prob)

        # Prometheus Metrics
        PREDICTION_COUNT.labels(model_name="fraud-model").inc()
        PREDICTION_LATENCY.observe(latency)

        logger.info(
            f'{{"event":"prediction","client":"{request.client.host}",'
            f'"prediction":{pred},"probability":{prob:.4f},'
            f'"latency":{latency:.4f}}}'
        )

        return {"prediction": pred, "probability": prob}

    except Exception as e:
        latency = time.time() - start

        # Error Metrics (single place)
        HTTP_ERRORS.labels(path="/predict", status="500").inc()
        PREDICTION_ERRORS.inc()
        PREDICTION_LATENCY.observe(latency)

        logger.exception('{"event":"prediction_failed"}')

        raise HTTPException(status_code=500, detail=str(e))
    
    
# --------------------------------------------------
# Metrics
# --------------------------------------------------
@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

