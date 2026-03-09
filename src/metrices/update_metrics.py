import pandas as pd
from src.metrics.fraud_metrics import *

def update_fraud_metrics_from_csv():
    tx = pd.read_csv("data/processed/fraudio_transactions.csv")
    merchants = pd.read_csv("data/processed/merchant_kpis.csv")
    bins = pd.read_csv("data/processed/bin_kpis.csv")

    # Transaction counters
    TRANSACTIONS_TOTAL.labels(status="success").inc(tx["successful"].sum())
    TRANSACTIONS_TOTAL.labels(status="failure").inc(len(tx) - tx["successful"].sum())

    # Chargebacks
    CHARGEBACKS_TOTAL.inc(tx["chargeback"].notna().sum())

    # Merchant risk gauges
    for _, r in merchants.iterrows():
        MERCHANT_CHARGEBACK_RATE.labels(
            merchant_id=str(r["merchant_id"])
        ).set(r["chargeback_rate"])

    # BIN risk gauges
    for _, r in bins.iterrows():
        BIN_FAILURE_RATE.labels(
            card_bin=str(r["card_bin"])
        ).set(r["failure_rate"])