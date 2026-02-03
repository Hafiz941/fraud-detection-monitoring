## Fraud Alert Runbook

### HighMerchantChargebackRate
Trigger: fraud_merchant_chargeback_rate > 3%

Actions:
- Review affected merchant
- Check transaction volume
- Enable manual review or restrict merchant if sustained

---

### HighBINFailureRate
Trigger: fraud_bin_failure_rate > 10%

Actions:
- Identify affected BIN
- Check spread across merchants
- Rate-limit or block BIN if sustained
