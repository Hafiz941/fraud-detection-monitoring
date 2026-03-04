import requests
import os
from datetime import datetime, timedelta, UTC

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

ERROR_THRESHOLD = 0.10   # 10%
COOLDOWN_MINUTES = 30

LAST_RETRAIN_FILE = "retraining/last_retrain.txt"


# -------------------------
# Prometheus Query Helper
# -------------------------
def query_prometheus(query):
    try:
        response = requests.get(PROMETHEUS_URL, params={"query": query})
        result = response.json()["data"]["result"]
        if result:
            return float(result[0]["value"][1])
        return 0.0
    except Exception as e:
        print(f"Prometheus query failed: {e}")
        return 0.0


# -------------------------
# Error Rate Calculation
# -------------------------
def get_error_rate():
    errors = query_prometheus("sum(prediction_errors_total)")
    successes = query_prometheus("sum(prediction_count_total)")

    total = errors + successes
    if total == 0:
        return 0.0

    return errors / total


# -------------------------
# Drift Flag Check
# -------------------------
def drift_active():
    drift_flag = query_prometheus("RETRAINING_REQUIRED")
    return drift_flag == 1.0


# -------------------------
# Cooldown Check
# -------------------------
def cooldown_active():
    if not os.path.exists(LAST_RETRAIN_FILE):
        return False

    with open(LAST_RETRAIN_FILE, "r") as f:
        last_time = datetime.fromisoformat(f.read().strip())

    return datetime.now(UTC) < last_time + timedelta(minutes=COOLDOWN_MINUTES)


def mark_retrain_time():
    with open(LAST_RETRAIN_FILE, "w") as f:
        f.write(datetime.now(UTC).isoformat())


# -------------------------
# Trigger Retraining
# -------------------------
def trigger_retraining():
    print("Retraining triggered...")
    os.system("python -m retraining.retrain candidate")
    mark_retrain_time()


# -------------------------
# Main Logic
# -------------------------
def main():
    error_rate = get_error_rate()
    drift = drift_active()

    print(f"Error Rate: {error_rate:.4f}")
    print(f"Drift Active: {drift}")

    if cooldown_active():
        print("Cooldown active. Skipping retraining.")
        return

    if error_rate > ERROR_THRESHOLD or drift:
        trigger_retraining()
    else:
        print("Model performance stable.")


if __name__ == "__main__":
    main()