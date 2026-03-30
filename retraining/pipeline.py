from retraining.preprocess import preprocess
from retraining.retrain import retrain


def run_pipeline():
    try:
        print("🚀 Starting full retraining pipeline...\n")

        print("🔹 Step 1: Data Preprocessing...")
        preprocess()

        print("\n🔹 Step 2: Model Retraining...")
        retrain()

        print("\n✅ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")


if __name__ == "__main__":
    run_pipeline()