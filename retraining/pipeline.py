from retraining.preprocess import preprocess
from retraining.retrain import retrain
from retraining.logger import get_logger

logger = get_logger()

def run_pipeline():
    try:
        logger.info("🚀 Starting full retraining pipeline...\n")

        logger.info("🔹 Step 1: Data Preprocessing...")
        preprocess()

        logger.info("\n🔹 Step 2: Model Retraining...")
        retrain()

        logger.info("\n✅ Pipeline completed successfully!")

    except Exception as e:
        logger.info(f"\n❌ Pipeline failed: {e}")


if __name__ == "__main__":
    run_pipeline()