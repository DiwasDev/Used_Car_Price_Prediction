"""
run_pipeline.py
---------------
Script to run the end-to-end pipeline: preprocessing, model building, and evaluation.
"""

from __future__ import annotations
import os
import joblib



# How to load it back later:
# loaded_model = joblib.load('my_model.pkl')


import logging
from training_pipeline import run_training_pipeline
from steps.build_model import build_model
import pandas as pd
from steps.evaluate import evaluate_model_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """
    Runs the entire pipeline including preprocessing, model building, and evaluationn .

    Parameters:
            None

    Returns:
           None: Nothing is returned
    """
    try:
        logger.info("Step 1: Running the training pipeline (preprocessing)...")
        train_df, test_df = run_training_pipeline()
        
        logger.info("Step 2: Building/training the model...")
        model = build_model(train_df)

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/model.pkl')
        
        logger.info("Step 3: Evaluating the trained model...")
        metrics = evaluate_model_step(model, test_df)
        
        logger.info("Pipeline executed successfully. Evaluation Metrics: %s", metrics)
    except Exception as e:
        logger.error("Pipeline run failed: %s", e)
        raise e


if __name__ == "__main__":
    run_pipeline()
