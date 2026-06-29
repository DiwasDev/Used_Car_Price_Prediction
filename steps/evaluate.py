"""
evaluate.py
-----------
Step function to evaluate the trained model on test data.
"""

from __future__ import annotations

from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from src.components.modeler.evaluator import ModelEvaluator


def evaluate_model_step(
    model: RegressorMixin, test_df: pd.DataFrame, target_col: str = "price_usd"
) -> dict[str, float]:
    """
    Splits test data and uses ModelEvaluator to log evaluation metrics.

    Parameters:
            model: RegressorMixin: Trained regression model to evaluate
            test_df: pd.DataFrame: Preprocessed test DataFrame
            target_col: str: Name of the target variable column

    Returns:
           metrics: dict[str, float]: Dictionary containing regression metrics (MSE, RMSE, R2, MAE, etc.)
    """
    try:
        logging.info("Splitting test data into features and target...")
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]


        logging.info("Running evaluator step...")
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, X_test, y_test)
        
        return metrics
    except Exception as e:
        logging.error("Error in evaluate_model_step: %s", e)
        raise e
