"""
build_model.py
--------------
Step function to build/train the regression model.
"""

from __future__ import annotations

import logging
import pandas as pd
from sklearn.base import RegressorMixin
from src.model_builder import ModelBuilder, LinearRegressionStrategy

logger = logging.getLogger(__name__)


def build_model(train_df: pd.DataFrame, target_col: str = "price_usd") -> RegressorMixin:
    """
    Splits the training data into features and target, and trains the model.

    Parameters:
            train_df: pd.DataFrame: Preprocessed training DataFrame
            target_col: str: Name of the target variable column

    Returns:
           model: RegressorMixin: Trained regression model
    """
    try:
        logger.info("Splitting training data into features and target...")
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        logger.info("Training Linear Regression model using ModelBuilder strategy...")
        strategy = LinearRegressionStrategy()
        builder = ModelBuilder(strategy)
        model = builder.build_model(X_train, y_train)
        
        return model
    except Exception as e:
        logger.error("Error in build_model step: %s", e)
        raise e
