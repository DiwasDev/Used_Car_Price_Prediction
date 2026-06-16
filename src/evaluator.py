"""
evaluator.py
------------
Evaluation module using strategy design pattern.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import RegressorMixin

logger = logging.getLogger(__name__)


class EvaluationStrategy(ABC):
    """
    Abstract base class for evaluation metrics.
    """

    @abstractmethod
    def evaluate(self, y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
        """
        Abstract method to compute an evaluation metric.

        Parameters:
                y_true: pd.Series | np.ndarray: True values
                y_pred: pd.Series | np.ndarray: Predicted values

        Returns:
               metric_value: float: Calculated metric value
        """
        pass


class MeanSquaredErrorEvaluation(EvaluationStrategy):
    """
    Mean Squared Error evaluation strategy.
    """

    def evaluate(self, y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
        """
        Computes the Mean Squared Error.

        Parameters:
                y_true: pd.Series | np.ndarray: True values
                y_pred: pd.Series | np.ndarray: Predicted values

        Returns:
               mse: float: Calculated Mean Squared Error
        """
        try:
            mse = mean_squared_error(y_true, y_pred)
            logger.info("Mean Squared Error: %f", mse)
            return float(mse)
        except Exception as e:
            logger.error("Error in MeanSquaredErrorEvaluation: %s", e)
            raise e


class RootMeanSquaredErrorEvaluation(EvaluationStrategy):
    """
    Root Mean Squared Error evaluation strategy.
    """

    def evaluate(self, y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
        """
        Computes the Root Mean Squared Error.

        Parameters:
                y_true: pd.Series | np.ndarray: True values
                y_pred: pd.Series | np.ndarray: Predicted values

        Returns:
               rmse: float: Calculated Root Mean Squared Error
        """
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logger.info("Root Mean Squared Error: %f", rmse)
            return float(rmse)
        except Exception as e:
            logger.error("Error in RootMeanSquaredErrorEvaluation: %s", e)
            raise e


class RSquaredEvaluation(EvaluationStrategy):
    """
    R-Squared evaluation strategy.
    """

    def evaluate(self, y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
        """
        Computes the R-Squared coefficient of determination.

        Parameters:
                y_true: pd.Series | np.ndarray: True values
                y_pred: pd.Series | np.ndarray: Predicted values

        Returns:
               r2: float: Calculated R-Squared value
        """
        try:
            r2 = r2_score(y_true, y_pred)
            logger.info("R-Squared: %f", r2)
            return float(r2)
        except Exception as e:
            logger.error("Error in RSquaredEvaluation: %s", e)
            raise e


class MeanAbsoluteErrorEvaluation(EvaluationStrategy):
    """
    Mean Absolute Error evaluation strategy.
    """

    def evaluate(self, y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
        """
        Computes the Mean Absolute Error.

        Parameters:
                y_true: pd.Series | np.ndarray: True values
                y_pred: pd.Series | np.ndarray: Predicted values

        Returns:
               mae: float: Calculated Mean Absolute Error
        """
        try:
            mae = mean_absolute_error(y_true, y_pred)
            logger.info("Mean Absolute Error: %f", mae)
            return float(mae)
        except Exception as e:
            logger.error("Error in MeanAbsoluteErrorEvaluation: %s", e)
            raise e



class ModelEvaluator:
    """
    Evaluator class that uses model to predict and logs multiple metrics.
    """

    def __init__(self) -> None:
        """
        Initializes the ModelEvaluator.

        Parameters:
                None

        Returns:
               None: Nothing is returned
        """
        self.metrics = {
            "MSE": MeanSquaredErrorEvaluation(),
            "RMSE": RootMeanSquaredErrorEvaluation(),
            "R2": RSquaredEvaluation(),
            "MAE": MeanAbsoluteErrorEvaluation()
        }

    def evaluate_model(
        self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, float]:
        """
        Uses the model to generate predictions and calculates/logs evaluation metrics.

        Parameters:
                model: RegressorMixin: The trained model to evaluate
                X_test: pd.DataFrame: Feature matrix for testing
                y_test: pd.Series: True target vector for testing

        Returns:
               results: dict[str, float]: Dictionary containing all metric values
        """
        try:
            logger.info("Generating predictions using the model...")
            y_pred = model.predict(X_test)
            
            results = {}
            for name, strategy in self.metrics.items():
                val = strategy.evaluate(y_test, y_pred)
                results[name] = val
                
            logger.info("Evaluation metrics: %s", results)
            return results
        except Exception as e:
            logger.error("Error during model evaluation: %s", e)
            raise e

