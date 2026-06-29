"""
model_builder.py
----------------
Model building module using the Strategy design pattern.
"""

from __future__ import annotations

from src.logger import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin


class ModelTrainingStrategy(ABC):
    """
    Abstract Base Class representing a model training strategy.
    """

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Abstract method to train a model.

        Parameters:
                X_train: pd.DataFrame: Feature matrix for training
                y_train: pd.Series: Target vector for training

        Returns:
               model: RegressorMixin: Trained regression model
        """
        pass


class LinearRegressionStrategy(ModelTrainingStrategy):
    """
    Linear Regression strategy for model training.
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        """
        Initializes the LinearRegressionStrategy with optional parameters.

        Parameters:
                fit_intercept: bool: Whether to calculate the intercept for this model

        Returns:
               None: Nothing is returned
        """
        self.fit_intercept = fit_intercept

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
        """
        Trains a Linear Regression model on the provided training data.

        Parameters:
                X_train: pd.DataFrame: Feature matrix for training
                y_train: pd.Series: Target vector for training

        Returns:
               model: LinearRegression: Fitted Linear Regression model
        """
        try:
            logging.info("Training Linear Regression model...")
            model = LinearRegression(fit_intercept=self.fit_intercept)
            model.fit(X_train, y_train)
            logging.info("Linear Regression model trained successfully.")
            return model
        except Exception as e:
            logging.error("Error training Linear Regression model: %s", e)
            raise e


class ModelBuilder:
    """
    Context class that uses a ModelTrainingStrategy to train a model.
    """

    def __init__(self, strategy: ModelTrainingStrategy) -> None:
        """
        Initializes the ModelBuilder context with a training strategy.

        Parameters:
                strategy: ModelTrainingStrategy: The strategy to be used for model training

        Returns:
               None: Nothing is returned
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelTrainingStrategy) -> None:
        """
        Sets a new training strategy for the builder context.

        Parameters:
                strategy: ModelTrainingStrategy: The new strategy to be used

        Returns:
               None: Nothing is returned
        """
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Delegates the model building/training to the active strategy.

        Parameters:
                X_train: pd.DataFrame: Feature matrix for training
                y_train: pd.Series: Target vector for training

        Returns:
               model: RegressorMixin: Trained regression model
        """
        try:
            logging.info("Building model using the selected strategy...")
            return self._strategy.train(X_train, y_train)
        except Exception as e:
            logging.error("Error in ModelBuilder.build_model: %s", e)
            raise e
