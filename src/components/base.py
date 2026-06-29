"""Shared abstractions for the preprocessing pipeline."""

from __future__ import annotations

from src.logger import logging
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class BaseTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Template-method base: optional fit, required transform.
    """

    def fit(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> BaseTransformer:
        """
        Fits the transformer on the input DataFrame.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target vector.
            
        Returns:
            BaseTransformer: Self instance.
        """
        try:
            return self
        except Exception as e:
            logging.error("Error in BaseTransformer.fit: %s", e)
            raise e

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame.
            
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        ...

    def fit_transform(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Fits the transformer and then transforms the input DataFrame.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target vector.
            
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        try:
            return self.fit(df, y).transform(df)
        except Exception as e:
            logging.error("Error in BaseTransformer.fit_transform: %s", e)
            raise e

    def __sklearn_is_fitted__(self) -> bool:
        """Indicate to scikit-learn check_is_fitted that this transformer is fitted."""
        return True
