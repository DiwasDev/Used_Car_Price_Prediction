"""Shared abstractions for the preprocessing pipeline."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

logger = logging.getLogger(__name__)


class BaseTransformer(BaseEstimator, TransformerMixin, ABC):
    """Template-method base: optional fit, required transform."""

    def fit(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> BaseTransformer:
        try:
            return self
        except Exception as e:
            logger.error("Error in BaseTransformer.fit: %s", e)
            raise e

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        try:
            return self.fit(df, y).transform(df)
        except Exception as e:
            logger.error("Error in BaseTransformer.fit_transform: %s", e)
            raise e

    def __sklearn_is_fitted__(self) -> bool:
        """Indicate to scikit-learn check_is_fitted that this transformer is fitted."""
        return True
