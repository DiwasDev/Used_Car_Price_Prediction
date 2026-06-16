"""
scaler.py
---------
Numeric feature scaling for model training.

Design patterns
  Strategy  – StandardScalerStrategy, RobustScalerStrategy, MinMaxScalerStrategy
  Factory   – ScalerFactory selects the scaling backend
  Facade    – FeatureScaler exposes fit/transform API
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.base import BaseTransformer

logger = logging.getLogger(__name__)


class ScalerType(Enum):
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"


class ScalingStrategy(ABC):
    """Strategy interface wrapping a sklearn scaler."""

    def __init__(self) -> None:
        try:
            self._fitted_columns: list[str] = []
        except Exception as e:
            logger.error("Error in ScalingStrategy.__init__: %s", e)
            raise e

    @abstractmethod
    def _create_scaler(self):
        ...

    def fit(
        self, df: pd.DataFrame, columns: list[str]
    ) -> ScalingStrategy:
        try:
            self._fitted_columns = [c for c in columns if c in df.columns]
            self._scaler = self._create_scaler()
            if self._fitted_columns:
                self._scaler.fit(df[self._fitted_columns])
            return self
        except Exception as e:
            logger.error("Error in ScalingStrategy.fit: %s", e)
            raise e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if not self._fitted_columns:
                return df.copy()
            result = df.copy()
            result[self._fitted_columns] = self._scaler.transform(
                df[self._fitted_columns]
            )
            return result
        except Exception as e:
            logger.error("Error in ScalingStrategy.transform: %s", e)
            raise e


class StandardScalerStrategy(ScalingStrategy):
    def _create_scaler(self):
        try:
            return StandardScaler()
        except Exception as e:
            logger.error("Error in StandardScalerStrategy._create_scaler: %s", e)
            raise e


class RobustScalerStrategy(ScalingStrategy):
    def _create_scaler(self):
        try:
            return RobustScaler()
        except Exception as e:
            logger.error("Error in RobustScalerStrategy._create_scaler: %s", e)
            raise e


class MinMaxScalerStrategy(ScalingStrategy):
    def _create_scaler(self):
        try:
            return MinMaxScaler()
        except Exception as e:
            logger.error("Error in MinMaxScalerStrategy._create_scaler: %s", e)
            raise e


class ScalerFactory:
    """Factory that returns the appropriate ScalingStrategy."""

    _REGISTRY = {
        ScalerType.STANDARD: StandardScalerStrategy,
        ScalerType.ROBUST: RobustScalerStrategy,
        ScalerType.MINMAX: MinMaxScalerStrategy,
    }

    @classmethod
    def create(cls, scaler_type: ScalerType = ScalerType.STANDARD) -> ScalingStrategy:
        try:
            return cls._REGISTRY[scaler_type]()
        except Exception as e:
            logger.error("Error in ScalerFactory.create for scaler_type %s: %s", scaler_type, e)
            raise e


class FeatureScaler(BaseTransformer):
    """
    Facade that scales selected numeric columns.

    By default scales continuous engine/year features, excluding the target
    and already-encoded binary/one-hot columns.
    """

    DEFAULT_COLUMNS = [
        "brand",
        "model_year",
        "mileage_num",
        "engine_hp",
        "engine_displacement",
    ]

    def __init__(
        self,
        columns: list[str] | None = None,
        scaler_type: ScalerType = ScalerType.STANDARD,
    ) -> None:
        try:
            self.columns = columns or self.DEFAULT_COLUMNS
            self.strategy = ScalerFactory.create(scaler_type)
            self._is_fitted = False
        except Exception as e:
            logger.error("Error in FeatureScaler.__init__: %s", e)
            raise e

    def fit(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> FeatureScaler:
        try:
            self.strategy.fit(df, self.columns)
            self._is_fitted = True
            return self
        except Exception as e:
            logger.error("Error in FeatureScaler.fit: %s", e)
            raise e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if not self._is_fitted:
                raise RuntimeError("FeatureScaler must be fitted before transform.")
            result = self.strategy.transform(df)
            logger.info("FeatureScaler applied to columns: %s", self.strategy._fitted_columns)
            return result
        except Exception as e:
            logger.error("Error in FeatureScaler.transform: %s", e)
            raise e
