"""
encoder.py
----------
Categorical encoding for model-ready features.

Design patterns
  Strategy  – OneHotEncodingStrategy, TargetEncodingStrategy
  Template  – CategoricalEncoder.fit → transform
  Facade    – CategoricalEncoder orchestrates all encoding strategies
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from src.components.base import BaseTransformer

logger = logging.getLogger(__name__)


class EncodingStrategy(ABC):
    """Strategy interface for a fit/transform encoding step."""

    def fit(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> None:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        try:
            return self.fit(df, y).transform(df)
        except Exception as e:
            logger.error("Error in EncodingStrategy.fit_transform: %s", e)
            raise e


class OneHotEncodingStrategy(EncodingStrategy):
    """One-hot encode low-cardinality categorical columns."""

    def __init__(
        self,
        columns: list[str] | None = None,
        drop_first: bool = True,
    ) -> None:
        try:
            self.columns = columns or [
                "fuel_type",   
                "transmission_type",
                "int_col",
                "ext_col",
            ]
            self.drop_first = drop_first
            self._encoder: OneHotEncoder | None = None
        except Exception as e:
            logger.error("Error in OneHotEncodingStrategy.__init__: %s", e)
            raise e

    def fit(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> OneHotEncodingStrategy:
        try:
            self._encoder = OneHotEncoder(
                drop="first" if self.drop_first else None,
                handle_unknown="ignore",  # error when it sees completely new category
                sparse_output=False,
            )
            self._encoder.fit(df[self.columns])
            return self
        except Exception as e:
            logger.error("Error in OneHotEncodingStrategy.fit: %s", e)
            raise e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Starting one hot encoding transform. in df shape:{df.shape}")
        try:
            if self._encoder is None:
                raise RuntimeError("OneHotEncodingStrategy must be fitted before transform.")

            transformed = self._encoder.transform(df[self.columns])
            # Convert to ndarray if sparse matrix is returned
            # if hasattr(transformed, "toarray"):
            #     transformed = transformed.toarray()
            encoded = pd.DataFrame(
                transformed,
                columns=self._encoder.get_feature_names_out(self.columns),
                index=df.index,
            )
            logger.info("Completed one hot encoding transformation.")
            return pd.concat([df.drop(columns=self.columns), encoded], axis=1)
        
        except Exception as e:
            logger.error("Error in OneHotEncodingStrategy.transform: %s", e)
            raise e
        


class TargetEncodingStrategy(EncodingStrategy):
    """
    Out-of-fold target encoding for training; master encoder for inference.

    fit_transform() produces leakage-free training values.
    transform() uses a master encoder fit on the full training set.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        target_col: str = "price_usd",
        smoothing: float = 10.0,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        try:
            self.columns = columns or ["brand"]
            self.target_col = target_col
            self.smoothing = smoothing
            self.n_splits = n_splits
            self.random_state = random_state
            self._master_encoder: TargetEncoder | None = None
            self._oof_values: pd.DataFrame | None = None
        except Exception as e:
            logger.error("Error in TargetEncodingStrategy.__init__: %s", e)
            raise e

    def fit(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> TargetEncodingStrategy:
        try:
            target = y if y is not None else df[self.target_col]
            self._master_encoder = TargetEncoder(
                cols=self.columns, smoothing=self.smoothing
            )
            self._master_encoder.fit(df, target)
            return self
        except Exception as e:
            logger.error("Error in TargetEncodingStrategy.fit: %s", e)
            raise e

    def fit_transform(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        try:
            target = y if y is not None else df[self.target_col]
            result = df.copy()

            kf = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            for col in self.columns:
                result[col] = np.nan

            for train_idx, val_idx in kf.split(df):
                fold_encoder = TargetEncoder(
                    cols=self.columns, smoothing=self.smoothing
                )
                fold_encoder.fit(df.iloc[train_idx], target.iloc[train_idx])
                encoded = fold_encoder.transform(df.iloc[val_idx])
                for col in self.columns:
                    result.iloc[val_idx, result.columns.get_loc(col)] = encoded[col]

            self._master_encoder = TargetEncoder(
                cols=self.columns, smoothing=self.smoothing
            )
            self._master_encoder.fit(df, target)
            return result
        except Exception as e:
            logger.error("Error in TargetEncodingStrategy.fit_transform: %s", e)
            raise e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if self._master_encoder is None:
                raise RuntimeError(
                    "TargetEncodingStrategy must be fitted before transform."
                )
            return self._master_encoder.transform(df)
        except Exception as e:
            logger.error("Error in TargetEncodingStrategy.transform: %s", e)
            raise e

# Generates objects for encoding
class CategoricalEncoderFactory:
    """Factory for assembling the default categorical encoding strategy chain."""

    @staticmethod
    def create_default(
        target_col: str = "price_usd", random_state: int = 42
    ) -> list[EncodingStrategy]:
        try:
            return [
                OneHotEncodingStrategy(),
                TargetEncodingStrategy(target_col=target_col, random_state=random_state),
            ]
        except Exception as e:
            logger.error("Error in CategoricalEncoderFactory.create_default: %s", e)
            raise e


#Context: Facade Pattern
class CategoricalEncoder(BaseTransformer):
    """Facade that chains encoding strategies with proper fit/transform semantics."""

    def __init__(
        self, strategies: list[EncodingStrategy] | None = None
    ) -> None:
        try:
            self.strategies = strategies or CategoricalEncoderFactory.create_default()
            self._is_fitted = False
        except Exception as e:
            logger.error("Error in CategoricalEncoder.__init__: %s", e)
            raise e

    def fit(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> CategoricalEncoder:
        try:
            current = df.copy()
            for strategy in self.strategies:
                if isinstance(strategy, TargetEncodingStrategy):
                    strategy.fit(current, y)
                else:
                    strategy.fit(current, y)
                    current = strategy.transform(current)
            self._is_fitted = True
            return self
        except Exception as e:
            logger.error("Error in CategoricalEncoder.fit: %s", e)
            raise e

    def fit_transform(
        self, df: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        try:
            result = df.copy()
            target = y if y is not None else df.get("price_usd")

            for strategy in self.strategies:
                    result = strategy.fit_transform(result, target)
            self._is_fitted = True
            logger.info("CategoricalEncoder fit_transform done. Shape: %s", result.shape)
            return result
        except Exception as e:
            logger.error("Error in CategoricalEncoder.fit_transform: %s", e)
            raise e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if not self._is_fitted:
                raise RuntimeError("CategoricalEncoder must be fitted before transform.")

            result = df.copy()
            for strategy in self.strategies:
                result = strategy.transform(result)
            return result
        except Exception as e:
            logger.error("Error in CategoricalEncoder.transform: %s", e)
            raise e
