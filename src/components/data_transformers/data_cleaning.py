"""
data_cleaning.py
----------------
Raw string columns → typed numeric/boolean columns.

Design patterns
  Strategy  – each column fix is a pluggable CleaningStrategy
  Factory   – CleaningStrategyFactory builds the default strategy list
  Facade    – DataCleaner exposes a single transform() entry point
"""

from __future__ import annotations

import logging
import os
import sys
from abc import ABC, abstractmethod

# Add project root to sys.path to allow direct script execution from src/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

from src.components.base import BaseTransformer

logger = logging.getLogger(__name__)


class CleaningStrategy(ABC):
    """Strategy interface for a single cleaning operation."""

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class PriceCleaningStrategy(CleaningStrategy):
    """'$10,300' → price_usd (float). Drops original price column."""

    SOURCE_COL = "price"
    TARGET_COL = "price_usd"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df[self.TARGET_COL] = (
                df[self.SOURCE_COL]
                .str.replace(r"[$,]", "", regex=True)
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
            )
            return df.drop(columns=[self.SOURCE_COL])
        except Exception as e:
            logger.error("Error in PriceCleaningStrategy.clean: %s", e)
            raise e


class MileageCleaningStrategy(CleaningStrategy):
    """'51,000 mi.' → mileage_num (float). Drops original milage column."""

    SOURCE_COL = "milage"
    TARGET_COL = "mileage_num"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df[self.TARGET_COL] = (
                df[self.SOURCE_COL]
                .str.replace(r"[,]", "", regex=True)
                .str.replace(r"\s*mi\.\s*$", "", regex=True)
                .str.strip()
                .pipe(pd.to_numeric, errors="coerce")
            )
            return df.drop(columns=[self.SOURCE_COL])
        except Exception as e:
            logger.error("Error in MileageCleaningStrategy.clean: %s", e)
            raise e


class FuelTypeCleaningStrategy(CleaningStrategy):
    """Replace known junk tokens with NaN for downstream imputation."""

    JUNK_VALUES = frozenset({"–", "not supported", ""})

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["fuel_type"] = df["fuel_type"].apply(
                lambda x: None if str(x).strip() in self.JUNK_VALUES else x
            )
            return df
        except Exception as e:
            logger.error("Error in FuelTypeCleaningStrategy.clean: %s", e)
            raise e


class CleanTitleCleaningStrategy(CleaningStrategy):
    """'Yes' / NaN → has_clean_title binary flag. Drops clean_title."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["has_clean_title"] = (
                df["clean_title"].map({"Yes": 1}).fillna(0).astype(int)
            )
            return df.drop(columns=["clean_title"])
        except Exception as e:
            logger.error("Error in CleanTitleCleaningStrategy.clean: %s", e)
            raise e

## Just if we wnat to reutrn default pattern of cleanign pipeline
class CleaningStrategyFactory:
    """Factory for assembling the default cleaning strategy chain."""

    @staticmethod
    def create_default() -> list[CleaningStrategy]:
        try:
            return [
                PriceCleaningStrategy(),
                MileageCleaningStrategy(),
                FuelTypeCleaningStrategy(),
                CleanTitleCleaningStrategy(),
            ]
        except Exception as e:
            logger.error("Error in CleaningStrategyFactory.create_default: %s", e)
            raise e

## Context class for strategy pattern
class DataCleaner(BaseTransformer):
    """
    Facade that runs a chain of CleaningStrategy objects in order.

    Parameters
    ----------
    strategies : list[CleaningStrategy] | None
        Custom strategy chain. Defaults to CleaningStrategyFactory.create_default().
    """

    def __init__(self, strategies: list[CleaningStrategy] | None = None) -> None:
        try:
            self.strategies = strategies or CleaningStrategyFactory.create_default()
        except Exception as e:
            logger.error("Error in DataCleaner.__init__: %s", e)
            raise e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        try:
            for strategy in self.strategies:
                result = strategy.clean(result)
            logger.info("DataCleaner done. Shape: %s", result.shape)
            return result
        except Exception as e:
            logger.error("Error cleaning data: %s", e)
            raise e


# if __name__ == "__main__":
#     df = pd.read_csv("/home/divas/ml/logistics_project/data/raw/used_cars.csv")
#     data_cleaner = DataCleaner()
#     data_cleaner.fit(df)
#     df = data_cleaner.transform(df)
#     print(df.head())