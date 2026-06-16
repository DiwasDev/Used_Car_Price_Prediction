"""
handle_missing.py
-----------------
Imputation strategies for post-feature-engineering gaps.

Design patterns
  Strategy  – each ImputationStrategy handles one imputation rule
  Template  – MissingValueHandler runs strategies in order
  Facade    – MissingValueHandler exposes a single transform() entry point
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.base import BaseTransformer
from src.feature_engineering import EV_BRANDS

logger = logging.getLogger(__name__)


class ImputationStrategy(ABC):
    """Strategy interface for a single imputation operation."""

    @abstractmethod
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class EngineHPImputer(ImputationStrategy):
    """Fill missing engine_hp with brand median, then global median."""

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            ## Grouping with brands median hp
            df["engine_hp"] = df.groupby("brand")["engine_hp"].transform(
                lambda x: x.fillna(x.median())
            )

            ## Fall back with global median hp
            df["engine_hp"] = df["engine_hp"].fillna(df["engine_hp"].median())
            return df
        except Exception as e:
            logger.error("Error in EngineHPImputer.impute: %s", e)
            raise e


class EVFuelDisplacementImputer(ImputationStrategy):
    """Set EV brand displacement to 0 and missing fuel_type to Electric."""

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            ## EV brand displacement and fuel type
            is_ev = df["brand"].astype(str).str.lower().str.strip().isin(EV_BRANDS)

            ## Ev brands has zero displacemnt
            df.loc[is_ev & df["engine_displacement"].isna(), "engine_displacement"] = 0.0

            ## Impure missing fuel_type with Electric
            df.loc[is_ev & df["fuel_type"].isna(), "fuel_type"] = "Electric"
    
            return df
        except Exception as e:
            logger.error("Error in EVFuelDisplacementImputer.impute: %s", e)
            raise e


class BrandHPNeighborDisplacementImputer(ImputationStrategy):
    """Fill displacement using same-brand vehicles with matching horsepower."""

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            brand_hp_map = df.groupby(["brand", "engine_hp"])[
                "engine_displacement"
            ].transform("median")
            df["engine_displacement"] = df["engine_displacement"].fillna(brand_hp_map)

            # Fall back
            df["engine_displacement"] = df["engine_displacement"].fillna(
                df["engine_displacement"].median()
            )
            return df
        except Exception as e:
            logger.error("Error in BrandHPNeighborDisplacementImputer.impute: %s", e)
            raise e


class FuelTypeBrandModeImputer(ImputationStrategy):
    """Fill missing fuel_type with the brand's most common fuel type."""

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            ## Impute missing fuel_type with the brand's most common fuel type
            brand_mode = df.groupby("brand")["fuel_type"].transform(
                lambda x: x.fillna(
                    x.mode()[0] if not x.mode().empty else "Gasoline"
                )
            )

            mode_fuel_type = df['fuel_type'].mode()
            df["fuel_type"] = df["fuel_type"].fillna(brand_mode).fillna(mode_fuel_type)
            
            return df
        except Exception as e:
            logger.error("Error in FuelTypeBrandModeImputer.impute: %s", e)
            raise e


class DisplacementHPBinImputer(ImputationStrategy):
    """
    Fill remaining displacement NaNs using horsepower quantile bins.

    Tier 1: brand + hp_bin median
    Tier 2: global hp_bin median
    Tier 3: global median fallback
    """

    def __init__(self, n_bins: int = 15) -> None:
        try:
            self.n_bins = n_bins
        except Exception as e:
            logger.error("Error in DisplacementHPBinImputer.__init__: %s", e)
            raise e

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            ## EV brand displacement and fuel type
            is_ev = df["brand"].astype(str).str.lower().str.strip().isin(EV_BRANDS)
            df.loc[is_ev & df["engine_displacement"].isna(), "engine_displacement"] = 0.0

            if "engine_hp" not in df.columns or not df["engine_hp"].notna().any():
                df["engine_displacement"] = df["engine_displacement"].fillna(
                    df["engine_displacement"].median()
                )
                return df

            ## Binning with hp_bin
            df["hp_bin"] = pd.qcut(
                df["engine_hp"], q=self.n_bins, labels=False, duplicates="drop"
            )

            ## Brand + hp_bin median displacement
            brand_bin_map = df.groupby(["brand", "hp_bin"])[
                "engine_displacement"
            ].transform("median")
            df["engine_displacement"] = df["engine_displacement"].fillna(brand_bin_map)

            ## Global bin median displacement fallback          
            global_bin_map = df.groupby("hp_bin")["engine_displacement"].transform(
                "median"
            )
            df["engine_displacement"] = df["engine_displacement"].fillna(global_bin_map)

            ## Global median fallback   
            df["engine_displacement"] = df["engine_displacement"].fillna(
                df["engine_displacement"].median()
            )
            return df.drop(columns=["hp_bin"])
        except Exception as e:
            logger.error("Error in DisplacementHPBinImputer.impute: %s", e)
            raise e
        



## Just if we wnat to reutrn default pattern of cleanign pipeline
class MissingValueHandlerFactory:
    """Factory for assembling the default missing value handling strategy chain."""

    @staticmethod
    def create_default() -> list[ImputationStrategy]:
        try:
            return [
                EngineHPImputer(),
                EVFuelDisplacementImputer(),
                BrandHPNeighborDisplacementImputer(),
                FuelTypeBrandModeImputer(),
                DisplacementHPBinImputer(),
            ]
        except Exception as e:
            logger.error("Error in MissingValueHandlerFactory.create_default: %s", e)
            raise e


class MissingValueHandler(BaseTransformer):
    """Facade that runs imputation strategies in pipeline order."""

    def __init__(
        self, strategies: list[ImputationStrategy] | None = None
    ) -> None:
        try:
            self.strategies = strategies or MissingValueHandlerFactory.create_default()
        except Exception as e:
            logger.error("Error in MissingValueHandler.__init__: %s", e)
            raise e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            result = df.copy()
            for strategy in self.strategies:
                result = strategy.impute(result)
            logger.info("MissingValueHandler done. Shape: %s", result.shape)
            return result
        except Exception as e:
            logger.error("Error in MissingValueHandler.transform: %s", e)
            raise e
