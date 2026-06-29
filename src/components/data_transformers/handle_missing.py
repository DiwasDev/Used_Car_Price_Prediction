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

from src.logger import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.components.base import BaseTransformer
from src.components.data_transformers.feature_engineering import EV_BRANDS


class ImputationStrategy(ABC):
    """Strategy interface for a single imputation operation."""

    def fit(self, df: pd.DataFrame) -> ImputationStrategy:
        return self

    @abstractmethod
    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class EngineHPImputer(ImputationStrategy):
    """Fill missing engine_hp with brand median, then global median."""

    def __init__(self) -> None:
        self.brand_medians_ = {}
        self.global_median_ = np.nan

    def fit(self, df: pd.DataFrame) -> EngineHPImputer:
        try:
            self.brand_medians_ = df.groupby("brand")["engine_hp"].median().to_dict()
            self.global_median_ = float(df["engine_hp"].median())
            return self
        except Exception as e:
            logging.error("Error in EngineHPImputer.fit: %s", e)
            raise e

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            # Map brand to its median hp
            brand_median_series = df["brand"].map(self.brand_medians_)
            df["engine_hp"] = df["engine_hp"].fillna(brand_median_series).fillna(self.global_median_)
            return df
        except Exception as e:
            logging.error("Error in EngineHPImputer.impute: %s", e)
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
            logging.error("Error in EVFuelDisplacementImputer.impute: %s", e)
            raise e


class BrandHPNeighborDisplacementImputer(ImputationStrategy):
    """Fill displacement using same-brand vehicles with matching horsepower."""

    def __init__(self) -> None:
        self.brand_hp_medians_ = {}
        self.global_median_ = np.nan

    def fit(self, df: pd.DataFrame) -> BrandHPNeighborDisplacementImputer:
        try:
            self.brand_hp_medians_ = df.groupby(["brand", "engine_hp"])["engine_displacement"].median().to_dict()
            self.global_median_ = float(df["engine_displacement"].median())
            return self
        except Exception as e:
            logging.error("Error in BrandHPNeighborDisplacementImputer.fit: %s", e)
            raise e

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            # Map brand and hp to median displacement
            keys = list(zip(df["brand"], df["engine_hp"]))
            mapped_displacement = pd.Series([self.brand_hp_medians_.get(k, np.nan) for k in keys], index=df.index)
            df["engine_displacement"] = df["engine_displacement"].fillna(mapped_displacement).fillna(self.global_median_)
            return df
        except Exception as e:
            logging.error("Error in BrandHPNeighborDisplacementImputer.impute: %s", e)
            raise e


class FuelTypeBrandModeImputer(ImputationStrategy):
    """Fill missing fuel_type with the brand's most common fuel type."""

    def __init__(self) -> None:
        self.brand_modes_ = {}
        self.global_mode_ = "Gasoline"

    def fit(self, df: pd.DataFrame) -> FuelTypeBrandModeImputer:
        try:
            self.brand_modes_ = {}
            for brand, group in df.groupby("brand"):
                modes = group["fuel_type"].mode()
                if not modes.empty:
                    self.brand_modes_[brand] = modes.iloc[0]
            global_modes = df["fuel_type"].mode()
            self.global_mode_ = global_modes.iloc[0] if not global_modes.empty else "Gasoline"
            return self
        except Exception as e:
            logging.error("Error in FuelTypeBrandModeImputer.fit: %s", e)
            raise e

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            brand_mode_series = df["brand"].map(self.brand_modes_)
            df["fuel_type"] = df["fuel_type"].fillna(brand_mode_series).fillna(self.global_mode_)
            return df
        except Exception as e:
            logging.error("Error in FuelTypeBrandModeImputer.impute: %s", e)
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
            self.hp_bin_edges_ = None
            self.brand_bin_medians_ = {}
            self.bin_medians_ = {}
            self.global_median_ = np.nan
        except Exception as e:
            logging.error("Error in DisplacementHPBinImputer.__init__: %s", e)
            raise e

    def fit(self, df: pd.DataFrame) -> DisplacementHPBinImputer:
        try:
            self.global_median_ = float(df["engine_displacement"].median())
            if "engine_hp" in df.columns and df["engine_hp"].notna().any():
                non_null_hp = df["engine_hp"].dropna()
                if len(non_null_hp) > 0:
                    _, bin_edges = pd.qcut(non_null_hp, q=self.n_bins, retbins=True, duplicates="drop")
                    self.hp_bin_edges_ = bin_edges
                    
                    df_copy = df.copy()
                    df_copy["hp_bin"] = pd.cut(df_copy["engine_hp"], bins=self.hp_bin_edges_, labels=False, include_lowest=True)
                    self.brand_bin_medians_ = df_copy.groupby(["brand", "hp_bin"])["engine_displacement"].median().to_dict()
                    self.bin_medians_ = df_copy.groupby("hp_bin")["engine_displacement"].median().to_dict()
            return self
        except Exception as e:
            logging.error("Error in DisplacementHPBinImputer.fit: %s", e)
            raise e

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            ## EV brand displacement and fuel type
            is_ev = df["brand"].astype(str).str.lower().str.strip().isin(EV_BRANDS)
            df.loc[is_ev & df["engine_displacement"].isna(), "engine_displacement"] = 0.0

            if self.hp_bin_edges_ is None or "engine_hp" not in df.columns or not df["engine_hp"].notna().any():
                df["engine_displacement"] = df["engine_displacement"].fillna(self.global_median_)
                return df

            ## Binning with hp_bin
            df["hp_bin"] = pd.cut(df["engine_hp"], bins=self.hp_bin_edges_, labels=False, include_lowest=True)

            ## Brand + hp_bin median displacement
            keys = list(zip(df["brand"], df["hp_bin"]))
            brand_bin_displacement = pd.Series([self.brand_bin_medians_.get(k, np.nan) for k in keys], index=df.index)
            df["engine_displacement"] = df["engine_displacement"].fillna(brand_bin_displacement)

            ## Global bin median displacement fallback          
            global_bin_displacement = df["hp_bin"].map(self.bin_medians_)
            df["engine_displacement"] = df["engine_displacement"].fillna(global_bin_displacement)

            ## Global median fallback   
            df["engine_displacement"] = df["engine_displacement"].fillna(self.global_median_)
            return df.drop(columns=["hp_bin"])
        except Exception as e:
            logging.error("Error in DisplacementHPBinImputer.impute: %s", e)
            raise e


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
            logging.error("Error in MissingValueHandlerFactory.create_default: %s", e)
            raise e


class MissingValueHandler(BaseTransformer):
    """Facade that runs imputation strategies in pipeline order."""

    def __init__(
        self, strategies: list[ImputationStrategy] | None = None
    ) -> None:
        try:
            self.strategies = strategies or MissingValueHandlerFactory.create_default()
        except Exception as e:
            logging.error("Error in MissingValueHandler.__init__: %s", e)
            raise e

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> MissingValueHandler:
        try:
            current = df.copy()
            for strategy in self.strategies:
                strategy.fit(current)
                current = strategy.impute(current)
            return self
        except Exception as e:
            logging.error("Error in MissingValueHandler.fit: %s", e)
            raise e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            result = df.copy()
            for strategy in self.strategies:
                result = strategy.impute(result)
            logging.info("MissingValueHandler done. Shape: %s", result.shape)
            return result
        except Exception as e:
            logging.error("Error in MissingValueHandler.transform: %s", e)
            raise e
