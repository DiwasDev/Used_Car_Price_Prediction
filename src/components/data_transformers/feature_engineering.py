"""
feature_engineering.py
----------------------
Domain-specific feature extraction from raw vehicle columns.

Design patterns
  Strategy  – each FeatureStep applies one transformation
  Template  – FeatureEngineer runs steps in a fixed order
  Facade    – FeatureEngineer exposes a single transform() entry point
"""

from __future__ import annotations

from src.logger import logging
import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import sys
import os

# Add project root to sys.path to allow direct script execution from src/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.components.base import BaseTransformer

EV_BRANDS = frozenset({"tesla", "rivian", "lucid", "polestar"})
COLOR_COLUMNS = ("int_col", "ext_col")
JUNK_COLOR_TOKENS = frozenset({"γçô", "nan", "other", ""})
JUNK_TRANSMISSION_TOKENS = frozenset(
    {"γçô", "scheduled for or in production", "nan", ""}
)
JUNK_ENGINE_TOKENS = frozenset({"ΓÇô", "nan", ""})


class FeatureStep(ABC):
    """Strategy interface for a single feature-engineering step."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class ModelFeatureStep(FeatureStep):
    """Extract binary flags from model name, then drop model column."""

    SPORT_PATTERN = r"Sport|S-Line|M Sport"
    PREMIUM_PATTERN = r"Premium|Platinum|Limited"
    AWD_PATTERN = r"4WD|xDrive|Quattro|AWD"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df["is_sport"] = (
                df["model"].str.contains(self.SPORT_PATTERN, case=False).astype(int)
            )
            df["is_premium"] = (
                df["model"].str.contains(self.PREMIUM_PATTERN, case=False).astype(int)
            )
            df["is_4WD_AWD"] = (
                df["model"].str.contains(self.AWD_PATTERN, case=False).astype(int)
            )
            return df.drop(columns=["model"])
        except Exception as e:
            logging.error("Error in ModelFeatureStep.apply: %s", e)
            raise e


class ColorMappingStep(FeatureStep):
    """Normalize and bucket interior/exterior color strings."""

    _CONDITIONS = [
        r"black|blk|ebony|nero|charcoal|obsidian|beluga|amg|graphite|carbon",
        r"beige|tan|parchment|sandstone|canberra|shara|macchiato|almond|shale|cashmere|linen|ivory|silk",
        r"gray|grey|slate|pewter|titan|boulder|ash|platinum|silver|galvanized|gideon",
        r"brown|walnut|espresso|caramel|cappuccino|nougat|sarder|mesa|tupelo|mocha|saddle|auburn|amber|brandy|mountain|aragon|chestnut|cocoa|dune|roast",
        r"red|hotspur|rioja|pimento|magma|garnet|chateau|adrenaline",
        r"blue|navy|cobalt|rhapsody|charles|mistral|porpoise",
        r"orange|sakhir|kyalami|giallo|taurus|yellow",
        r"white|ice|pearl|grace|cloud|whisper|bianco|polar",
    ]
    _CHOICES = [
        "Black", "Beige", "Gray", "Brown", "Red", "Blue", "Orange", "White", None
    ]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            ## Normalize and bucket interior/exterior color strings
            for col in COLOR_COLUMNS:
                clean = (
                    df[col]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .str.replace(r"[.\-]", "", regex=True)
                )

                ## Remove junk color and NaNs tokens as Unknowns 
                junk_mask = clean.isin(JUNK_COLOR_TOKENS) | clean.isna()
                clean = pd.Series(
                    np.where(junk_mask, None, clean), index=df.index
                )

                ## Categorize colors using regex conditions
                # FIXED: Fill NaN values in conditions with False to avoid np.select errors
                conditions = [
                    clean.str.contains(p, regex=True).fillna(False) for p in self._CONDITIONS
                ] + [clean.isna()]
   

                # Find most frequent color
                mode_color = clean.mode()

                ## Assign colors to buckets
                df[col] = np.select(conditions, self._CHOICES, default=None)
                # FIXED: Properly assign the fillna result back to df[col]
                if len(mode_color) > 0:
                    df[col] = df[col].fillna(mode_color[0])
                
            return df
        except Exception as e:
            logging.error("Error in ColorMappingStep.apply: %s", e)
            raise e


class AccidentMappingStep(FeatureStep):
    """Map accident text labels to a binary flag."""

    ACCIDENT_MAP = {
        "no accidents": 0,
        "At least 1 accident or damage reported": 1,
    }

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df["accident"] = df["accident"].map(self.ACCIDENT_MAP).fillna(0)
            return df
        except Exception as e:
            logging.error("Error in AccidentMappingStep.apply: %s", e)
            raise e


class TransmissionFeatureStep(FeatureStep):
    """Extract transmission_type from transmission."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            # Normalize and bucket transmission strings, map junk tokens and NaNs as Unknowns
            raw = df["transmission"].astype(str).str.lower().str.strip()
            raw = pd.Series(
                np.where(raw.isin(JUNK_TRANSMISSION_TOKENS),None, raw),
                index=df.index,
            )

            # FIXED: Fill NaN values in conditions with False
            is_cvt = raw.str.contains(r"cvt|variable").fillna(False)
            is_manual = raw.str.contains(r"m/t|manual|mt").fillna(False)
            is_auto = raw.str.contains(
                r"a/t|automatic|auto|at|pdk|dct|steptronic|tronic|shift"
            ).fillna(False)
            
            mode_transmisson_type = raw.mode()
            df["transmission_type"] = np.select(
                [is_cvt, is_manual, is_auto, raw.isna()],
                ["CVT", "Manual", "Automatic", None],
                default="Automatic" if len(mode_transmisson_type) == 0 else mode_transmisson_type[0],
            )

                # # Extract transmission gears
                # extracted = raw.str.extract(r"(\d+)\s*(?:-speed|spd|speed|gear)")
                # fallback = raw.str.extract(r"\b(\d+)\b")
                # df["transmission_gears"] = pd.to_numeric(
                #     extracted[0].fillna(fallback[0]), errors="coerce"
                # )

            # Set transmission gears to 0 for CVT
            # df.loc[df["transmission_type"] == "CVT", "transmission_gears"] = 0

            # Drop the raw transmission column
            return df.drop(columns=["transmission"])
        except Exception as e:
            logging.error("Error in TransmissionFeatureStep.apply: %s", e)
            raise e


class EngineSpecFeatureStep(FeatureStep):
    """Extract HP, displacement, cylinders, and engine-derived fuel type."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()

            # Normalize and bucket engine strings, map junk tokens and NaNs as Unknowns
            raw = df["engine"].astype(str).str.strip()
            junk_mask = raw.isin(JUNK_ENGINE_TOKENS) | raw.isna()
            raw = pd.Series(np.where(junk_mask, None, raw), index=df.index)

            # Extract engine hp    
            df["engine_hp"] = pd.to_numeric(
                raw.str.extract(r"(\d+(?:\.\d+)?)\s*HP", flags=re.IGNORECASE)[0],
                errors="coerce",
            )
            # Extract engine displacement    
            df["engine_displacement"] = pd.to_numeric(
                raw.str.extract(
                    r"(\d+(?:\.\d+)?)\s*(?:L|Liter)", flags=re.IGNORECASE
                )[0],
                errors="coerce",
            )

            # # Extract engine cylinders
            # cyl_word = raw.str.extract(r"(\d+)\s*Cylinder", flags=re.IGNORECASE)[0]
            # cyl_short = raw.str.extract(
            #     r"\b(?:V|I|H|V-)(\d+)\b", flags=re.IGNORECASE
            # )[0]    
            # cyl_text = np.where(
            #     raw.str.contains(r"Straight 6|Flat 6", flags=re.IGNORECASE), "6", None
            # )

            # # Combine cylinder extracts
            # df["engine_cylinders"] = pd.to_numeric(
            #     cyl_word.fillna(cyl_short).fillna(
            #         pd.Series(cyl_text, index=df.index)
            #     ),
            #     errors="coerce",
            # )

            # Extract fuel type from engine
            # FIXED: Fill NaN values in conditions with False
            is_electric = raw.str.contains(
                r"Electric Motor|Electric Fuel|Electric", flags=re.IGNORECASE
            ).fillna(False)
            is_hybrid = raw.str.contains(r"Hybrid|Plug-In", flags=re.IGNORECASE).fillna(False)
            is_diesel = raw.str.contains(r"Diesel", flags=re.IGNORECASE).fillna(False)
            is_flex = raw.str.contains(r"Flex Fuel|Flexible", flags=re.IGNORECASE).fillna(False)
            is_gasoline = raw.str.contains(r"Gasoline|Gas|GDI|MPFI", flags=re.IGNORECASE).fillna(False)

            df["fuel_type_engine"] = np.select(
                [
                    is_electric & ~is_hybrid,
                    is_hybrid,
                    is_diesel,
                    is_flex,
                    is_gasoline,
                    raw.isna(),
                ],
                ["Electric", "Hybrid", "Diesel", "Flex Fuel", "Gasoline", None],
                default=None,
            )

            # Set engine displacement and cylinders to 0 for electric vehicles
            df.loc[
                df["fuel_type_engine"] == "Electric",
                ["engine_displacement"],
            ] = 0
            return df.drop(columns=["engine"])  
        except Exception as e:
            logging.error("Error in EngineSpecFeatureStep.apply: %s", e)
            raise e


class FuelTypeMergeStep(FeatureStep):
    """Back-fill fuel_type from engine-derived fuel_type_engine, then drop helper."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            df["fuel_type"] = df["fuel_type"].fillna(df["fuel_type_engine"])
            return df.drop(columns=["fuel_type_engine"])
        except Exception as e:
            logging.error("Error in FuelTypeMergeStep.apply: %s", e)
            raise e


# class DropRedundantColumnsStep(FeatureStep):
#     """Remove columns that are redundant or dropped before modelling."""

#     COLUMNS = ("engine_cylinders", "transmission_gears")

#     def apply(self, df: pd.DataFrame) -> pd.DataFrame:
#         try:
#             existing = [c for c in self.COLUMNS if c in df.columns]
#             return df.drop(columns=existing) if existing else df
#         except Exception as e:
#             logging.error("Error in DropRedundantColumnsStep.apply: %s", e)
#             raise e




## Just if we wnat to reutrn default pattern of feature engineering pipeline
class FeatureEngineerFactory:
    """Factory for assembling the default feature engineering strategy chain."""

    @staticmethod
    def create_default() -> list[FeatureStep]:
        try:
            return [
                ModelFeatureStep(),
                ColorMappingStep(),
                AccidentMappingStep(),
                TransmissionFeatureStep(),
                EngineSpecFeatureStep(),
                FuelTypeMergeStep(),
            ]
        except Exception as e:
            logging.error("Error in FeatureEngineerFactory.create_default: %s", e)
            raise e


class FeatureEngineer(BaseTransformer):
    """Facade that runs feature-engineering steps in pipeline order."""

    def __init__(self, steps: list[FeatureStep] | None = None) -> None:
        try:
            self.steps = steps or FeatureEngineerFactory.create_default()
        except Exception as e:
            logging.error("Error in FeatureEngineer.__init__: %s", e)
            raise e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            result = df.copy()
            for step in self.steps:
                result = step.apply(result)
            logging.info("FeatureEngineer done. Shape: %s", result.shape)
            return result
        except Exception as e:
            logging.error("Error in FeatureEngineer.transform: %s", e)
            raise e


if __name__ == "__main__":
    df = pd.read_csv("/home/divas/ml/logistics_project/data/raw/used_cars.csv")
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(df)
    df = feature_engineer.transform(df)
    print(df.head())
