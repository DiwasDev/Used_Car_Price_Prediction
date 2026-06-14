"""
data_cleaning.py
----------------
Fixes raw data problems identified during EDA:

  Problem 1 – price is a string:  "$10,300" → 10300.0  (float, new col price_usd)
  Problem 2 – milage is a string: "51,000 mi." → 51000.0 (float, new col mileage_num)
  Problem 3 – fuel_type has junk: "–" and "not supported" → NaN (handled later by imputer)
  Problem 4 – clean_title only has "Yes"; NaN means no clean title → binary flag
  Problem 5 – outlier prices)
  Problem 6 – outlier mileage values

Design Pattern: Strategy
  Each column-level fix is a small private method.
  The public transform() orchestrates them in order.
  Swapping or skipping a fix = one line change.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from src.base import BaseTransformer

class DataCleaner(BaseTransformer):
    """
    Converts raw string columns to numeric, flags junk values,
    and caps statistical outliers.

    Parameters
    ----------
    price_upper_cap : float
        Rows with price above this are dropped as outliers.
    price_lower_cap : float
        Rows with price below this are dropped (likely data errors).
    milage_upper_cap : float
        Rows with mileage above this are dropped.
    """

    JUNK_FUEL_VALUES = {"–", "not supported", ""}

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._clean_price(df)
        df = self._clean_mileage(df)
        df = self._clean_fuel_type(df)
        df = self._clean_clean_title(df)
        # df = self._remove_price_outliers(df)
        # df = self._remove_mileage_outliers(df)
        logger.info("DataCleaner done. Shape after cleaning: %s", df.shape)
        return df

    # ------------------------------------------------------------------ #
    # Private helpers (Strategy methods)                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_price(df: pd.DataFrame) -> pd.DataFrame:
        """'$10,300' → 10300.0  stored in 'price_usd'. Drop original."""
        df["price_usd"] = (
            df["price"]
            .str.replace(r"[$,]", "", regex=True)
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
        )
        df = df.drop(columns=["price"])
        logger.debug("price_usd: %d non-null", df["price_usd"].notna().sum())
        return df

    @staticmethod
    def _clean_mileage(df: pd.DataFrame) -> pd.DataFrame:
        """'51,000 mi.' → 51000.0  stored in 'mileage_num'. Drop original."""
        df["mileage_num"] = (
            df["milage"]
            .str.replace(r"[,]", "", regex=True)   # remove commas
            .str.replace(r"\s*mi\.\s*$", "", regex=True)  # remove ' mi.'
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")
        )
        df = df.drop(columns=["milage"])
        logger.debug("mileage_num: %d non-null", df["mileage_num"].notna().sum())
        return df

    def _clean_fuel_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace known junk tokens with NaN so the imputer can fill them."""
        df["fuel_type"] = df["fuel_type"].apply(
            lambda x: None if str(x).strip() in self.JUNK_FUEL_VALUES else x
        )
        return df

    @staticmethod
    def _clean_clean_title(df: pd.DataFrame) -> pd.DataFrame:
        """
        'clean_title' has only "Yes" and NaN.
        NaN = no clean title → encode as binary 0/1.
        Column renamed to 'has_clean_title'.
        """
        df["has_clean_title"] = df["clean_title"].map({"Yes": 1}).fillna(0).astype(int)
        df = df.drop(columns=["clean_title"])
        return df

    # def _remove_price_outliers(self, df: pd.DataFrame) -> pd.DataFrame:


    #     before = len(df)

    #     ## Find lower and upper caps using IQR method
    #     q1 = df["price_usd"].quantile(0.25)
    #     q3 = df["price_usd"].quantile(0.75)
    #     iqr = q3 - q1
    #     price_lower_cap = q1 - 1.5 * iqr
    #     price_upper_cap = q3 + 1.5 * iqr
        
    #     df['price_usd'] = df['price_usd'].clip(lower=price_lower_cap, upper=price_upper_cap)
    #     logger.info(
    #         "Price outlier removal: %d rows dropped (price outside [%s, %s])",
    #         before - len(df),
    #         price_lower_cap,
    #         price_upper_cap,
    #     )
    #     return df

    # def _remove_mileage_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
    #     before = len(df)

    #     ## Find lower and upper caps using IQR method
    #     q1 = df["mileage_num"].quantile(0.25)
    #     q3 = df["mileage_num"].quantile(0.75)
    #     iqr = q3 - q1
    #     mileage_lower_cap = q1 - 1.5 * iqr
    #     mileage_upper_cap = q3 + 1.5 * iqr
    #     df['mileage_num'] = df['mileage_num'].clip(lower=mileage_lower_cap, upper=mileage_upper_cap)
    #     logger.info(
    #         "Mileage outlier removal: %d rows dropped (mileage > %s)",
    #         before - len(df),
    #         mileage_upper_cap,
    #     )
        
    #     return df
