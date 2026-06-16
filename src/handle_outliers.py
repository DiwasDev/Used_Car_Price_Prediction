"""
handle_outliers.py
------------------
Outlier detection, row removal, and skew correction.

Design patterns
  Strategy  – RowCapStrategy, LogTransformStrategy, IQRReportStrategy
  Facade    – OutlierHandler orchestrates configured strategies
  Factory   – OutlierHandlerFactory for common presets
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.base import BaseTransformer

logger = logging.getLogger(__name__)


class OutlierStrategy(ABC):
    """Strategy interface for outlier handling."""

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> OutlierStrategy:
        return self

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


class IQROutlierHandlerStrategy(OutlierStrategy):
    """Detect outliers using the IQR method and cap them at the lower and upper bounds."""

    def __init__(self, columns: list[str] | None = None) -> None:
        try:
            self.columns = columns or ["engine_hp", "engine_displacement"]
            self.capping_bounds ={}
        except Exception as e:
            logger.error("Error in IQROutlierHandlerStrategy.__init__: %s", e)
            raise e
        
    def fit(self, df: pd.DataFrame)-> None:
      
      """
    Calculates the lower and upper bounds for each column based on the IQR method.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns: 
     dict: A dictionary containing the lower and upper bounds for each column.
       """
    
      for col in self.columns:
          if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            # Save lower and upper bounds for future use
            self.capping_bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
            logger.info(
                "IQROutlierHandlerStrategy calculated lower and upper bounds for '%s' using IQR method.",
                col,
            )
            


    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            for col in self.columns:
                    lower_bound = self.capping_bounds[col]['lower']
                    upper_bound = self.capping_bounds[col]['upper']

                    df[col] = df[col].clip(lower= lower_bound, upper=upper_bound)

            logger.info("IQROutlierHandlerStrategy applied IQR method to %s columns.", len(self.columns))
            return df
        except Exception as e:
            logger.error("Error in IQROutlierHandlerStrategy.apply: %s", e)
            raise e




class LogTransformStrategy(OutlierStrategy):
    """Apply log1p to reduce right-skew on specified columns."""

    def __init__(self, columns: list[str] | None = None) -> None:
        try:
            self.columns = columns or ["price_usd", "mileage_num"]
        except Exception as e:
            logger.error("Error in LogTransformStrategy.__init__: %s", e)
            raise e
        
    def fit(self, df: pd.DataFrame) -> None:
        """No-op for LogTransformStrategy."""
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            for col in self.columns:
                if col in df.columns:
                    df[col] = np.log1p(df[col])
            return df
        except Exception as e:
            logger.error("Error in LogTransformStrategy.apply: %s", e)
            raise e


class OutlierHandlerFactory:
    """Factory for common outlier-handler configurations."""
    @staticmethod
    def create_default() -> list[OutlierStrategy]:
        """Creates a default list of outlier strategies."""
        try:
            return [
             LogTransformStrategy(columns=["price_usd", "mileage_num"]),
             IQROutlierHandlerStrategy(columns=["engine_hp", "engine_displacement"]),
         ]
        except Exception as e:
            logger.error("Error in OutlierHandlerFactory.create_default: %s", e)
            raise e

#Context:Facade pattern here
class OutlierHandler(BaseTransformer):
    """Facade that applies outlier strategies in order."""

    def __init__(self, strategies: list[OutlierStrategy] | None = None) -> None:
        try:
            self.strategies = strategies or OutlierHandlerFactory.create_default()
        except Exception as e:
            logger.error("Error in OutlierHandler.__init__: %s", e)
            raise e
        
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the outlier strategies in order."""
        try:
            for strategy in self.strategies:
                strategy.fit(df)
            return df
        except Exception as e:
            logger.error("Error in OutlierHandler.fit: %s", e)
            raise e
        

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            result = df.copy()
            for strategy in self.strategies:
                result = strategy.apply(result)
            return result
        except Exception as e:
            logger.error("Error in OutlierHandler.transform: %s", e)
            raise e


# class OutlierHandlerFactory:
#     """Factory for common outlier-handler configurations."""
#     @staticmethod
#     def log_transform_handler(
#         columns: list[str] | None = None,
#     ) -> OutlierHandler:
#         try:
#             return OutlierHandler(strategies=[LogTransformStrategy(columns=columns)])
#         except Exception as e:
#             logger.error("Error in OutlierHandlerFactory.log_transform_handler: %s", e)
#             raise e

#     @staticmethod
#     def iqr_outlier_handler(
#         columns: list[str] | None = None,
#     ) -> OutlierHandler:
#         try:
#             return OutlierHandler(
#                 strategies=[IQROutlierHandlerStrategy(columns=columns)]
#             )
#         except Exception as e:
#             logger.error("Error in OutlierHandlerFactory.iqr_outlier_handler: %s", e)
#             raise e


# if __name__ == "__main__":
#     df = pd.read_csv("/home/divas/ml/logistics_project/data/raw/used_cars.csv")
#     outlier_handler = OutlierHandlerFactory.iqr_outlier_handler()
#     df = outlier_handler.transform(df)
#     print(df.head())