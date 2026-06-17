"""
data_split.py
-------------
Strategy pattern for splitting data into train and test sets.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitStrategy(ABC):
    """Abstract base strategy for data splitting."""
    
    @abstractmethod
    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train and test splits."""
        ...


class TrainTestSplitStrategy(DataSplitStrategy):
    """Concrete strategy implementing standard random train/test split."""

    def __init__(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> None:
        try:
            self.test_size = test_size
            self.random_state = random_state
        except Exception as e:
            logger.error("Error in TrainTestSplitStrategy.__init__: %s", e)
            raise e

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            return train_df, test_df
        except Exception as e:
            logger.error("Error in TrainTestSplitStrategy.split: %s", e)
            raise e
