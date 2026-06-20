import sys
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from src.exception import MyException
from src.logger import logging

class DataSplitStrategy(ABC):
    """
    Abstract strategy class for data splitting.
    """

    @abstractmethod
    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Abstract method to split a DataFrame.

        Parameters:
            df: pd.DataFrame: Input DataFrame.

        Returns:
            split_data: tuple[pd.DataFrame, pd.DataFrame]: Split train and test DataFrames.
        """
        pass


class TrainTestSplitStrategy(DataSplitStrategy):
    """
    Concrete split strategy using train_test_split from sklearn.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Initializes train-test split strategy.

        Parameters:
            test_size: float: Split ratio for testing data.
            random_state: int: Random state for reproducibility.

        Returns:
            None
        """
        try:
            self.test_size = test_size
            self.random_state = random_state
        except Exception as e:
            raise MyException(e, sys)

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the input DataFrame into train and test sets.

        Parameters:
            df: pd.DataFrame: Input DataFrame to split.

        Returns:
            split_data: tuple[pd.DataFrame, pd.DataFrame]: Tuple containing train and test DataFrames.
        """
        try:
            logging.info(f"Splitting data with test_size={self.test_size} and random_state={self.random_state}")
            train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
            return train_df, test_df
        except Exception as e:
            raise MyException(e, sys)
