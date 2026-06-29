import sys
import pandas as pd
from src.logger import logging
from src.exception import MyException
from src.components.data_transformers.feature_engineering import FeatureEngineer

def feature_engineer(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies feature engineering transformations to train and test datasets.

    Parameters:
        data (tuple[pd.DataFrame, pd.DataFrame]): A tuple containing train and test DataFrames.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Feature-engineered train and test DataFrames.
    """
    try:
        logging.info("Initiating feature engineering step...")
        train_df, test_df = data
        engineer = FeatureEngineer()
        
        logging.info("Fitting and transforming training data using FeatureEngineer...")
        train_df = engineer.fit_transform(train_df)
        
        logging.info("Transforming testing data using FeatureEngineer...")
        test_df = engineer.transform(test_df)
        
        logging.info("Feature engineering step completed successfully.")
        return train_df, test_df
    except Exception as e:
        logging.error("Exception occurred in feature_engineer step.")
        raise MyException(e, sys)
