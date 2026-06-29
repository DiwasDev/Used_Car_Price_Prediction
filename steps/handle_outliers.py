import sys
import pandas as pd
from src.logger import logging
from src.exception import MyException
from src.components.data_transformers.handle_outliers import OutlierHandler

def handle_outliers(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Caps outliers using IQR/Log methods for specified columns in train and test datasets.

    Parameters:
        data (tuple[pd.DataFrame, pd.DataFrame]): A tuple containing train and test DataFrames.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Outlier-handled train and test DataFrames.
    """
    try:
        logging.info("Initiating outliers handling step...")
        train_df, test_df = data
        outlier_handler = OutlierHandler()

        logging.info("Fitting outlier handler parameters on training data...")
        outlier_handler.fit(train_df)

        logging.info("Transforming training data using OutlierHandler...")
        train_df = outlier_handler.transform(train_df)
        
        logging.info("Transforming testing data using OutlierHandler...")
        test_df = outlier_handler.transform(test_df)

        logging.info("Outliers handling step completed successfully.")
        return train_df, test_df
    except Exception as e:
        logging.error("Exception occurred in handle_outliers step.")
        raise MyException(e, sys)