import sys
import pandas as pd
from src.logger import logging
from src.exception import MyException
from src.components.data_transformers.handle_missing import MissingValueHandler

def handle_missing(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputes missing values in train and test datasets using MissingValueHandler.

    Parameters:
        data (tuple[pd.DataFrame, pd.DataFrame]): A tuple containing train and test DataFrames.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Imputed train and test DataFrames.
    """
    try:
        logging.info("Initiating missing values handling step...")
        train_df, test_df = data
        missing_handler = MissingValueHandler()
        
        logging.info("Fitting and transforming training data using MissingValueHandler...")
        train_df = missing_handler.fit_transform(train_df)
        
        logging.info("Transforming testing data using MissingValueHandler...")
        test_df = missing_handler.transform(test_df)
        
        logging.info("Missing values handling step completed successfully.")
        return train_df, test_df
    except Exception as e:
        logging.error("Exception occurred in handle_missing step.")
        raise MyException(e, sys)
