import sys
import pandas as pd
from src.logger import logging
from src.exception import MyException
from src.components.data_transformers.data_cleaning import DataCleaner

def clean_data(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans train and test datasets by removing characters and converting types using DataCleaner.

    Parameters:
        data (tuple[pd.DataFrame, pd.DataFrame]): A tuple containing train and test DataFrames.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Cleaned train and test DataFrames.
    """
    try:
        logging.info("Initiating data cleaning step...")
        train_df, test_df = data
        cleaner = DataCleaner()
        
        logging.info("Fitting and transforming training data using DataCleaner...")
        train_df = cleaner.fit_transform(train_df)
        
        logging.info("Transforming testing data using DataCleaner...")
        test_df = cleaner.transform(test_df)
        
        logging.info("Data cleaning step completed successfully.")
        return train_df, test_df
    except Exception as e:
        logging.error("Exception occurred in clean_data step.")
        raise MyException(e, sys)
