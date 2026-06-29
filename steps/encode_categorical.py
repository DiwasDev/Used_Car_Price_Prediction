import sys
import pandas as pd
from src.logger import logging
from src.exception import MyException
from src.components.data_transformers.encoder import CategoricalEncoder

def encode_categorical(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies target and one-hot encoding to the train and test datasets using CategoricalEncoder.

    Parameters:
        data (tuple[pd.DataFrame, pd.DataFrame]): A tuple containing train and test DataFrames.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Encoded train and test DataFrames.
    """
    try:
        logging.info("Initiating categorical encoding step...")
        train_df, test_df = data
        encoder = CategoricalEncoder()
        
        logging.info("Fitting and transforming training data using CategoricalEncoder...")
        train_df = encoder.fit_transform(train_df)
        
        logging.info("Transforming testing data using CategoricalEncoder...")
        test_df = encoder.transform(test_df)
        
        logging.info("Categorical encoding step completed successfully.")
        return train_df, test_df
    except Exception as e:
        logging.error("Exception occurred in encode_categorical step.")
        raise MyException(e, sys)