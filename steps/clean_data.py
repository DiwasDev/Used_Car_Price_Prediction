import pandas as pd
from src.data_cleaning import DataCleaner

def clean_data(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleans train and test datasets using DataCleaner."""
    train_df, test_df = data
    cleaner = DataCleaner()
    train_df = cleaner.fit_transform(train_df)
    test_df = cleaner.transform(test_df)
    return train_df, test_df
