import pandas as pd
from src.handle_missing import MissingValueHandler

def handle_missing(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imputes missing values in train and test datasets."""
    train_df, test_df = data
    missing_handler = MissingValueHandler()
    train_df = missing_handler.fit_transform(train_df)
    test_df = missing_handler.transform(test_df)
    return train_df, test_df
