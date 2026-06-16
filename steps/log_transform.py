import pandas as pd
from src.handle_outliers import OutlierHandlerFactory

def log_transform(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies log transformation to train and test datasets."""
    train_df, test_df = data
    log_transformer = OutlierHandlerFactory.log_transform_handler()
    train_df = log_transformer.fit_transform(train_df)
    test_df = log_transformer.transform(test_df)
    return train_df, test_df
