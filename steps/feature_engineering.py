import pandas as pd
from src.components.data_transformers.feature_engineering import FeatureEngineer

def feature_engineer(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies feature engineering to train and test datasets."""
    train_df, test_df = data
    engineer = FeatureEngineer()
    train_df = engineer.fit_transform(train_df)
    test_df = engineer.transform(test_df)
    return train_df, test_df
