import pandas as pd
from src.data_splitter import TrainTestSplitStrategy

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the input DataFrame into train and test sets."""
    splitter = TrainTestSplitStrategy(test_size=test_size, random_state=random_state)
    train_df, test_df = splitter.split(df)
    return train_df, test_df
