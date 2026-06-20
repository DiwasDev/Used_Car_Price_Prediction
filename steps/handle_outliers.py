import pandas as pd
from src.handle_outliers import OutlierHandler
from src.handle_outliers import LogTransformStrategy
from src.handle_outliers import IQROutlierHandlerStrategy

def handle_outliers(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Caps outliers using IQR method for specified columns."""
    train_df, test_df = data
    outlier_handler = OutlierHandler()

    # Fitting the parameters
    outlier_handler.fit_transform(train_df)

    # Applying the changes
    train_df = outlier_handler.transform(train_df)
    test_df = outlier_handler.transform(test_df)

    return train_df, test_df