import pandas as pd
from src.encoder import CategoricalEncoder, OneHotEncodingStrategy, TargetEncodingStrategy

# class SafeOneHotEncodingStrategy(OneHotEncodingStrategy):
#     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
#         if self._encoder is None:
#             raise RuntimeError("SafeOneHotEncodingStrategy must be fitted before transform.")
#         df_copy = df.copy()
#         for i, col in enumerate(self.columns):
#             valid_cats = set(self._encoder.categories_[i])
#             df_copy[col] = df_copy[col].apply(lambda x: x if x in valid_cats else self._encoder.categories_[i][0])
#         return super().transform(df_copy)

def encode_categorical(data: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies target and one-hot encoding to the train and test datasets."""
    train_df, test_df = data
    encoder  = CategoricalEncoder()
    train_df = encoder.fit_transform(train_df)
    test_df = encoder.transform(test_df)
    return train_df, test_df