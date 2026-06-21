import sys
import pandas as pd
from src.exception import MyException

class MyModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        Wrapper model that bundles the preprocessing pipeline and the trained estimator.
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, df: pd.DataFrame):
        try:
            transformed_df = self.preprocessing_object.transform(df)
            return self.trained_model_object.predict(transformed_df)
        except Exception as e:
            raise MyException(e, sys)

    def __repr__(self):
        return f"MyModel(preprocessing_object={self.preprocessing_object}, trained_model_object={self.trained_model_object})"
