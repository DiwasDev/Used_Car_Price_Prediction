import os
import sys
from src.exception import MyException
from src.utils.main_utils import load_object

class Proj1Estimator:
    def __init__(self, bucket_name: str, model_path: str):
        self.bucket_name = bucket_name
        self.model_path = model_path

    def is_model_present(self, model_path: str) -> bool:
        # Check if the file exists locally or returns False by default
        return os.path.exists(model_path)

    def predict(self, df):
        try:
            model = load_object(self.model_path)
            return model.predict(df)
        except Exception as e:
            raise MyException(e, sys)
