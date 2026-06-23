import sys
import os
import pandas as pd
import numpy as np
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object

class VehicleData:
    def __init__(self,
                 brand: str,
                 model: str,
                 model_year: int,
                 milage: str,
                 fuel_type: str,
                 engine: str,
                 transmission: str,
                 ext_col: str,
                 int_col: str,
                 accident: str,
                 clean_title: str):
        self.brand = brand
        self.model = model
        self.model_year = model_year
        self.milage = milage
        self.fuel_type = fuel_type
        self.engine = engine
        self.transmission = transmission
        self.ext_col = ext_col
        self.int_col = int_col
        self.accident = accident
        self.clean_title = clean_title

    def get_vehicle_input_data_frame(self) -> pd.DataFrame:
        try:
            input_dict = {
                "brand": [self.brand],
                "model": [self.model],
                "model_year": [int(self.model_year)],
                "milage": [self.milage],
                "fuel_type": [self.fuel_type],
                "engine": [self.engine],
                "transmission": [self.transmission],
                "ext_col": [self.ext_col],
                "int_col": [self.int_col],
                "accident": [self.accident],
                "clean_title": [self.clean_title],
                "price": ["0"]  # dummy price since the data cleaner expects 'price'
            }
            return pd.DataFrame(input_dict)
        except Exception as e:
            raise MyException(e, sys)

class VehiclePredictor:
    def __init__(self):
        pass

    def get_latest_model_path(self) -> str:
        try:
            artifact_dir = "artifact"
            if not os.path.exists(artifact_dir):
                raise Exception("Artifact directory does not exist. Please train the model first.")
            timestamps = os.listdir(artifact_dir)
            if not timestamps:
                raise Exception("No training artifacts found. Please train the model first.")
            
            from datetime import datetime
            parsed_timestamps = []
            for ts in timestamps:
                try:
                    dt = datetime.strptime(ts, "%m_%d_%Y_%H_%M_%S")
                    parsed_timestamps.append((dt, ts))
                except Exception:
                    continue
            
            if not parsed_timestamps:
                raise Exception("No valid timestamped artifact folders found.")
            
            parsed_timestamps.sort(reverse=True)
            for _, ts in parsed_timestamps:
                model_path = os.path.join(artifact_dir, ts, "model_trainer", "trained_model", "model.pkl")
                if os.path.exists(model_path):
                    return model_path
            
            raise Exception("No trained model.pkl found in any artifact directory.")
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: pd.DataFrame) -> float:
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            
            # The model is MyModel wrapper which bundles preprocessing_object and trained_model_object
            transformed_df = model.preprocessing_object.transform(dataframe)
            
            # Drop price_usd if present since the estimator expects only features
            if "price_usd" in transformed_df.columns:
                transformed_df = transformed_df.drop(columns=["price_usd"])
                
            prediction = model.trained_model_object.predict(transformed_df)
            
            # The target was log-transformed, so we invert the log transform
            actual_price = np.expm1(prediction[0])
            return float(actual_price)
        except Exception as e:
            raise MyException(e, sys)
