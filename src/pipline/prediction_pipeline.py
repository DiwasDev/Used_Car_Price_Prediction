import sys
import os
import json
import pandas as pd
import numpy as np
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object

def sync_latest_artifact():
    try:
        import shutil
        from datetime import datetime
        
        artifact_dir = "artifact"
        if not os.path.exists(artifact_dir):
            return
            
        # Find all timestamped directories
        timestamps = []
        for name in os.listdir(artifact_dir):
            if name == "latest_artifact":
                continue
            path = os.path.join(artifact_dir, name)
            if os.path.isdir(path):
                try:
                    dt = datetime.strptime(name, "%m_%d_%Y_%H_%M_%S")
                    timestamps.append((dt, name))
                except ValueError:
                    continue
                    
        if not timestamps:
            return
            
        # Get the latest directory
        timestamps.sort(reverse=True)
        latest_name = timestamps[0][1]
        latest_path = os.path.join(artifact_dir, latest_name)
        
        # Define destination path
        dest_path = os.path.join(artifact_dir, "latest_artifact")
        
        # Remove existing symlink/directory
        if os.path.exists(dest_path) or os.path.islink(dest_path):
            if os.path.islink(dest_path):
                os.unlink(dest_path)
            elif os.path.isdir(dest_path):
                shutil.rmtree(dest_path)
            else:
                os.remove(dest_path)
                
        # Copy the latest timestamped folder to latest_artifact
        shutil.copytree(latest_path, dest_path)
        
        # Ensure transformation_object exists (points to transformed_object)
        data_trans_dir = os.path.join(dest_path, "data_transformation")
        if os.path.exists(data_trans_dir):
            transformed_obj_dir = os.path.join(data_trans_dir, "transformed_object")
            transformation_obj_dir = os.path.join(data_trans_dir, "transformation_object")
            if os.path.exists(transformed_obj_dir) and not os.path.exists(transformation_obj_dir):
                shutil.copytree(transformed_obj_dir, transformation_obj_dir)
                
    except Exception as e:
        logging.error(f"Error syncing latest artifact: {e}")

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

    def predict(self, dataframe: pd.DataFrame):
        try:
            sync_latest_artifact()
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            
            # Check if dataframe is raw or preprocessed[ TODO: check if this is needed later ]
            if "model" in dataframe.columns:
                # Raw input: transform using preprocessor
                transformed_df = model.preprocessing_object.transform(dataframe)
                # Drop price_usd if present since the estimator expects only features
                if "price_usd" in transformed_df.columns:
                    transformed_df = transformed_df.drop(columns=["price_usd"])
            else:
                # Preprocessed input (like test.csv):
                dummy_raw = pd.DataFrame({
                    "brand": ["BMW"],
                    "model": ["3 Series"],
                    "model_year": [2020],
                    "milage": ["12,000 mi"],
                    "fuel_type": ["Gasoline"],
                    "engine": ["2.0L I4"],
                    "transmission": ["Automatic"],
                    "ext_col": ["Black"],
                    "int_col": ["Black"],
                    "accident": ["None reported"],
                    "clean_title": ["Yes"],
                    "price": ["0"]
                })
                # dummy_transformed = model.preprocessing_object.transform(dummy_raw)
                expected_cols = [col for col in dummy_transformed.columns if col != "price_usd"]
                
                transformed_df = dataframe.copy()
                
                # Drop target and index/unneeded columns
                for col_to_drop in ["price_usd", "price", "Unnamed: 0", ""]:
                    if col_to_drop in transformed_df.columns:
                        transformed_df = transformed_df.drop(columns=[col_to_drop])
                
                # If brand is string, encode it using inference_meta.json
                if "brand" in transformed_df.columns and transformed_df["brand"].dtype == object:
                    meta_path = os.path.join("artifact", "latest_artifact", "data_transformation", "transformation_object", "inference_meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        brand_mapping = meta.get("target_encodings", {}).get("brand", {})
                        fallback = brand_mapping.get("__global_fallback__", 10.310518988406342)
                        transformed_df["brand"] = transformed_df["brand"].map(brand_mapping).fillna(fallback)
                    else:
                        transformed_df["brand"] = 10.310518988406342
                
                # Align columns
                transformed_df = transformed_df.reindex(columns=expected_cols, fill_value=0.0)
            
            prediction = model.trained_model_object.predict(transformed_df)
            actual_prices = np.expm1(prediction)
            
            if len(actual_prices) == 1:
                return float(actual_prices[0])
            return [float(p) for p in actual_prices]
        except Exception as e:
            raise MyException(e, sys)
