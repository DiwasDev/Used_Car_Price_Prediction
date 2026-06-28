import sys
import os
import json
import pandas as pd
import numpy as np
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object
from src.constants import (
    MODEL_CONTAINER_NAME,
    MODEL_PUSHER_BLOB_PATH,
    MODEL_FILE_NAME,
    PREPROCSSING_OBJECT_FILE_NAME,
    INFERENCE_META_FILE_NAME,
    TRANSFORMATION_PUSHER_BLOB_PATH,
)

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
        self._model_from_blob = False
        self._preprocessing_from_blob = False
        self._inference_meta_from_blob = False

    def _get_azure_estimator(self):
        from src.entity.azure_estimator import AzureEstimator
        return AzureEstimator()

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

    def get_latest_preprocessing_path(self) -> str:
        try:
            artifact_dir = "artifact"
            if not os.path.exists(artifact_dir):
                raise Exception("Artifact directory does not exist. Please train the model first.")

            from datetime import datetime
            parsed_timestamps = []
            for name in os.listdir(artifact_dir):
                if name == "latest_artifact":
                    continue
                try:
                    dt = datetime.strptime(name, "%m_%d_%Y_%H_%M_%S")
                    parsed_timestamps.append((dt, name))
                except ValueError:
                    continue

            if not parsed_timestamps:
                latest_path = os.path.join(artifact_dir, "latest_artifact", "data_transformation", "transformed_object", PREPROCSSING_OBJECT_FILE_NAME)
                if os.path.exists(latest_path):
                    return latest_path
                raise Exception("No valid timestamped artifact folders found.")

            parsed_timestamps.sort(reverse=True)
            for _, ts in parsed_timestamps:
                preprocessing_path = os.path.join(
                    artifact_dir,
                    ts,
                    "data_transformation",
                    "transformed_object",
                    PREPROCSSING_OBJECT_FILE_NAME,
                )
                if os.path.exists(preprocessing_path):
                    return preprocessing_path

            raise Exception("No preprocessing.pkl found in any artifact directory.")
        except Exception as e:
            raise MyException(e, sys)

    def get_latest_inference_meta_path(self) -> str:
        try:
            artifact_dir = "artifact"
            candidate_paths = [
                os.path.join(
                    artifact_dir,
                    "latest_artifact",
                    "data_transformation",
                    "transformation_object",
                    INFERENCE_META_FILE_NAME,
                ),
                os.path.join(
                    artifact_dir,
                    "latest_artifact",
                    "data_transformation",
                    "transformed_object",
                    INFERENCE_META_FILE_NAME,
                ),
            ]

            from datetime import datetime
            parsed_timestamps = []
            if os.path.exists(artifact_dir):
                for name in os.listdir(artifact_dir):
                    if name == "latest_artifact":
                        continue
                    try:
                        dt = datetime.strptime(name, "%m_%d_%Y_%H_%M_%S")
                        parsed_timestamps.append((dt, name))
                    except ValueError:
                        continue

            parsed_timestamps.sort(reverse=True)
            for _, ts in parsed_timestamps:
                candidate_paths.extend([
                    os.path.join(
                        artifact_dir,
                        ts,
                        "data_transformation",
                        "transformed_object",
                        INFERENCE_META_FILE_NAME,
                    ),
                    os.path.join(
                        artifact_dir,
                        ts,
                        "data_transformation",
                        "transformation_object",
                        INFERENCE_META_FILE_NAME,
                    ),
                ])

            for path in candidate_paths:
                if os.path.exists(path):
                    return path

            raise Exception("No inference_meta.json found in any artifact directory.")
        except Exception as e:
            raise MyException(e, sys)

    def get_latest_model_from_blob(self):
        try:
            azure_estimator = self._get_azure_estimator()
            blobs = azure_estimator.list_blobs_with_prefix(prefix=MODEL_PUSHER_BLOB_PATH)
            model_blobs = [blob for blob in blobs if blob.name.endswith(MODEL_FILE_NAME)]
            if not model_blobs:
                raise Exception("No model blobs found in container.")
            latest_blob = max(model_blobs, key=lambda blob: blob.last_modified)
            model_obj = azure_estimator.download_model_object(blob_name=latest_blob.name)
            self._model_from_blob = True
            return model_obj
        except Exception as e:
            raise MyException(e, sys)

    def get_latest_preprocessing_from_blob(self):
        try:
            azure_estimator = self._get_azure_estimator()
            blob_name = f"{TRANSFORMATION_PUSHER_BLOB_PATH}/{PREPROCSSING_OBJECT_FILE_NAME}"
            preprocessing_obj = azure_estimator.download_preprocessing_object(
                blob_name=blob_name,
                container_name=MODEL_CONTAINER_NAME,
            )
            self._preprocessing_from_blob = True
            return preprocessing_obj
        except Exception as e:
            raise MyException(e, sys)

    def get_latest_inference_meta_from_blob(self) -> dict:
        try:
            azure_estimator = self._get_azure_estimator()
            blob_name = f"{TRANSFORMATION_PUSHER_BLOB_PATH}/{INFERENCE_META_FILE_NAME}"
            inference_meta = azure_estimator.download_blob_json(
                blob_name=blob_name,
                container_name=MODEL_CONTAINER_NAME,
            )
            self._inference_meta_from_blob = True
            return inference_meta
        except Exception as e:
            raise MyException(e, sys)

    def pull_model(self):
        try:
            model = None
            try:
                model = self.get_latest_model_from_blob()
            except Exception as e:
                logging.info(f"Model not found in blob storage or download failed: {e}")
            
            if model is None:
                logging.info("Attempting to get model from local artifact...")
                sync_latest_artifact()
                try:
                    model_path = self.get_latest_model_path()
                    if os.path.exists(model_path):
                        model = load_object(file_path=model_path)
                except Exception as e:
                    logging.info(f"Model not found in artifact: {e}")
            
            if model is None:
                raise Exception("Neither blob model nor artifact model is available.")
            
            return model
        except Exception as e:
            raise MyException(e, sys)

    def pull_preprocessing(self):
        try:
            preprocessing = None
            try:
                preprocessing = self.get_latest_preprocessing_from_blob()
            except Exception as e:
                logging.info(f"Preprocessing not found in blob storage or download failed: {e}")

            if preprocessing is None:
                logging.info("Attempting to get preprocessing object from local artifact...")
                sync_latest_artifact()
                try:
                    preprocessing_path = self.get_latest_preprocessing_path()
                    if os.path.exists(preprocessing_path):
                        preprocessing = load_object(file_path=preprocessing_path)
                except Exception as e:
                    logging.info(f"Preprocessing not found in artifact: {e}")

            if preprocessing is None:
                raise Exception("Neither blob preprocessing nor local preprocessing is available.")

            return preprocessing
        except Exception as e:
            raise MyException(e, sys)

    def pull_inference_meta(self) -> dict:
        try:
            inference_meta = None
            try:
                inference_meta = self.get_latest_inference_meta_from_blob()
            except Exception as e:
                logging.info(f"Inference metadata not found in blob storage or download failed: {e}")

            if inference_meta is None:
                logging.info("Attempting to get inference metadata from local artifact...")
                sync_latest_artifact()
                try:
                    meta_path = self.get_latest_inference_meta_path()
                    if os.path.exists(meta_path):
                        with open(meta_path, "r") as meta_file:
                            inference_meta = json.load(meta_file)
                except Exception as e:
                    logging.info(f"Inference metadata not found in artifact: {e}")

            if inference_meta is None:
                raise Exception("Neither blob inference metadata nor local inference metadata is available.")

            return inference_meta
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: pd.DataFrame):
        try:
            model = self.pull_model()
            inference_meta = self.pull_inference_meta()

            if "model" in dataframe.columns:
                if getattr(model, "preprocessing_object", None) is not None:
                    preprocessing = model.preprocessing_object
                else:
                    preprocessing = self.pull_preprocessing()

                transformed_df = preprocessing.transform(dataframe)
                if "price_usd" in transformed_df.columns:
                    transformed_df = transformed_df.drop(columns=["price_usd"])
            else:
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

                if getattr(model, "preprocessing_object", None) is not None:
                    expected_cols = [
                        col for col in model.preprocessing_object.transform(dummy_raw).columns
                        if col != "price_usd"
                    ]
                else:
                    preprocessing = self.pull_preprocessing()
                    expected_cols = [
                        col for col in preprocessing.transform(dummy_raw).columns
                        if col != "price_usd"
                    ]

                transformed_df = dataframe.copy()

                for col_to_drop in ["price_usd", "price", "Unnamed: 0", ""]:
                    if col_to_drop in transformed_df.columns:
                        transformed_df = transformed_df.drop(columns=[col_to_drop])

                if "brand" in transformed_df.columns and transformed_df["brand"].dtype == object:
                    brand_mapping = inference_meta.get("target_encodings", {}).get("brand", {})
                    fallback = brand_mapping.get("__global_fallback__", 10.310518988406342)
                    transformed_df["brand"] = transformed_df["brand"].map(brand_mapping).fillna(fallback)

                transformed_df = transformed_df.reindex(columns=expected_cols, fill_value=0.0)

            prediction = model.trained_model_object.predict(transformed_df)
            actual_prices = np.expm1(prediction)

            if len(actual_prices) == 1:
                return float(actual_prices[0])
            return [float(price) for price in actual_prices]
        except Exception as e:
            raise MyException(e, sys)
