import sys
import os
import json # Added for handling the JSON metadata dump
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.artifact_entity import DataTransformationArtifact, DataSplitterArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data

from src.components.data_transformers.data_cleaning import DataCleaner
from src.components.data_transformers.feature_engineering import FeatureEngineer
from src.components.data_transformers.handle_missing import MissingValueHandler
from src.components.data_transformers.handle_outliers import OutlierHandler
from src.components.data_transformers.encoder import CategoricalEncoder, TargetEncodingStrategy, OneHotEncodingStrategy


class DataTransformation:
    """
    Component for handling data transformation using a unified preprocessing pipeline.
    """

    def __init__(self, splitter_artifact: DataSplitterArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact) -> None:
        try:
            self.splitter_artifact = splitter_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Assembling preprocessing pipeline...")
            preprocessor = Pipeline(steps=[
                ('cleaner', DataCleaner()),
                ('engineer', FeatureEngineer()),
                ('missing_handler', MissingValueHandler()),
                ('outlier_handler', OutlierHandler()),
                ('encoder', CategoricalEncoder())
            ])
            logging.info("Preprocessing pipeline assembled successfully.")
            return preprocessor
        except Exception as e:
            raise MyException(e, sys)

    def _extract_inference_meta(self, preprocessor: Pipeline) -> dict:
        """
        Extracts lightweight lookup weights from the fitted Pipeline steps,
        mapping raw string categories directly to their target encoded numerical values.
        """
        try:
            logging.info("Extracting light-inference weights metadata...")
            meta = {
                "target_encodings": {},
                "one_hot_features": []
            }
            
            # Extract from the Facade encoder step
            categorical_encoder = preprocessor.named_steps['encoder']
            
            for strategy in categorical_encoder.strategies:
                # 1. Corrected Target Encoding Extraction Logic
                if isinstance(strategy, TargetEncodingStrategy):
                    master_enc = strategy._master_encoder
                    if master_enc is not None:
                        for col in strategy.columns:
                            meta["target_encodings"][col] = {}
                            
                            # A. Extract target statistics (mapped to integer codes e.g. 1, 2, 3...)
                            target_stats = master_enc.mapping[col].to_dict()
                            
                            # B. Extract the string-to-integer mappings from the internal ordinal encoder
                            # category_encoders stores this as a list of dicts/dataframes inside ordinal_encoder.mapping
                            ord_mapping_list = master_enc.ordinal_encoder.mapping
                            col_ord_mapping = next(item for item in ord_mapping_list if item["col"] == col)
                            
                            # mapping series looks like: pd.Series([1, 2, 3], index=['Ford', 'Hyundai', 'Audi'])
                            string_to_int_series = col_ord_mapping["mapping"]
                            
                            # C. Combine them: Map raw string directly to final float target values
                            for string_category, int_code in string_to_int_series.items():
                                # Keep missing value categories or system codes clean
                                if pd.isna(string_category) or int_code < 0:
                                    continue
                                
                                # Fetch the float target code value using the integer code
                                encoded_value = target_stats.get(int_code) or target_stats.get(str(int_code))
                                if encoded_value is not None:
                                    meta["target_encodings"][col][str(string_category)] = float(encoded_value)
                            
                            # D. Grab the overall dataset mean as a fallback for completely new categories
                            meta["target_encodings"][col]["__global_fallback__"] = float(master_enc._mean)
                
                # 2. Extract One-Hot Schema ordering
                elif isinstance(strategy, OneHotEncodingStrategy):
                    if strategy._encoder is not None:
                        feature_names = strategy._encoder.get_feature_names_out(strategy.columns)
                        meta["one_hot_features"] = list(feature_names)
            
            logging.info("Successfully joined internal ordinal maps with target values.")
            return meta
            
        except Exception as e:
            logging.error(f"Failed to extract inference metadata: {e}")
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation process, saves numpy structures, 
        and dumps lightweight json metadata for production inference.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            preprocessor = self.get_data_transformer_object()

            # Load train and test data
            train_df = self.read_data(file_path=self.splitter_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.splitter_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # Fit and transform using the unified pipeline
            train_df_transformed = preprocessor.fit_transform(train_df)
            test_df_transformed = preprocessor.transform(test_df)

            # Convert to numpy arrays
            train_arr = train_df_transformed.to_numpy()
            test_arr = test_df_transformed.to_numpy()

            # --- MLOps Optimization: Extract and Save Light Artifact ---
            inference_meta = self._extract_inference_meta(preprocessor)
            
            # Define output JSON file path near your pickle path
            meta_file_path = os.path.join(
                os.path.dirname(self.data_transformation_config.transformed_object_file_path),
                "inference_meta.json"
            )
            
            with open(meta_file_path, "w") as f:
                json.dump(inference_meta, f, indent=4)
            logging.info(f"Lightweight inference metadata saved at: {meta_file_path}")
            # -----------------------------------------------------------

            # Save heavy preprocessing object and transformed data arrays
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e