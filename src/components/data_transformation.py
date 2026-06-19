import sys
import numpy as np
import pandas as pd
# from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data

from src.components.data_transformers.data_cleaning import DataCleaner
from src.components.data_transformers.feature_engineering import FeatureEngineer
from src.components.data_transformers.handle_missing import MissingValueHandler
from src.components.data_transformers.handle_outliers import OutlierHandler
from src.components.data_transformers.encoder import CategoricalEncoder


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
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

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Create the data transformer object
            cleaner = DataCleaner()
            engineer = FeatureEngineer()
            missing_handler = MissingValueHandler()
            outlier_handler = OutlierHandler()
            encoder  = CategoricalEncoder()

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # Clean the data
            train_df = cleaner.fit_transform(train_df)
            test_df = cleaner.transform(test_df)

            # Feature engineer the data
            train_df = engineer.fit_transform(train_df)
            test_df = engineer.transform(test_df)

            # Impute missing values
            train_df = missing_handler.fit_transform(train_df)
            test_df = missing_handler.transform(test_df)

            # Cap outliers
            outlier_handler.fit(train_df)
            train_df = outlier_handler.transform(train_df)
            test_df = outlier_handler.transform(test_df)

            # Encode categorical features
            train_df = encoder.fit_transform(train_df)
            test_df = encoder.transform(test_df)

            # Convert to numpy arrays
            train_arr= train_df.to_numpy()
            test_arr = test_df.to_numpy()
                        
            # save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
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