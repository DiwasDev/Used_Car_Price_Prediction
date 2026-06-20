import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataSplitterArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data

from src.components.data_transformers.data_cleaning import DataCleaner
from src.components.data_transformers.feature_engineering import FeatureEngineer
from src.components.data_transformers.handle_missing import MissingValueHandler
from src.components.data_transformers.handle_outliers import OutlierHandler
from src.components.data_transformers.encoder import CategoricalEncoder


class DataTransformation:
    """
    Component for handling data transformation using a unified preprocessing pipeline.
    """

    def __init__(self, splitter_artifact: DataSplitterArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact) -> None:
        """
        Initializes the DataTransformation class.

        Parameters:
            splitter_artifact: DataSplitterArtifact: Reference to train/test split files.
            data_transformation_config: DataTransformationConfig: Configuration for transformation.
            data_validation_artifact: DataValidationArtifact: Reference to validation status.

        Returns:
            None
        """
        try:
            self.splitter_artifact = splitter_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads CSV data into a pandas DataFrame.

        Parameters:
            file_path: str: Path to the CSV file.

        Returns:
            df: pd.DataFrame: Loaded DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a scikit-learn Pipeline containing all preprocessors.

        Parameters:
            None

        Returns:
            preprocessor: Pipeline: Scikit-learn Pipeline representing the preprocessing steps.
        """
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

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation process and saves outputs.

        Parameters:
            None

        Returns:
            artifact: DataTransformationArtifact: Artifact containing paths to saved outputs.
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

            # Save preprocessing object and transformed data
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
