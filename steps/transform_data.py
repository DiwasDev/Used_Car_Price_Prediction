import os
import sys
import json  # Added to prevent NameError on json.dump
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataSplitterArtifact, DataValidationArtifact
from src.exception import MyException
from src.utils.main_utils import save_object, save_numpy_array_data
from src.logger import logging

def transform_data(
    config: DataTransformationConfig,
    splitter_artifact: DataSplitterArtifact,
    validation_artifact: DataValidationArtifact
) -> DataTransformationArtifact:
    """
    Transforms the train and test datasets using the unified preprocessing pipeline.

    Parameters:
        config: DataTransformationConfig: Configuration for data transformation.
        splitter_artifact: DataSplitterArtifact: Split datasets containing train and test file paths.
        validation_artifact: DataValidationArtifact: Validation status of the dataset.

    Returns:
        artifact: DataTransformationArtifact: Transformed object and data file paths.
    """
    logging.info("Starting data transformation process...")
    try:
        # Check validation status
        if not validation_artifact.validation_status:
            error_msg = f"Data validation failed. Cannot proceed with transformation. Message: {validation_artifact.message}"
            logging.error(error_msg)
            raise Exception(error_msg)

        logging.info("Data validation status verified successfully. Instantiating DataTransformation component...")
        # Instantiate components
        data_transformation = DataTransformation(
            splitter_artifact=splitter_artifact,
            data_transformation_config=config,
            data_validation_artifact=validation_artifact
        )

        # Retrieve the sklearn Pipeline
        preprocessor = data_transformation.get_data_transformer_object()

        # Load train and test data
        logging.info(f"Loading training data from: {splitter_artifact.trained_file_path}")
        train_df = data_transformation.read_data(file_path=splitter_artifact.trained_file_path)
        
        logging.info(f"Loading test data from: {splitter_artifact.test_file_path}")
        test_df = data_transformation.read_data(file_path=splitter_artifact.test_file_path)

        # Fit and transform train data, transform test data
        logging.info("Executing pipeline fit_transform on training dataset...")
        train_df_transformed = preprocessor.fit_transform(train_df)
        
        logging.info("Executing pipeline transform on test dataset...")
        test_df_transformed = preprocessor.transform(test_df)
        
        # Convert to numpy arrays
        logging.info("Converting transformed DataFrames to NumPy arrays...")
        train_arr = train_df_transformed.to_numpy()
        test_arr = test_df_transformed.to_numpy()
        logging.info(f"Transformed train array shape: {train_arr.shape}, test array shape: {test_arr.shape}")

        # Export files and save preprocessing object
        logging.info("Ensuring target output directories exist...")
        os.makedirs(os.path.dirname(config.transformed_object_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(config.transformed_train_file_path), exist_ok=True)

        # Save lightweight inference metadata
        logging.info("Extracting lightweight inference metadata from preprocessor...")
        inference_meta = data_transformation._extract_inference_meta(preprocessor)

        meta_file_path = os.path.join(
            os.path.dirname(data_transformation.data_transformation_config.transformed_object_file_path),
            "inference_meta.json"
        )
        
        logging.info(f"Saving inference metadata JSON to: {meta_file_path}")
        with open(meta_file_path, "w") as f:
            json.dump(inference_meta, f, indent=4)
        logging.info("Lightweight inference metadata saved successfully.")

        # Save binary artifacts
        logging.info(f"Saving pickle preprocessor object to: {config.transformed_object_file_path}")
        save_object(config.transformed_object_file_path, preprocessor)
        
        logging.info(f"Saving transformed training NumPy array to: {config.transformed_train_file_path}")
        save_numpy_array_data(config.transformed_train_file_path, array=train_arr)
        
        logging.info(f"Saving transformed test NumPy array to: {config.transformed_test_file_path}")
        save_numpy_array_data(config.transformed_test_file_path, array=test_arr)

        logging.info("All data transformation artifacts exported successfully.")
        return DataTransformationArtifact(
            transformed_object_file_path=config.transformed_object_file_path,
            transformed_train_file_path=config.transformed_train_file_path,
            transformed_test_file_path=config.transformed_test_file_path
        )
        
    except Exception as e:
        logging.exception("An exception occurred during the data transformation stage.")
        raise MyException(e, sys)