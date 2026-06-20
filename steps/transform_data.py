import os
import sys
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataSplitterArtifact, DataValidationArtifact
from src.exception import MyException
from src.utils.main_utils import save_object, save_numpy_array_data

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
    try:
        if not validation_artifact.validation_status:
            raise Exception(validation_artifact.message)

        # Instantiate components
        data_transformation = DataTransformation(
            splitter_artifact=splitter_artifact,
            data_transformation_config=config,
            data_validation_artifact=validation_artifact
        )

        # Retrieve the sklearn Pipeline
        preprocessor = data_transformation.get_data_transformer_object()

        # Load train and test data
        train_df = data_transformation.read_data(file_path=splitter_artifact.trained_file_path)
        test_df = data_transformation.read_data(file_path=splitter_artifact.test_file_path)

        # Fit and transform train data, transform test data
        train_df_transformed = preprocessor.fit_transform(train_df)
        test_df_transformed = preprocessor.transform(test_df)

        # Convert to numpy arrays
        train_arr = train_df_transformed.to_numpy()
        test_arr = test_df_transformed.to_numpy()

        # Export files and save preprocessing object (logics implemented in the step)
        os.makedirs(os.path.dirname(config.transformed_object_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(config.transformed_train_file_path), exist_ok=True)

        save_object(config.transformed_object_file_path, preprocessor)
        save_numpy_array_data(config.transformed_train_file_path, array=train_arr)
        save_numpy_array_data(config.transformed_test_file_path, array=test_arr)

        return DataTransformationArtifact(
            transformed_object_file_path=config.transformed_object_file_path,
            transformed_train_file_path=config.transformed_train_file_path,
            transformed_test_file_path=config.transformed_test_file_path
        )
    except Exception as e:
        raise MyException(e, sys)
