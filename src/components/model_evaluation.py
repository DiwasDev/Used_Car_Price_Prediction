from dataclasses import dataclass
import sys
from typing import Optional, Any

import pandas as pd
from sklearn.metrics import r2_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataSplitterArtifact,
    ModelEvaluationArtifact,
    DataTransformationArtifact,
)
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object, load_numpy_array_data
import numpy as np

# Azure helper functions (assumes you added src/entity/azure_estimator.py)
from src.entity.azure_estimator import AzureEstimator

from src.components.data_transformers.data_cleaning import DataCleaner
from src.components.data_transformers.feature_engineering import FeatureEngineer
from src.components.data_transformers.handle_missing import MissingValueHandler
from src.components.data_transformers.handle_outliers import OutlierHandler
from src.components.data_transformers.encoder import CategoricalEncoder

@dataclass
class EvaluateModelResponse:
    trained_model_r2_score: float
    best_model_r2_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_splitter_artifact: DataSplitterArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_splitter_artifact = data_splitter_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Any]:
        """
        Fetch the current production model from Azure Blob Storage (if present)
        and return the loaded model object (e.g., a scikit-learn estimator).
        """
        try:
            # The config fields were named for S3 in the original tutorial.
            # We treat `bucket_name` as the Azure container name and
            # `s3_model_key_path` as the blob prefix/path to look under.
            container_name = self.model_eval_config.container_name
            blob_prefix = self.model_eval_config.blob_prefix

            # List blobs under the prefix in the configured container
            azure_estimator = AzureEstimator()
            blobs = azure_estimator.list_blobs_with_prefix(prefix=blob_prefix, container_name=container_name)
            if not blobs:
                logging.info(f"No blobs found with prefix '{blob_prefix}' in container '{container_name}'.")
                return None

            # Select the latest blob by last_modified time
            latest_blob = max(blobs, key=lambda b: b.last_modified)
            logging.info(f"Found production model blob: {latest_blob.name} (last_modified={latest_blob.last_modified})")

            # Download blob into memory and load the model object (joblib/pickle)
            model_obj = azure_estimator.download_model_object(blob_name=latest_blob.name, container_name=container_name)
            logging.info("Production model loaded from Azure Blob Storage.")
            return model_obj

        except Exception as e:
            raise MyException(e, sys) from e

    # Main function
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluate the newly trained model against the production model (if available)
        and decide whether to accept the new model based on r2 score delta.
        """
        try:
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            X_test, y_test = np.delete(test_arr, 3, axis=1), test_arr[:, 3]

            trained_model_r2_score = self.model_trainer_artifact.metric_artifact.r2_score
            logging.info(f"r2_score for this trained model: {trained_model_r2_score}")

            best_model_r2_score: Optional[float] = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info("Computing r2_score for production model...")
                y_hat_best_model = best_model.predict(X_test)    ## todo: fix this
                best_model_r2_score = float(r2_score(y_test, y_hat_best_model))
                logging.info(
                    f"r2_score-Production Model: {best_model_r2_score}, r2_score-New Trained Model: {trained_model_r2_score}"
                )

            tmp_best_model_score = 0.0 if best_model_r2_score is None else best_model_r2_score
            is_accepted = trained_model_r2_score > tmp_best_model_score
            difference = trained_model_r2_score - tmp_best_model_score

            result = EvaluateModelResponse(
                trained_model_r2_score=trained_model_r2_score,
                best_model_r2_score=best_model_r2_score,
                is_model_accepted=is_accepted,
                difference=difference,
            )
            logging.info(f"Model evaluation result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Orchestrate model evaluation and return a ModelEvaluationArtifact describing the outcome.
        """
        try:
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            # keep the same field name used elsewhere; this points to the blob prefix
            azure_blob_path = self.model_eval_config.blob_prefix

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                azure_blob_path=azure_blob_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e