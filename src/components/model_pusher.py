import os
import sys
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelPusherArtifact,
    DataTransformationArtifact,
)
from src.entity.azure_estimator import AzureEstimator
from src.constants import (
    MODEL_CONTAINER_NAME,
    MODEL_PUSHER_BLOB_PATH,
    MODEL_FILE_NAME,
    PREPROCSSING_OBJECT_FILE_NAME,
    INFERENCE_META_FILE_NAME,
    TRANSFORMATION_PUSHER_BLOB_PATH,
)

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig) -> None:
        """
        Initializes the ModelPusher with config and Azure Estimator.
        """
        try:
            self.model_pusher_config = model_pusher_config
            self.azure_estimator = AzureEstimator()
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_pusher(self, model_trainer_artifact: ModelTrainerArtifact) -> ModelPusherArtifact:
        """
        Pushes the trained model file to Azure Blob Storage.
        """
        try:
            logging.info("Starting model pusher component to upload trained model to Azure Blob Storage.")
            
            trained_model_path = model_trainer_artifact.trained_model_file_path
            blob_name = f"{MODEL_PUSHER_BLOB_PATH}/{MODEL_FILE_NAME}"
            container_name = MODEL_CONTAINER_NAME
            
            logging.info(f"Uploading model from local path: '{trained_model_path}' to container: '{container_name}', blob: '{blob_name}'")
            
            uploaded_blob_path = self.azure_estimator.upload_model_file(
                local_file_path=trained_model_path,
                blob_name=blob_name,
                container_name=container_name
            )
            
            model_pusher_artifact = ModelPusherArtifact(
                container_name=container_name,
                azure_blob_path=uploaded_blob_path
            )
            
            logging.info(f"Model pusher artifact created successfully: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def push_transformation_artifacts(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_pusher_artifact: ModelPusherArtifact,
    ) -> ModelPusherArtifact:
        """
        Pushes preprocessing.pkl and inference_meta.json to Azure Blob Storage
        under the transformation prefix.
        """
        try:
            logging.info("Starting transformation artifact upload to Azure Blob Storage.")

            preprocessing_path = data_transformation_artifact.transformed_object_file_path
            inference_meta_path = os.path.join(
                os.path.dirname(preprocessing_path),
                INFERENCE_META_FILE_NAME,
            )

            if not os.path.exists(preprocessing_path):
                raise FileNotFoundError(f"Preprocessing object not found at: {preprocessing_path}")
            if not os.path.exists(inference_meta_path):
                raise FileNotFoundError(f"Inference metadata not found at: {inference_meta_path}")

            container_name = model_pusher_artifact.container_name
            preprocessing_blob = f"{TRANSFORMATION_PUSHER_BLOB_PATH}/{PREPROCSSING_OBJECT_FILE_NAME}"
            inference_meta_blob = f"{TRANSFORMATION_PUSHER_BLOB_PATH}/{INFERENCE_META_FILE_NAME}"

            logging.info(
                f"Uploading preprocessing object from '{preprocessing_path}' to blob '{preprocessing_blob}'"
            )
            uploaded_preprocessing_blob = self.azure_estimator.upload_model_file(
                local_file_path=preprocessing_path,
                blob_name=preprocessing_blob,
                container_name=container_name,
            )

            logging.info(
                f"Uploading inference metadata from '{inference_meta_path}' to blob '{inference_meta_blob}'"
            )
            uploaded_inference_meta_blob = self.azure_estimator.upload_model_file(
                local_file_path=inference_meta_path,
                blob_name=inference_meta_blob,
                container_name=container_name,
            )

            model_pusher_artifact.preprocessing_blob_path = uploaded_preprocessing_blob
            model_pusher_artifact.inference_meta_blob_path = uploaded_inference_meta_blob

            logging.info(
                "Transformation artifacts uploaded successfully: "
                f"preprocessing='{uploaded_preprocessing_blob}', "
                f"inference_meta='{uploaded_inference_meta_blob}'"
            )
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e
