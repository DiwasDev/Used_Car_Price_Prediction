import sys
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelTrainerArtifact, ModelPusherArtifact
from src.entity.azure_estimator import AzureEstimator
from src.constants import MODEL_CONTAINER_NAME, MODEL_PUSHER_BLOB_PATH, MODEL_FILE_NAME

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
