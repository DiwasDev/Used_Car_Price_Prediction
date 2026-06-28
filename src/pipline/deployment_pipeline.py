import sys
from typing import Optional
from src.exception import MyException
from src.logger import logging
from src.pipline.training_pipeline import TrainPipeline
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelPusherArtifact,
    DataTransformationArtifact,
)

class DeploymentPipeline(TrainPipeline):
    """
    Orchestration pipeline class to run training/preprocessing workflows and deploy
    the model to Azure Blob Storage if it meets evaluation/deployment criteria.
    """

    def __init__(self) -> None:
        """
        Initializes configurations for training steps and the model pusher.
        """
        try:
            super().__init__()
            self.model_pusher_config = ModelPusherConfig()
        except Exception as e:
            raise MyException(e, sys) from e

    def start_model_pusher(self, model_trainer_artifact: ModelTrainerArtifact) -> ModelPusherArtifact:
        """
        Runs the model pusher step to deploy/upload the model to cloud storage.

        Parameters:
            model_trainer_artifact: ModelTrainerArtifact: Artifact containing the trained model path.

        Returns:
            artifact: ModelPusherArtifact: Artifact containing the upload details.
        """
        try:
            logging.info("Entered the start_model_pusher method of DeploymentPipeline class")
            model_pusher = ModelPusher(model_pusher_config=self.model_pusher_config)
            model_pusher_artifact = model_pusher.initiate_model_pusher(
                model_trainer_artifact=model_trainer_artifact
            )
            logging.info("Exited the start_model_pusher method of DeploymentPipeline class")
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def start_transformation_pusher(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_pusher_artifact: ModelPusherArtifact,
    ) -> ModelPusherArtifact:
        """
        Runs the transformation pusher step to upload preprocessing.pkl and
        inference_meta.json to Azure Blob Storage.
        """
        try:
            logging.info("Entered the start_transformation_pusher method of DeploymentPipeline class")
            model_pusher = ModelPusher(model_pusher_config=self.model_pusher_config)
            model_pusher_artifact = model_pusher.push_transformation_artifacts(
                data_transformation_artifact=data_transformation_artifact,
                model_pusher_artifact=model_pusher_artifact,
            )
            logging.info("Exited the start_transformation_pusher method of DeploymentPipeline class")
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(self) -> Optional[ModelPusherArtifact]:
        """
        Runs the full training/preprocessing pipeline followed by deployment if criteria is met.

        Parameters:
            None

        Returns:
            artifact: Optional[ModelPusherArtifact]: Artifact representing successful deployment, or None if skipped.
        """
        try:
            logging.info("Starting DeploymentPipeline execution.")
            
            # Step 1: Ingest Data
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Step 2: Split Data
            data_splitter_artifact = self.start_data_splitter(data_ingestion_artifact=data_ingestion_artifact)
            
            # Step 3: Validate Data
            data_validation_artifact = self.start_data_validation(splitter_artifact=data_splitter_artifact)
            
            # Step 4: Transform Data
            data_transformation_artifact = self.start_data_transformation(
                splitter_artifact=data_splitter_artifact,
                data_validation_artifact=data_validation_artifact
            )
            
            # Step 5: Train Model
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            
            # Step 6: Evaluate Model
            model_evaluation_artifact = self.start_model_evaluation(
                data_splitter_artifact=data_splitter_artifact,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )

            # Step 7: Check deployment criteria (is_model_accepted)
            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Trained model did not meet the deployment criteria. Deployment skipped.")
                return None
                
            logging.info("Trained model meets deployment criteria. Proceeding to push to Azure Blob Storage.")
            
            # Step 8: Deploy model
            model_pusher_artifact = self.start_model_pusher(
                model_trainer_artifact=model_trainer_artifact
            )

            # Step 9: Deploy transformation artifacts
            model_pusher_artifact = self.start_transformation_pusher(
                data_transformation_artifact=data_transformation_artifact,
                model_pusher_artifact=model_pusher_artifact,
            )

            logging.info("DeploymentPipeline execution completed successfully.")
            return model_pusher_artifact
            
        except Exception as e:
            raise MyException(e, sys) from e
