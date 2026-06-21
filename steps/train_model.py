import sys
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.exception import MyException

def train_model(
    config: ModelTrainerConfig,
    transformation_artifact: DataTransformationArtifact
) -> ModelTrainerArtifact:
    """
    Trains the model using the ModelTrainer component.

    Parameters:
        config: ModelTrainerConfig: Configuration for model training.
        transformation_artifact: DataTransformationArtifact: Transformation artifact containing transformed data paths.

    Returns:
        artifact: ModelTrainerArtifact: Artifact containing trained model file path and metrics.
    """
    try:
        trainer = ModelTrainer(
            data_transformation_artifact=transformation_artifact,
            model_trainer_config=config
        )
        return trainer.initiate_model_trainer()
    except Exception as e:
        raise MyException(e, sys)
