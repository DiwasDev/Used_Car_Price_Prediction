import sys
from src.components.model_evaluation import ModelEvaluation
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ModelEvaluationArtifact
from src.exception import MyException

def evaluate_model(
    config: ModelEvaluationConfig,
    data_transformation_artifact: DataTransformationArtifact,
    model_trainer_artifact: ModelTrainerArtifact
) -> ModelEvaluationArtifact:
    """
    Evaluates the trained model against the production/best model using ModelEvaluation component.

    Parameters:
        config: ModelEvaluationConfig: Configuration for model evaluation.
        splitter_artifact: DataSplitterArtifact: Split datasets containing test file path.
        model_trainer_artifact: ModelTrainerArtifact: Output reference of model trainer stage.

    Returns:
        artifact: ModelEvaluationArtifact: Artifact containing evaluation results.
    """
    try:
        evaluator = ModelEvaluation(
            model_eval_config=config,
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_artifact=model_trainer_artifact
        )
        return evaluator.initiate_model_evaluation()
    except Exception as e:
        raise MyException(e, sys)
