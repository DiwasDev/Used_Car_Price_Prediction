import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import r2_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ModelEvaluationArtifact
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
from src.entity.s3_estimator import Proj1Estimator

@dataclass
class EvaluateModelResponse:
    trained_model_r2_score: float
    best_model_r2_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        """
        :param model_eval_config: Configuration for model evaluation
        :param splitter_artifact: Output reference of data splitter artifact stage
        :param model_trainer_artifact: Output reference of model trainer artifact stage
        """
        try:
            self.model_eval_config = model_eval_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Retrieves the best model from production/registry.
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name, model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise MyException(e, sys) from e

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Compares the newly trained model's performance with the best/production model.
        """
        try:
            # Read split test data (raw/untransformed DataFrame)
            test_df = pd.read_csv(self.splitter_artifact.test_file_path)
            x_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            logging.info("Test data loaded for evaluation.")

            # Load the new trained model (MyModel instance)
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded.")

            # Calculate predictions on raw test features
            y_pred_new = trained_model.predict(x_test)
            trained_model_r2_score = r2_score(y_test, y_pred_new)
            logging.info(f"R2 score for new trained model: {trained_model_r2_score}")

            best_model_r2_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info("Production model found. Computing R2 score for production model...")
                y_pred_best = best_model.predict(x_test)
                best_model_r2_score = r2_score(y_test, y_pred_best)
                logging.info(f"R2 score for production model: {best_model_r2_score}")

            tmp_best_model_score = 0.0 if best_model_r2_score is None else best_model_r2_score
            difference = trained_model_r2_score - tmp_best_model_score

            # Accept if new model is better or if there's no production model
            is_model_accepted = difference > self.model_eval_config.changed_threshold_score or best_model_r2_score is None

            result = EvaluateModelResponse(
                trained_model_r2_score=trained_model_r2_score,
                best_model_r2_score=best_model_r2_score if best_model_r2_score is not None else 0.0,
                is_model_accepted=is_model_accepted,
                difference=difference
            )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiates all steps of model evaluation and returns the evaluation artifact.
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
