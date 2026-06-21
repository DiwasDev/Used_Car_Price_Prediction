import sys
import pandas as pd
import numpy as np
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, RegressionMetricArtifact
from src.entity.estimator import MyModel
from src.components.modeler.model_builder import ModelBuilder, LinearRegressionStrategy
from src.components.modeler.evaluator import ModelEvaluator
from src.constants import TARGET_COLUMN

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Transformed train-test data loaded.")
            
            # Split features and target 
            logging.info(f"Training data {train_arr[:, 3]}")
            X_train, y_train =  np.delete(train_arr, 3, axis=1),  train_arr[:, 3]
            X_test, y_test =  np.delete(test_arr, 3, axis=1),  test_arr[:, 3]
            
            # Train model using LinearRegressionStrategy
            logging.info("Training Linear Regression model using ModelBuilder strategy...")
            strategy = LinearRegressionStrategy()
            builder = ModelBuilder(strategy)
            trained_model = builder.build_model(pd.DataFrame(X_train), pd.Series(y_train))
            
            # Evaluate using ModelEvaluator
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_model(trained_model, pd.DataFrame(X_test), pd.Series(y_test))
            
            r2 = metrics["R2"]
            mae = metrics["MAE"]
            mse = metrics["MSE"]
            rmse = metrics["RMSE"]
            
            metric_artifact = RegressionMetricArtifact(r2_score=r2, mae=mae, mse=mse, rmse=rmse)
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing object loaded.")
            
            # Check threshold
            if r2 < self.model_trainer_config.expected_accuracy:
                logging.info(f"Trained model R2 score ({r2}) is below the expected r2 score ({self.model_trainer_config.expected_accuracy})")
                raise Exception("Trained model R2 score is below the expected accuracy threshold.")
                
            # Save final MyModel object
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object containing preprocessor and estimator.")
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )
            logging.info(f"Model trainer artifact created: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise MyException(e, sys) from e
