import sys
from src.exception import MyException
from src.logger import logging

from steps.ingest_data import ingest_data
from steps.split_data import split_data
from steps.transform_data import transform_data

from src.components.data_validation import DataValidation

from src.entity.config_entity import (
    DataIngestionConfig,
    DataSplitterConfig,
    DataValidationConfig,
    DataTransformationConfig,
)
                                          
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataSplitterArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
)


class TrainPipeline:
    """
    Orchestration pipeline class to run training/preprocessing workflows.
    """

    def __init__(self) -> None:
        """
        Initializes configurations for all pipeline steps.

        Parameters:
            None

        Returns:
            None
        """
        try:
            self.data_ingestion_config = DataIngestionConfig()
            self.data_splitter_config = DataSplitterConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
        except Exception as e:
            raise MyException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Runs the data ingestion step to fetch raw data.

        Parameters:
            None

        Returns:
            artifact: DataIngestionArtifact: Artifact containing the ingested file path.
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            data_ingestion_artifact = ingest_data(config_or_path=self.data_ingestion_config)
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_splitter(self, data_ingestion_artifact: DataIngestionArtifact) -> DataSplitterArtifact:
        """
        Runs the data splitter step to split the data.

        Parameters:
            data_ingestion_artifact: DataIngestionArtifact: Artifact from data ingestion.

        Returns:
            artifact: DataSplitterArtifact: Artifact containing train/test split file paths.
        """
        try:
            logging.info("Entered the start_data_splitter method of TrainPipeline class")
            splitter_artifact = split_data(
                config_or_df=self.data_splitter_config,
                ingestion_artifact=data_ingestion_artifact
            )
            logging.info("Exited the start_data_splitter method of TrainPipeline class")
            return splitter_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_validation(self, splitter_artifact: DataSplitterArtifact) -> DataValidationArtifact:
        """
        Runs the data validation step on split datasets.

        Parameters:
            splitter_artifact: DataSplitterArtifact: Artifact from data splitting.

        Returns:
            artifact: DataValidationArtifact: Artifact containing validation results.
        """
        try:
            logging.info("Entered the start_data_validation method of TrainPipeline class")
            data_validation = DataValidation(
                splitter_artifact=splitter_artifact,
                data_validation_config=self.data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_transformation(
        self,
        splitter_artifact: DataSplitterArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        Runs the data transformation step using the unified preprocessing pipeline.

        Parameters:
            splitter_artifact: DataSplitterArtifact: Artifact from data splitting.
            data_validation_artifact: DataValidationArtifact: Artifact from data validation.

        Returns:
            artifact: DataTransformationArtifact: Artifact containing transformed file paths.
        """
        try:
            logging.info("Entered the start_data_transformation method of TrainPipeline class")
            data_transformation_artifact = transform_data(
                config=self.data_transformation_config,
                splitter_artifact=splitter_artifact,
                validation_artifact=data_validation_artifact
            )
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(self) -> None:
        """
        Runs the full training/preprocessing pipeline.

        Parameters:
            None

        Returns:
            None
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_splitter_artifact = self.start_data_splitter(data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = self.start_data_validation(splitter_artifact=data_splitter_artifact)
            data_transformation_artifact = self.start_data_transformation(
                splitter_artifact=data_splitter_artifact,
                data_validation_artifact=data_validation_artifact
            )
            logging.info("Entire pipeline run completed successfully.")
        except Exception as e:
            raise MyException(e, sys)
