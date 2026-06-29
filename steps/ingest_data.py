import os
import sys
import pandas as pd
from src.logger import logging
from src.components.data_ingestion import MongoDataIngestion, CSVDataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException

def ingest_data(config_or_path: DataIngestionConfig = DataIngestionConfig | str) -> pd.DataFrame | DataIngestionArtifact:
    """
    Ingests data using the appropriate DataIngestion strategy/template.

    Parameters:
        config_or_path: DataIngestionConfig or str: Path to raw file or ingestion config object.

    Returns:
        result: pd.DataFrame or DataIngestionArtifact: Dataframe or Ingestion artifact.
    """
    try:
        logging.info("Initiating data ingestion step...")
        if isinstance(config_or_path, str):
            logging.info(f"Ingesting data directly from CSV file path: {config_or_path}")
            ingestion = CSVDataIngestion()
            df = ingestion.ingest(config_or_path)
            logging.info("Direct CSV data ingestion completed successfully.")
            return df
        else:
            config = config_or_path
            if config.collection_name:
                logging.info(f"Ingesting data from MongoDB collection: {config.collection_name}")
                ingestion = MongoDataIngestion(collection_name=config.collection_name)
                df = ingestion.ingest()
            else:
                logging.info(f"Ingesting data from fallback CSV path: {config.feature_store_file_path}")
                ingestion = CSVDataIngestion()
                df = ingestion.ingest(config.feature_store_file_path)
            
            # Exporting the file is implemented in the step
            dir_path = os.path.dirname(config.feature_store_file_path)
            logging.info(f"Creating parent directories for feature store: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Saving ingested data as CSV to: {config.feature_store_file_path}")
            df.to_csv(config.feature_store_file_path, index=False, header=True)
            
            logging.info("MongoDB/CSV config-based data ingestion step completed successfully.")
            return DataIngestionArtifact(feature_store_file_path=config.feature_store_file_path)
    except Exception as e:
        logging.error("Exception occurred during data ingestion step.")
        raise MyException(e, sys)
