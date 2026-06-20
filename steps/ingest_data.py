import os
import sys
import pandas as pd
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
        if isinstance(config_or_path, str):
            ingestion = CSVDataIngestion()
            return ingestion.ingest(config_or_path)
        else:
            config = config_or_path
            if config.collection_name:
                ingestion = MongoDataIngestion(collection_name=config.collection_name)
                df = ingestion.ingest()
            else:
                ingestion = CSVDataIngestion()
                df = ingestion.ingest(config.feature_store_file_path)
            
            # Exporting the file is implemented in the step
            dir_path = os.path.dirname(config.feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(config.feature_store_file_path, index=False, header=True)
            
            return DataIngestionArtifact(feature_store_file_path=config.feature_store_file_path)
    except Exception as e:
        raise MyException(e, sys)
