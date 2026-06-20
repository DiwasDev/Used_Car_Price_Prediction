import os
import sys
import pandas as pd
from src.components.data_splitter import TrainTestSplitStrategy
from src.entity.config_entity import DataSplitterConfig
from src.entity.artifact_entity import DataSplitterArtifact, DataIngestionArtifact
from src.exception import MyException

def split_data(
    config_or_df = DataSplitterConfig,
    ingestion_artifact: DataIngestionArtifact | None = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame] | DataSplitterArtifact:
    """
    Splits the dataframe or feature store file into train and test sets.

    Parameters:
        config_or_df: DataSplitterConfig or pd.DataFrame: Config object or input DataFrame.
        ingestion_artifact: DataIngestionArtifact: Ingested data artifact.
        test_size: float: Proportion of the dataset to include in the test split.
        random_state: int: Controls the shuffling applied to the data.

    Returns:
        result: tuple[pd.DataFrame, pd.DataFrame] or DataSplitterArtifact: Train/Test data or split artifact.
    """
    try:
        if isinstance(config_or_df, pd.DataFrame):
            splitter = TrainTestSplitStrategy(test_size=test_size, random_state=random_state)
            train_df, test_df = splitter.split(config_or_df)
            return train_df, test_df
        else:
            config = config_or_df
            df = pd.read_csv(ingestion_artifact.feature_store_file_path)
            
            splitter = TrainTestSplitStrategy(
                test_size=config.train_test_split_ratio,
                random_state=config.random_state
            )
            train_df, test_df = splitter.split(df)
            
            # Exporting the files is implemented in the step
            dir_path = os.path.dirname(config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            train_df.to_csv(config.training_file_path, index=False, header=True)
            test_df.to_csv(config.testing_file_path, index=False, header=True)
            
            return DataSplitterArtifact(
                trained_file_path=config.training_file_path,
                test_file_path=config.testing_file_path
            )
    except Exception as e:
        raise MyException(e, sys)
