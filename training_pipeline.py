import logging
import os
import pandas as pd

from steps.ingest_data import ingest_data
from steps.split_data import split_data
from steps.clean_data import clean_data
from steps.feature_engineering import feature_engineer
from steps.handle_missing import handle_missing
from steps.handle_outliers import handle_outliers
from steps.encode_categorical import encode_categorical

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline(raw_path: str = "data/raw/used_cars.csv", output_dir: str = "data/processed") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the end-to-end data preprocessing pipeline and returns the train and test DataFrames."""
    logger.info("Starting end-to-end training pipeline...")
    
    # Ingest
    df = ingest_data(raw_path)
    logger.info("Ingested data shape: %s", df.shape)
    
    # Split
    data = split_data(df)
    logger.info("Split data into train: %s and test: %s", data[0].shape, data[1].shape)
    
    # Clean
    data = clean_data(data)
    logger.info("Cleaned data shape - train: %s, test: %s", data[0].shape, data[1].shape)
    
    # Feature Engineer
    data = feature_engineer(data)
    logger.info("Feature engineered shape - train: %s, test: %s", data[0].shape, data[1].shape)
    
    # Handle Missing Values
    data = handle_missing(data)
    logger.info("Imputed missing values shape - train: %s, test: %s", data[0].shape, data[1].shape)
    
    # Handle Outliers (IQR)
    data = handle_outliers(data)
    logger.info("Outliers handled (IQR) shape - train: %s, test: %s", data[0].shape, data[1].shape)
    
    # Categorical Encoding
    data = encode_categorical(data)
    logger.info("Categorically encoded shape - train: %s, test: %s", data[0].shape, data[1].shape)

    #
    
    
    # Save the files
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df, test_df = data
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info("Train dataset successfully saved to %s", train_path)
    logger.info("Test dataset successfully saved to %s", test_path)
    
    logger.info(f"Train_df: {train_df.shape}, Test_df :{test_df.shape}")
    return train_df, test_df


if __name__ == "__main__":
    run_training_pipeline()
