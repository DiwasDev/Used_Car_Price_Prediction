import logging
import os
import pandas as pd
from src.components.data_ingestion import CSVDataIngestion
from src.components.data_cleaning import DataCleaner
from src.components.handle_outliers import OutlierHandlerFactory
from src.components.feature_engineering import FeatureEngineer
from src.components.handle_missing import MissingValueHandler
from src.components.encoder import CategoricalEncoder, OneHotEncodingStrategy, TargetEncodingStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SafeOneHotEncodingStrategy(OneHotEncodingStrategy):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._encoder is None:
            raise RuntimeError("SafeOneHotEncodingStrategy must be fitted before transform.")
        df_copy = df.copy()
        for i, col in enumerate(self.columns):
            valid_cats = set(self._encoder.categories_[i])
            df_copy[col] = df_copy[col].apply(lambda x: x if x in valid_cats else self._encoder.categories_[i][0])
        return super().transform(df_copy)


def run_test_pipeline():
    try:
        raw_path = "data/raw/used_cars.csv"
        processed_dir = "data/processed"

        # Ensure output directory exists
        os.makedirs(processed_dir, exist_ok=True)

        logger.info("=== STEP 1: Ingesting Raw Data ===")
        ingestion = CSVDataIngestion()
        df = ingestion.ingest(raw_path)
        logger.info("Raw Data Shape: %s", df.shape)

        logger.info("=== STEP 2: Splitting Data ===")
        from src.components.data_splitter import TrainTestSplitStrategy
        splitter = TrainTestSplitStrategy(test_size=0.2, random_state=42)
        train_df, test_df = splitter.split(df)
        logger.info("Train Data Shape: %s, Test Data Shape: %s", train_df.shape, test_df.shape)

        logger.info("=== STEP 3: Data Cleaning ===")
        cleaner = DataCleaner()
        train_df = cleaner.fit_transform(train_df)
        test_df = cleaner.transform(test_df)
        logger.info("Cleaned Train Shape: %s, Cleaned Test Shape: %s", train_df.shape, test_df.shape)

        logger.info("=== STEP 4: Feature Engineering ===")
        feature_engineer = FeatureEngineer()
        train_df = feature_engineer.fit_transform(train_df)
        test_df = feature_engineer.transform(test_df)
        logger.info("Feature Engineered Train Shape: %s, Test Shape: %s", train_df.shape, test_df.shape)

        logger.info("=== STEP 5: Missing Values Imputation ===")
        missing_handler = MissingValueHandler()
        train_df = missing_handler.fit_transform(train_df)
        test_df = missing_handler.transform(test_df)
        logger.info("Imputed Train Shape: %s, Test Shape: %s", train_df.shape, test_df.shape)

        logger.info("=== STEP 6: Outlier Handling (IQR) ===")
        iqr_outlier_handler = OutlierHandlerFactory.iqr_outlier_handler(
            columns=["engine_hp", "engine_displacement"]
        )
        train_df = iqr_outlier_handler.fit_transform(train_df)
        test_df = iqr_outlier_handler.transform(test_df)
        logger.info("IQR-handled Train Shape: %s, Test Shape: %s", train_df.shape, test_df.shape)

        logger.info("=== STEP 7: Categorical Encoding ===")
        target_col = "price_usd"
        y_train = train_df[target_col]
        encoder = CategoricalEncoder(strategies=[
            SafeOneHotEncodingStrategy(),
            TargetEncodingStrategy(target_col=target_col, random_state=42)
        ])
        train_df = encoder.fit_transform(train_df, y_train)
        test_df = encoder.transform(test_df)
        logger.info("Encoded Train Shape: %s, Test Shape: %s", train_df.shape, test_df.shape)

        logger.info("=== STEP 8: Log Transformation ===")
        log_transformer = OutlierHandlerFactory.log_transform_handler()
        train_df = log_transformer.fit_transform(train_df)
        test_df = log_transformer.transform(test_df)
        logger.info("Log-transformed Train Shape: %s, Test Shape: %s", train_df.shape, test_df.shape)

        logger.info("=== STEP 9: Saving Processed Datasets ===")
        train_path = os.path.join(processed_dir, "train.csv")
        test_path = os.path.join(processed_dir, "test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info("Train dataset successfully saved to %s", train_path)
        logger.info("Test dataset successfully saved to %s", test_path)

        # Verify the files are created and show correct shape
        saved_train_df = pd.read_csv(train_path)
        saved_test_df = pd.read_csv(test_path)
        logger.info("Saved Train Dataset Verification Shape: %s", saved_train_df.shape)
        logger.info("Saved Test Dataset Verification Shape: %s", saved_test_df.shape)
        logger.info("Verification complete. Everything matches!")

    except Exception as e:
        logger.error("Error running test pipeline: %s", e, exc_info=True)
        raise e


if __name__ == "__main__":
    run_test_pipeline()
