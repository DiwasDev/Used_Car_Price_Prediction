import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataIngestionTemplate(ABC):
    """Abstract base class defining the data ingestion pipeline."""

    def ingest(self, source: str) -> pd.DataFrame:
        """Template method: orchestrates the ingestion workflow."""
        try:
            self.validate_source(source)
            raw_data = self.read_data(source)
            return self.transform(raw_data)
        except Exception as e:
            logger.error("Error in DataIngestionTemplate.ingest for source %s: %s", source, e)
            raise e

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Default/optional transform step in ingestion pipeline."""
        try:
            return df
        except Exception as e:
            logger.error("Error in DataIngestionTemplate.transform: %s", e)
            raise e

    @abstractmethod
    def validate_source(self, source: str) -> None:
        """Validate that the source exists and is accessible."""
        ...

    @abstractmethod
    def read_data(self, source: str) -> pd.DataFrame:
        """Read raw data from the source."""
        ...


class CSVDataIngestion(DataIngestionTemplate):
    """Ingest data from CSV files."""

    def validate_source(self, source: str) -> None:
        try:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found: {source}")
            if not path.is_file():
                raise ValueError(f"Source is not a file: {source}")
            if path.suffix.lower() != ".csv":
                raise ValueError(f"Source is not a CSV file: {source}")
        except Exception as e:
            logger.error("Error in CSVDataIngestion.validate_source for source %s: %s", source, e)
            raise e

    def read_data(self, source: str) -> pd.DataFrame:
        try:
            return pd.read_csv(source)
        except Exception as e:
            logger.error("Error in CSVDataIngestion.read_data for source %s: %s", source, e)
            raise e

