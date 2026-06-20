import os
import sys
import pandas as pd
from abc import ABC, abstractmethod
from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data

class DataIngestion(ABC):
    """
    Abstract base class for Data Ingestion using the Template Method design pattern.
    """

    def ingest(self, *args, **kwargs) -> pd.DataFrame:
        """
        Template method defining the skeleton of data ingestion.

        Parameters:
            *args: Any: Variable length argument list.
            **kwargs: Any: Arbitrary keyword arguments.

        Returns:
            df: pd.DataFrame: Ingested dataframe.
        """
        try:
            logging.info("Starting the data ingestion template method.")
            df = self.fetch_data(*args, **kwargs)
            df = self.process_data(df)
            logging.info("Data ingestion completed successfully.")
            return df
        except Exception as e:
            raise MyException(e, sys)

    @abstractmethod
    def fetch_data(self, *args, **kwargs) -> pd.DataFrame:
        """
        Abstract method to fetch data from the source.

        Parameters:
            *args: Any: Variable length argument list.
            **kwargs: Any: Arbitrary keyword arguments.

        Returns:
            df: pd.DataFrame: Fetched dataframe.
        """
        pass

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optional step to process/clean the fetched data.

        Parameters:
            df: pd.DataFrame: Input dataframe to process.

        Returns:
            df: pd.DataFrame: Processed dataframe.
        """
        try:
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
            return df
        except Exception as e:
            raise MyException(e, sys)


class MongoDataIngestion(DataIngestion):
    """
    Concrete implementation of DataIngestion for MongoDB.
    """

    def __init__(self, collection_name: str) -> None:
        """
        Initializes MongoDB data ingestion.

        Parameters:
            collection_name: str: Name of the MongoDB collection.

        Returns:
            None
        """
        try:
            self.collection_name = collection_name
        except Exception as e:
            raise MyException(e, sys)

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches data from MongoDB collection.

        Parameters:
            None

        Returns:
            df: pd.DataFrame: Data fetched from MongoDB.
        """
        try:
            logging.info(f"Fetching data from MongoDB collection: {self.collection_name}")
            my_data = Proj1Data()
            df = my_data.export_collection_as_dataframe(collection_name=self.collection_name)
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
            return df
        except Exception as e:
            raise MyException(e, sys)


class CSVDataIngestion(DataIngestion):
    """
    Concrete implementation of DataIngestion for CSV files.
    """

    def fetch_data(self, file_path: str) -> pd.DataFrame:
        """
        Fetches data from a CSV file.

        Parameters:
            file_path: str: Path to the CSV file.

        Returns:
            df: pd.DataFrame: Data loaded from CSV.
        """
        try:
            logging.info(f"Reading CSV file from path: {file_path}")
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise MyException(e, sys)
