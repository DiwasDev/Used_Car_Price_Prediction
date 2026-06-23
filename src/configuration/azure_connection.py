from azure.storage.blob import BlobServiceClient
from typing import Optional
import os
import sys

from src.logger import logging
from src.exception import MyException

from src.constants import (
    AZURE_STORAGE_CONNECTION_STRING,
    # AZURE_ACCOUNT_NAME,
    # AZURE_ACCOUNT_KEY,
)

class AzureConnection:
    def __init__(self) -> None:
        """
        Initializes the AzureConnection class.
        """
        self.connection_string = AZURE_STORAGE_CONNECTION_STRING
        pass

    def get_blob_service_client(self) -> BlobServiceClient:
        """
        Return a BlobServiceClient using the connection string (preferred) or
        account name + key. Raises ValueError if neither is provided.
        """
        # Use connection string if set
        conn_str = AZURE_STORAGE_CONNECTION_STRING
        if not conn_str:
            logging.error("No Azure credentials found. Set AZURE_STORAGE_CONNECTION_STRING in environment or constants.")
            raise ValueError(
                "No Azure credentials found. Set AZURE_STORAGE_CONNECTION_STRING "
                "or AZURE_ACCOUNT_NAME and AZURE_ACCOUNT_KEY in environment or constants."
            )

        try:
            client = BlobServiceClient.from_connection_string(self.connection_string)
            return client
        except Exception as e:
            raise MyException(e, sys) from e

