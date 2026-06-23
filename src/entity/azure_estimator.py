import os
import tempfile
from io import BytesIO
from typing import Any, List, Optional
import joblib

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobProperties, BlobServiceClient

from src.configuration.azure_connection import AzureConnection
from src.constants import MODEL_CONTAINER_NAME, MODEL_PUSHER_BLOB_PATH
# Import your custom logging utility
from src.logger import logging

# Set up module-level logging
logger = logging.getLogger(__name__)


class AzureEstimator:
    def __init__(self, blob_service_client: Optional[BlobServiceClient] = None) -> None:
        """
        Initializes the AzureEstimator class. Accepts an optional pre-configured client 
        to facilitate dependency injection during testing.
        """
        if blob_service_client:
            logger.info("Using injected BlobServiceClient instance.")
            self.client = blob_service_client
        else:
            logger.info("Initializing Azure Blob Service Client via AzureConnection...")
            self.client = AzureConnection().get_blob_service_client()
        logger.info("AzureEstimator successfully initialized.")

    def ensure_container_exists(self, container_name: str = MODEL_CONTAINER_NAME) -> None:
        """Optimistically checks and ensures the storage container exists."""
        logger.info(f"Checking if container '{container_name}' exists.")
        container_client = self.client.get_container_client(container_name)
        
        if not container_client.exists():
            logger.info(f"Container '{container_name}' not found. Attempting creation...")
            try:
                container_client.create_container()
                logger.info(f"Successfully created container: '{container_name}'")
            except ResourceExistsError:
                logger.warning(
                    f"Container '{container_name}' was created concurrently by another process."
                )
        else:
            logger.info(f"Container '{container_name}' already exists.")

    def upload_model_file(
        self, local_file_path: str, blob_name: Optional[str] = None, container_name: str = MODEL_CONTAINER_NAME
    ) -> str:
        """Uploads a local file to the blob container securely. Returns the destination blob name."""
        logger.info(f"Starting file upload. local_path='{local_file_path}', container='{container_name}'")
        
        if not os.path.exists(local_file_path):
            error_msg = f"Local file not found at path: {local_file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.ensure_container_exists(container_name)
        destination_blob = blob_name or MODEL_PUSHER_BLOB_PATH
        
        blob_client = self.client.get_blob_client(container=container_name, blob=destination_blob)
        
        logger.info(f"Uploading file '{local_file_path}' to blob path '{destination_blob}'...")
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            
        logger.info(f"Successfully uploaded file to blob '{destination_blob}'")
        return destination_blob

    def download_blob_to_file(self, blob_name: str, download_path: str, container_name: str = MODEL_CONTAINER_NAME) -> str:
        """Downloads a blob directly to a local disk path using an optimized stream write."""
        logger.info(f"Starting blob download. blob='{blob_name}', destination='{download_path}', container='{container_name}'")
        
        abs_download_path = os.path.abspath(download_path)
        logger.info(f"Ensuring local directory structure exists for: {abs_download_path}")
        os.makedirs(os.path.dirname(abs_download_path), exist_ok=True)
        
        blob_client = self.client.get_blob_client(container=container_name, blob=blob_name)
        
        logger.info(f"Streaming data from blob '{blob_name}' to local file...")
        with open(download_path, "wb") as local_file:
            blob_client.download_blob().download_to_stream(local_file)
            
        logger.info(f"Successfully downloaded blob to local path: '{download_path}'")
        return download_path

    def list_blobs_with_prefix(
        self, prefix: str = MODEL_PUSHER_BLOB_PATH, container_name: str = MODEL_CONTAINER_NAME
    ) -> List[BlobProperties]:
        """Returns a list of BlobProperties for blobs that match the given prefix filtering."""
        logger.info(f"Listing blobs in container '{container_name}' with prefix '{prefix}'")
        container_client = self.client.get_container_client(container_name)
        blobs_list = list(container_client.list_blobs(name_starts_with=prefix))
        
        logger.info(f"Found {len(blobs_list)} matching blob(s) for prefix '{prefix}'")
        return blobs_list

    def download_latest_model(
        self, prefix: str = MODEL_PUSHER_BLOB_PATH, download_dir: str = ".", container_name: str = MODEL_CONTAINER_NAME
    ) -> str:
        """Identifies the newest blob matching the prefix by modification time and downloads it."""
        logger.info(f"Locating latest model blob using prefix '{prefix}' in container '{container_name}'")
        blobs = self.list_blobs_with_prefix(prefix=prefix, container_name=container_name)
        
        if not blobs:
            error_msg = f"No blobs found with prefix '{prefix}' in container '{container_name}'"
            logger.error(error_msg)
            raise ResourceNotFoundError(error_msg)
        
        latest_blob = max(blobs, key=lambda b: b.last_modified)
        logger.info(f"Identified newest blob: '{latest_blob.name}' (Last Modified: {latest_blob.last_modified})")
        
        local_path = os.path.join(download_dir, os.path.basename(latest_blob.name))
        return self.download_blob_to_file(latest_blob.name, local_path, container_name=container_name)

    def upload_model_object(
        self, model_object: Any, blob_name: Optional[str] = None, container_name: str = MODEL_CONTAINER_NAME
    ) -> str:
        """Serializes and uploads a scikit-learn/joblib model object cleanly using a secure tempfile."""
        destination_blob = blob_name or MODEL_PUSHER_BLOB_PATH
        logger.info(f"Serializing model object for upload to blob '{destination_blob}'")
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name
            logger.info(f"Serializing model object to temporary local file: '{tmp_path}'")
            joblib.dump(model_object, tmp_path)
            
        try:
            logger.info("Temporary file serialization complete. Proceeding with upload workflow.")
            uploaded_blob = self.upload_model_file(tmp_path, blob_name=destination_blob, container_name=container_name)
            return uploaded_blob
        finally:
            if os.path.exists(tmp_path):
                logger.info(f"Cleaning up temporary file: '{tmp_path}'")
                os.remove(tmp_path)

    def download_model_object(self, blob_name: str, container_name: str = MODEL_CONTAINER_NAME) -> Any:
        """Downloads a model directly into memory and deserializes it, bypassing local disk footprints."""
        logger.info(f"Downloading model blob '{blob_name}' from container '{container_name}' directly into memory...")
        blob_client = self.client.get_blob_client(container=container_name, blob=blob_name)
        
        buffer = BytesIO()
        blob_client.download_blob().download_to_stream(buffer)
        buffer.seek(0)
        
        logger.info("In-memory download complete. Deserializing model object via joblib...")
        model = joblib.load(buffer)
        
        logger.info("Model object successfully deserialized.")
        return model