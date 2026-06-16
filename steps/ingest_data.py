import pandas as pd
from src.data_ingestion import CSVDataIngestion

def ingest_data(file_path: str) -> pd.DataFrame:
    """Ingests raw CSV data and returns a DataFrame."""
    ingestion = CSVDataIngestion()
    data = ingestion.ingest(file_path)
    return data
