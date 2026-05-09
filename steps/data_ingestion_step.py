import os

import pandas as pd
from zenml import step

# Columns to drop on ingestion:
# - 'id'           : no predictive value
# - 'price_per_m2' : data leakage — derived from the target (price / surface_m2)
COLUMNS_TO_DROP = ["id", "price_per_m2"]


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file to ingest.

    Returns:
        pd.DataFrame: The ingested data, with leaky/non-predictive columns dropped.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension != ".csv":
        raise ValueError(
            f"Unsupported file extension: {file_extension}. Only .csv files are supported."
        )

    df = pd.read_csv(file_path)

    # Drop columns that shouldn't be features
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)

    return df