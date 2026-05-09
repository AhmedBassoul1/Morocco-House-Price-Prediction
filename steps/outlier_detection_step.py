import logging

import pandas as pd
from src.outlier_detection import OutlierDetector, ZScoreOutlierDetection
from zenml import step


@step
def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Detects and removes outliers using OutlierDetector.

    Only numeric columns are used for z-score outlier detection, but non-numeric
    columns (e.g. property_type, city) are preserved in the returned DataFrame.
    """
    logging.info(f"Starting outlier detection step with DataFrame of shape: {df.shape}")

    if df is None:
        logging.error("Received a NoneType DataFrame.")
        raise ValueError("Input df must be a non-null pandas DataFrame.")

    if not isinstance(df, pd.DataFrame):
        logging.error(f"Expected pandas DataFrame, got {type(df)} instead.")
        raise ValueError("Input df must be a pandas DataFrame.")

    if column_name not in df.columns:
        logging.error(f"Column '{column_name}' does not exist in the DataFrame.")
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Cast booleans to int so they don't break numeric-only outlier detection
    df = df.copy()
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Numeric subset for the z-score check
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df_numeric = df[numeric_cols]

    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    outliers = outlier_detector.detect_outliers(df_numeric)
    df_numeric_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")

    # Re-attach the non-numeric columns (e.g. property_type, city) using the surviving rows
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    if non_numeric_cols:
        df_cleaned = df_numeric_cleaned.join(df.loc[df_numeric_cleaned.index, non_numeric_cols])
        # Restore original column order
        df_cleaned = df_cleaned[df.columns]
    else:
        df_cleaned = df_numeric_cleaned

    logging.info(f"DataFrame shape after outlier removal: {df_cleaned.shape}")
    return df_cleaned