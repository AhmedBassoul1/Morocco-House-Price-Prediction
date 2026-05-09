import json

import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


# Columns the model expects at inference time
# (matches training schema after dropping 'id', 'price_per_m2', and target 'price')
EXPECTED_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "surface_m2",
    "elevator",
    "floor",
    "terrace",
    "parking",
    "property_type",
    "city",
]


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (str): The input data as a JSON string (pandas 'split' orientation).

    Returns:
        np.ndarray: The model's prediction.
    """
    service.start(timeout=10)

    payload = json.loads(input_data)
    columns = payload.get("columns", EXPECTED_COLUMNS)
    rows = payload["data"]

    df = pd.DataFrame(rows, columns=columns)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in input data: {missing}")
    df = df[EXPECTED_COLUMNS]

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    prediction = service.predict(df.to_numpy())
    return prediction