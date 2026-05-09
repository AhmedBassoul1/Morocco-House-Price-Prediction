import sys
from pathlib import Path

import numpy as np
import pandas as pd
from zenml import Model

# Make sure the project root is on the path so any relative imports keep working
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---- Configuration ---------------------------------------------------------

MODEL_NAME = "prices_predictor"
MODEL_VERSION = "latest"  # could also be "production", or a specific version number
ARTIFACT_NAME = "sklearn_pipeline"

# Columns the trained model expects (must match training schema exactly)
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

# These columns were log-transformed in feature_engineering_step during training,
# so the model expects them in log-space.
LOG_TRANSFORMED_FEATURES = ["surface_m2"]

# The target was also log-transformed, so predictions come out in log-space.
TARGET_WAS_LOG_TRANSFORMED = True


# ---- Test samples ----------------------------------------------------------

SAMPLE_HOUSES = [
    {
        "description": "Mid-range apartment in Casablanca",
        "features": {
            "bedrooms": 2,
            "bathrooms": 1,
            "surface_m2": 98,
            "elevator": True,
            "floor": 5,
            "terrace": True,
            "parking": False,
            "property_type": "Appartement",
            "city": "Casablanca",
        },
    },
    {
        "description": "Large villa in Fès with no elevator",
        "features": {
            "bedrooms": 5,
            "bathrooms": 3,
            "surface_m2": 720,
            "elevator": False,
            "floor": 3,
            "terrace": False,
            "parking": False,
            "property_type": "Villa",
            "city": "Fès",
        },
    },
    {
        "description": "Small studio in Marrakech",
        "features": {
            "bedrooms": 1,
            "bathrooms": 1,
            "surface_m2": 45,
            "elevator": True,
            "floor": 2,
            "terrace": False,
            "parking": False,
            "property_type": "Studio",
            "city": "Marrakech",
        },
    },
]


# ---- Helpers ---------------------------------------------------------------

def load_model():
    """Load the trained sklearn pipeline from the ZenML artifact store."""
    print(f"Loading model '{MODEL_NAME}' (version='{MODEL_VERSION}')...")
    zenml_model = Model(name=MODEL_NAME, version=MODEL_VERSION)
    pipeline = zenml_model.load_artifact(ARTIFACT_NAME)
    print(f"  → Loaded: {type(pipeline).__name__}\n")
    return pipeline


def prepare_input(features: dict) -> pd.DataFrame:
    """Turn a single feature dict into a model-ready DataFrame."""
    df = pd.DataFrame([features])

    # Reorder to match training column order
    df = df[EXPECTED_COLUMNS]

    # Cast booleans to int (same as training)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Apply the same log transformation that feature_engineering_step did
    for col in LOG_TRANSFORMED_FEATURES:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    return df


def predict_price(pipeline, features: dict) -> float:
    """Run a single prediction and return the price in original units."""
    X = prepare_input(features)
    raw_pred = pipeline.predict(X)[0]

    # Reverse the log transformation on the target
    if TARGET_WAS_LOG_TRANSFORMED:
        price = np.expm1(raw_pred)
    else:
        price = raw_pred

    return float(price)


# ---- Main ------------------------------------------------------------------

def main():
    pipeline = load_model()

    print("=" * 70)
    print("Predictions on test samples")
    print("=" * 70)

    for i, sample in enumerate(SAMPLE_HOUSES, start=1):
        desc = sample["description"]
        feats = sample["features"]

        try:
            price = predict_price(pipeline, feats)
            # Prices in the CSV are in thousands of MAD (e.g. 100.0 = 100,000 MAD)
            print(f"\n[{i}] {desc}")
            print(f"    Surface : {feats['surface_m2']} m²")
            print(f"    Type    : {feats['property_type']} in {feats['city']}")
            print(f"    Layout  : {feats['bedrooms']} bed / {feats['bathrooms']} bath")
            print(f"    → Predicted price : {price:,.2f}  (≈ {price * 1000:,.0f} MAD)")
        except Exception as e:
            print(f"\n[{i}] {desc}")
            print(f"    ✗ Prediction failed: {e}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()