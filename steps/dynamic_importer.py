import pandas as pd
from zenml import step


@step
def dynamic_importer() -> str:
    """Dynamically imports sample data for testing the Moroccan housing model.

    Returns a JSON string in pandas 'split' orientation. Note: the 'price' target
    and the leaky 'price_per_m2' column are intentionally NOT included.
    """
    # Sample inference batch matching the training schema (after data_ingestion_step
    # drops 'id' and 'price_per_m2', and excluding the target 'price')
    data = {
        "bedrooms": [2, 3],
        "bathrooms": [1, 2],
        "surface_m2": [98, 150],
        "elevator": [True, True],
        "floor": [5, 3],
        "terrace": [True, True],
        "parking": [False, True],
        "property_type": ["Appartement", "Appartement"],
        "city": ["Casablanca", "Casablanca"],
    }

    df = pd.DataFrame(data)

    json_data = df.to_json(orient="split")
    return json_data