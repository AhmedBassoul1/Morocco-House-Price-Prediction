from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from steps.outlier_detection_step import outlier_detection_step
from zenml import Model, pipeline


@pipeline(
    model=Model(
        name="prices_predictor",
    ),
)
def ml_pipeline():
    """End-to-end ML pipeline for the Moroccan housing dataset."""

    # Data Ingestion: load the cleaned CSV
    raw_data = data_ingestion_step(
        file_path="/home/ahmed/Desktop/Morocco_House_Price_Prediction/Morocco_House_Price_Pred/cleaned_data/cleaned_housing_data.csv"
    )

    # Handle Missing Values
    filled_data = handle_missing_values_step(raw_data)

    # Feature Engineering: log-transform skewed numeric features
    engineered_data = feature_engineering_step(
        filled_data,
        strategy="log",
        features=["surface_m2", "price"],
    )

    # Outlier Detection on the (transformed) target
    clean_data = outlier_detection_step(engineered_data, column_name="price")

    # Train/Test Split
    X_train, X_test, y_train, y_test = data_splitter_step(
        clean_data, target_column="price"
    )

    # Model Building
    model = model_building_step(X_train=X_train, y_train=y_train)

    # Model Evaluation
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model


if __name__ == "__main__":
    run = ml_pipeline()