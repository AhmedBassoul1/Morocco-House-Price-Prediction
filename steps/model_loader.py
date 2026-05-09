from sklearn.pipeline import Pipeline
from zenml import Model, step


@step
def model_loader(model_name: str) -> Pipeline:
    """
    Loads the current production model pipeline.

    Args:
        model_name: Name of the Model to load.

    Returns:
        Pipeline: The loaded scikit-learn pipeline.
    """
    model = Model(name=model_name, version="production")
    model_pipeline: Pipeline = model.load_artifact("sklearn_pipeline")
    return model_pipeline