import pandas as pd
import logging
from zenml import step
from src.model_dev import LogisticRegressionModel
from .config import ModelNameConfig
from sklearn.base import ClassifierMixin
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, enable_cache=False)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig = ModelNameConfig()
) -> ClassifierMixin:
    """
    Trains the model on the ingested data.
    
    Args:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Test labels
        config: Model configuration
    
    Returns:
        Trained model
    """
    try:
        model = None
        if config.model_name == "LogisticRegression":
            model = LogisticRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} is not supported")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
