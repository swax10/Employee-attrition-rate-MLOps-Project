from zenml import pipeline
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from steps.config import ModelNameConfig

@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    """
    Training pipeline that handles data ingestion, cleaning, model training and evaluation.
    
    Args:
        data_path: Path to the training data
    """
    df = ingest_data(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=ModelNameConfig()
    )
    evaluation_metrics = evaluate_model(model, X_test, y_test)
