from zenml import pipeline
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model
"""caching is useful when we dont want to re run the entire model, if we have the saem code with same parameter, we dont want to waste the time adn esources, the results will be saved locally, default caching is true, setting the cached =false, is useful when,u rneed to rerun the model every time wiht the updated data"""
@pipeline(enable_cache=False)
def train_pipeline(data_path:str):
    df=ingest_data(data_path=data_path)
    X_train,X_test,Y_train,Y_test=clean_df(df)
    model=train_model(X_train,X_test,Y_train,Y_test)
    evaluation_metrics=evaluate_model(model,X_test,Y_test)

