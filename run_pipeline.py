from sklearn.base import ClassifierMixin
from zenml import pipeline
from pipelines.training_pipeline import train_pipeline
from zenml.client import Client
import pandas as pd

if __name__ == "__main__":
    uri = Client().active_stack.experiment_tracker.get_tracking_uri()
    print(uri)
    train_pipeline(data_path="./data/HR-Employee-Attrition.csv")


