import mlflow
from zenml import step
import pandas as pd
from sklearn.base import ClassifierMixin
import logging
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import ClassificationReport 
from zenml.client import Client
#The  zenml.client.Client  is a class from the ZenML library that allows you to interact with the ZenML platform. It provides functionalities to create, manage, and execute machine learning pipelines. 
 
experiment_tracker=Client().active_stack.experiment_tracker
#   the ZenML stack is the overall collection of components and configurations used in ZenML, while the active stack refers to the specific stack that is currently active and relevant in the ZenML execution context. The active stack is a subset of the ZenML stack, containing the components and configurations applicable to the current pipeline or experiment.ZenML client is used to interact with the ZenML platform, MLflow is a platform for managing the machine learning lifecycle,MLflow offers experiment tracking, model versioning, model deployment, and other features to help streamline the machine learning workflow.   ZenML and MLflow can be used together, with ZenML providing additional functionalities and abstractions for managing machine learning workflows on top of MLflow.
# Here, In this line, we are creating a variable called  experiment_tracker  and assigning it the value of the experiment tracker from the active stack.  
 
#    - The  Client()  creates an instance of the ZenML client, which allows interaction with the ZenML platform. 
#    - The  active_stack  refers to the active stack in ZenML, which represents the current execution context. 
#    -  experiment_tracker  is a component in ZenML that tracks and logs information related to experiments, such as pipeline runs, metrics, and artifacts. 
 
@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin,
                   X_test: pd.DataFrame,
                   Y_test: pd.DataFrame) -> Annotated[float, "classification_report"]:
    try:
        prediction = model.predict(X_test)
        report_class = ClassificationReport()  # Create an instance of ClassificationReport
        report = report_class.calculate_scores(Y_test, prediction)
        
        # Log individual metrics
        mlflow.log_metrics({
            "precision_0": report["0"]["precision"],
            "recall_0": report["0"]["recall"],
            "f1_score_0": report["0"]["f1-score"],
            "precision_1": report["1"]["precision"],
            "recall_1": report["1"]["recall"],
            "f1_score_1": report["1"]["f1-score"],
            "accuracy": report["accuracy"],
        })
        return report["accuracy"]
    except Exception as e:
        logging.error(f"Error in evaluating the model: {e}")
        raise e
