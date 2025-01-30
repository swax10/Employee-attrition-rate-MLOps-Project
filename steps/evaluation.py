import mlflow
from zenml import step
import pandas as pd
from sklearn.base import ClassifierMixin
import logging
from typing import Tuple
from typing_extensions import Annotated
from src.evaluation import ClassificationReport
from zenml.client import Client
import matplotlib.pyplot as plt

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin,
                  X_test: pd.DataFrame,
                  Y_test: pd.Series) -> Annotated[float, "classification_report"]:
    """Evaluate the model and log metrics to MLflow"""
    try:
        # Make predictions and calculate metrics
        prediction = model.predict(X_test)
        report_class = ClassificationReport()
        report = report_class.calculate_scores(Y_test, prediction)
        
        # Get the current MLflow run if exists, otherwise create new one
        active_run = mlflow.active_run()
        if active_run is None:
            mlflow.set_experiment("employee-attrition")
            active_run = mlflow.start_run()
            
        # Log all metrics
        mlflow.log_metrics({
            "accuracy": report["accuracy"],
            "precision_0": report["0"]["precision"],
            "recall_0": report["0"]["recall"],
            "f1_score_0": report["0"]["f1-score"],
            "precision_1": report["1"]["precision"],
            "recall_1": report["1"]["recall"],
            "f1_score_1": report["1"]["f1-score"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"]
        })
        
        # Log confusion matrix
        fig = report_class.plot_confusion_matrix(Y_test, prediction)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        
        # Log feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Create feature importance plot
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
            plt.xticks(rotation=45, ha='right')
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close(fig)
        elif hasattr(model, 'coef_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            # Create feature importance plot
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
            plt.xticks(rotation=45, ha='right')
            plt.title('Top 10 Feature Importances (Absolute Coefficients)')
            plt.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close(fig)
        
        # Only end run if we created it
        if active_run.info.run_id != mlflow.active_run().info.run_id:
            mlflow.end_run()
            
        return report["accuracy"]
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
