from typing import cast
import click
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Choose to run deployment pipeline, prediction pipeline, or both",
)
@click.option(
    "--min-accuracy",
    default=0.9,
    help="Minimum accuracy required for model deployment",
)
@click.option(
    "--workers",
    default=1,
    help="Number of workers to use for deployment",
)
@click.option(
    "--data-path",
    default="./data/HR-Employee-Attrition.csv",
    help="Path to the data file",
)
def main(config: str, min_accuracy: float, workers: int, data_path: str):
    """Run the deployment pipeline"""
    mlflow_model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    
    if config in [DEPLOY, DEPLOY_AND_PREDICT]:
        print(
            "Launching deployment pipeline with minimum accuracy threshold "
            f"{min_accuracy} and {workers} workers..."
        )
        continuous_deployment_pipeline(
            min_accuracy=min_accuracy,
            workers=workers,
            data_path=data_path,
        )

    if config in [PREDICT, DEPLOY_AND_PREDICT]:
        print("\nLaunching prediction pipeline...")
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            data_path=data_path,
        )

    print("\nPipeline run complete. Starting prediction service...")
    # Get the deployment service
    services = mlflow_model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=True,
    )

    if services:
        service = cast(MLFlowDeploymentService, services[0])
        if service.is_running:
            print(
                f"\nPrediction service is running at:\n"
                f"    {service.prediction_url}\n"
                f"    MLflow tracking URI: {get_tracking_uri()}\n"
            )
            print(
                "\nTo see the MLflow UI, run:\n"
                f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'"
            )
        else:
            print(
                "Prediction service is not running. Please ensure the deployment "
                "pipeline completed successfully."
            )
    else:
        print(
            "No prediction service found. Please ensure the deployment pipeline "
            "completed successfully."
        )

if __name__ == "__main__":
    main()