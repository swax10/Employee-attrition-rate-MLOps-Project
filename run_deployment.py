import click
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)
import mlflow
from rich import print

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
    default=0.1,
    help="Minimum accuracy required for model deployment",
)
@click.option(
    "--data-path",
    default="./data/HR-Employee-Attrition.csv",
    help="Path to the data file",
)
def main(config: str, min_accuracy: float, data_path: str):
    """Run the deployment pipeline"""
    if config in [DEPLOY, DEPLOY_AND_PREDICT]:
        print(
            "Launching deployment pipeline with minimum accuracy threshold "
            f"{min_accuracy}..."
        )
        continuous_deployment_pipeline(
            min_accuracy=min_accuracy,
            data_path=data_path,
        )

    if config in [PREDICT, DEPLOY_AND_PREDICT]:
        print("\nLaunching prediction pipeline...")
        inference_pipeline(data_path=data_path)

    print("\nPipeline run complete.")
    print(
        "\nTo view the MLflow UI, run:\n"
        "    mlflow ui\n"
        "Then visit http://localhost:5000"
    )

if __name__ == "__main__":
    main()