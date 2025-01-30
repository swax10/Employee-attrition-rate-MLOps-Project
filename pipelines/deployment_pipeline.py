import os
import json
import numpy as np
import pandas as pd
from .utils import get_data_for_test
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
import mlflow
from mlflow.tracking import MlflowClient

docker_settings = DockerSettings(required_integrations=[MLFLOW])

##################################
#        Continuous Deployment   #
#              Pipeline          #
##################################

class DeploymentTriggerConfig:
    """Class for configuring deployment trigger"""
    def __init__(self, min_accuracy: float = 0.1):
        self.min_accuracy = min_accuracy

@step 
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig = DeploymentTriggerConfig(),
) -> bool:
    """Trigger model deployment if accuracy meets threshold"""
    return accuracy >= config.min_accuracy

@step
def mlflow_deployment_step(
    model,
    deploy_decision: bool,
) -> None:
    """Deploy model to MLflow model registry"""
    if deploy_decision:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("employee-attrition")
        
        with mlflow.start_run() as run:
            # Log the model to MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="employee_attrition_model"
            )
            
            # Get the latest version and transition it to Production
            client = MlflowClient()
            model_version = client.get_latest_versions("employee_attrition_model", stages=["None"])[0].version
            client.transition_model_version_stage(
                name="employee_attrition_model",
                version=model_version,
                stage="Production"
            )
            
            print(f"Model version {model_version} has been moved to Production")
    else:
        print("Model deployment was skipped due to inadequate performance.")

@pipeline(enable_cache=False)
def continuous_deployment_pipeline(
    data_path: str = "./data/HR-Employee-Attrition.csv",
    min_accuracy: float = 0.1,
):
    """Main deployment pipeline"""
    # Enable MLflow autologging
    mlflow.sklearn.autolog()
    
    # Data ingestion and preprocessing
    df = ingest_data(data_path=data_path)
    X_train, X_test, Y_train, Y_test = clean_df(df)
    
    # Model training and evaluation
    model = train_model(X_train=X_train, X_test=X_test, y_train=Y_train, y_test=Y_test)
    accuracy = evaluate_model(model=model, X_test=X_test, Y_test=Y_test)
    
    # Deployment decision and execution
    deployment_decision = deployment_trigger(accuracy=accuracy)
    mlflow_deployment_step(model=model, deploy_decision=deployment_decision)

##################################
#        Inference               #
#              Pipeline          #
##################################

@step
def get_production_model_uri() -> str:
    """Get the URI of the production model"""
    client = MlflowClient()
    production_model = client.get_latest_versions("employee_attrition_model", stages=["Production"])[0]
    return production_model.source

@step(enable_cache=False)
def predictor(
    data: str,
) -> np.ndarray:
    """Make predictions using the production model"""
    try:
        # Parse input data
        data = json.loads(data)
        data.pop("columns")
        data.pop("index")
        columns_for_df = [
            'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently',
            'BusinessTravel_Travel_Rarely', 'Department_Human Resources',
            'Department_Research & Development', 'Department_Sales',
            'EducationField_Human Resources', 'EducationField_Life Sciences',
            'EducationField_Marketing', 'EducationField_Medical',
            'EducationField_Other', 'EducationField_Technical Degree',
            'Gender_Female', 'Gender_Male', 'JobRole_Healthcare Representative',
            'JobRole_Human Resources', 'JobRole_Laboratory Technician',
            'JobRole_Manager', 'JobRole_Manufacturing Director',
            'JobRole_Research Director', 'JobRole_Research Scientist',
            'JobRole_Sales Executive', 'JobRole_Sales Representative',
            'MaritalStatus_Divorced', 'MaritalStatus_Married',
            'MaritalStatus_Single', 'Age', 'DailyRate', 'DistanceFromHome',
            'Education', 'EnvironmentSatisfaction', 'HourlyRate',
            'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
            'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime',
            'PercentSalaryHike', 'PerformanceRating',
            'RelationshipSatisfaction', 'StockOptionLevel',
            'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
            'YearsAtCompany', 'YearsInCurrentRole',
            'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]
        
        # Create DataFrame
        df = pd.DataFrame(data["data"], columns=columns_for_df)
        
        # Load the production model
        model_uri = get_production_model_uri()
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Make prediction
        prediction = loaded_model.predict(df)
        return prediction
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        raise e

@pipeline(enable_cache=False)
def inference_pipeline():
    """Pipeline for making predictions"""
    data = get_data_for_test()
    prediction = predictor(data=data)
    return prediction