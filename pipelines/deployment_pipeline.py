import os
import json
import numpy as np
import pandas as pd
from .utils import get_data_for_test
#from materializer.custom_materializer import cs_materializer
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import(MLFlowModelDeployer,)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pydantic import BaseModel
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from typing import cast
import mlflow

docker_settings=DockerSettings(required_integrations=[MLFLOW])

##################################
#        Continuous Deployment   #
#              Pipeline          #
##################################

class DeploymentTriggerConfig(BaseModel):
  """Class for configuring deployment trigger"""
  min_accuracy: float=0.1


@step 
def deployment_trigger(
  accuracy:float,
  config: DeploymentTriggerConfig = DeploymentTriggerConfig(),
)->bool:
  return accuracy>=config.min_accuracy


@pipeline(enable_cache=False, settings={"docker":docker_settings})
def continuous_deployment_pipeline(
   data_path: str = "./data/HR-Employee-Attrition.csv",
   min_accuracy:float=0.1,
   workers: int=1,
   timeout: int=DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
   # Enable MLflow autologging
   mlflow.sklearn.autolog()
   
   #ingest the data
   df=ingest_data(data_path=data_path)
   # Clean the data and split into training/test sets
   X_train,X_test,Y_train,Y_test=clean_df(df)
   #print("X_train Shape:", X_train.shape)  # Print the shape of X_train
   #print("Y_train Shape:", Y_train.shape)  # Print the shape of Y_train
   model=train_model(X_train=X_train, X_test=X_test, y_train=Y_train, y_test=Y_test)
   accuracy=evaluate_model(model=model, X_test=X_test, Y_test=Y_test)
   deployment_decision=deployment_trigger(accuracy=accuracy)    
   mlflow_model_deployer_step(
      model=model,
      deploy_decision=deployment_decision,
      workers=workers,
      timeout=timeout,
    )

##################################
#        Inference               #
#              Pipeline          #
##################################
    
class MLFlowDeploymentLoaderStepParameters(BaseModel):
   pipeline_name:str
   step_name:str
   running:bool=True


@step(enable_cache=False)
def dynamic_importer()->str:
   data=get_data_for_test()
   return data  

@step(enable_cache=False)
def prediction_service_loader(
   pipeline_name: str,
   pipeline_step_name: str,
   running:bool=True,
   model_name: str="model", 
)->MLFlowDeploymentService:
   mlflow_model_deployer_component=MLFlowModelDeployer.get_active_model_deployer()
   existing_services=mlflow_model_deployer_component.find_model_server(
   pipeline_name=pipeline_name,
   pipeline_step_name=pipeline_step_name,
   model_name=model_name,
   running=running,
)  
   
   if not existing_services:
      raise RuntimeError(
         f"No MLFlow deployment service found for pipeline {pipeline_name},step {pipeline_step_name} and model{model_name} and pipeline for the model {model_name} is currently running"
      )
   print("Existing Services:", existing_services)
   print(type(existing_services))
   return existing_services[0]


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=21)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = ['BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently',
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
       'MaritalStatus_Single', 'Age', 'DailyRate',
       'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
       'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
       'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18',
       'OverTime', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']
    try:
      df = pd.DataFrame(data["data"], columns=columns_for_df)
      json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
      data = np.array(json_list)
      #print("Input Data Shape:", data.shape)
      #print("Input Data Sample:", data[:5])
      prediction = service.predict(data)
      return prediction
    except Exception as e:
        print(f"Prediction error: {str(e)}")

   
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(
    pipeline_name: str,
    pipeline_step_name: str,
    data_path: str = "./data/HR-Employee-Attrition.csv",
):
    """Inference pipeline for making predictions"""
    # Load and preprocess data
    df = ingest_data(data_path=data_path)
    X_train, X_test, Y_train, Y_test = clean_df(df)
    
    # Get the prediction service
    model_deployment_service = MLFlowModelDeployer.get_active_model_deployer()
    model_server = model_deployment_service.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=True,
    )

    if model_server:
        service = cast(MLFlowDeploymentService, model_server[0])
        if service.is_running:
            print(
                f"Model server is running and accepting predictions at:\n"
                f"    {service.prediction_url}\n"
            )
            # You can now use service.predict() to get predictions
            try:
                predictions = service.predict(X_test)
                print(f"Made predictions for {len(predictions)} samples")
                print(f"Sample predictions: {predictions[:5]}")
            except Exception as e:
                print(f"Error making predictions: {e}")
        else:
            print("Model server is not running. Please deploy the model first.")
    else:
        print("No model server found. Please deploy the model first.")