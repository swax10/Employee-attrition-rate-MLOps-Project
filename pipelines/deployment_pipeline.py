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
from zenml.steps import BaseParameters, Output
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model

docker_settings=DockerSettings(required_integrations=[MLFLOW])
class DeploymentTriggerConfig(BaseParameters):
  """Class for configuring deployment trigger"""
  min_accuracy: float=0.0


@step 
def deployment_trigger(
  accuracy:float,
  config: DeploymentTriggerConfig,
)->bool:
  return accuracy>config.min_accuracy


@pipeline(enable_cache=False, settings={"docker":docker_settings})
def continuous_deployment_pipeline(
   data_path: str,
   #data_path="C:/Users/user/Desktop/machine learning/Project/zenml Pipeline/Customer_Satisfaction_project/data/olist_customers_dataset.csv",
   min_accuracy:float=0.0,
   workers: int=1,
   timeout: int=DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    
   df=ingest_data()
   # Clean the data and split into training/test sets
   X_train,X_test,Y_train,Y_test=clean_df(df)
   #print("X_train Shape:", X_train.shape)  # Print the shape of X_train
   #print("Y_train Shape:", Y_train.shape)  # Print the shape of Y_train
   model=train_model(X_train,X_test,Y_train,Y_test)
   evaluation_metrics=evaluate_model(model,X_test,Y_test)
   deployment_decision=deployment_trigger(evaluation_metrics)    
   mlflow_model_deployer_step(
      model=model,
      deploy_decision=deployment_decision,
      workers=workers,
      timeout=timeout,
    )
    
class MLFlowDeploymentLoaderStepParameters(BaseParameters):
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


@step
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
       'YearsWithCurrManager',]
    try:
      df = pd.DataFrame(data["data"], columns=columns_for_df)
      json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
      data = np.array(json_list)
      print("Input Data Shape:", data.shape)
      print("Input Data Sample:", data[:5])
      prediction = service.predict(data)
      return prediction
    except Exception as e:
        print(f"Prediction error: {str(e)}")

   
@pipeline(enable_cache=False,settings={"docker":docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name:str):
   data=dynamic_importer()
   service=prediction_service_loader(
      pipeline_name=pipeline_name,
      pipeline_step_name=pipeline_step_name,
      running=False,
      )
   prediction=predictor(service=service,data=data)
   return prediction
