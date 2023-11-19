import json
import numpy as np
import pandas as pd
import streamlit as st
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main

def main():
    st.title("Employee Attrition Prediction")

    # Define your list of columns
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

    # Create a dictionary to store the selected inputs
    selected_data = {col: 0 for col in columns_for_df}

    # Create a dictionary mapping each categorical feature to its options
    categorical_options = {
    'BusinessTravel': ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'],
    'Department': ['Human Resources', 'Research & Development', 'Sales'],
    'EducationField': ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'],
    'Gender': ['Female', 'Male'],
    'JobRole': ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative'],
    'MaritalStatus': ['Divorced', 'Married', 'Single']
}

    # Create select boxes for categorical features
    for feature, options in categorical_options.items():
        selected_value = st.sidebar.selectbox(f"Select {feature}:", options)
        selected_data[feature + "_" + selected_value] = 1

    # Define numerical features and their corresponding slider ranges
    numerical_features = {
        'Age': (18, 90, 30),
        'DailyRate': (0, 1500, 0),
        'DistanceFromHome': (0, 30, 0),
        'Education': (1, 5, 1),
        'EnvironmentSatisfaction': (1, 4, 1),
        'HourlyRate': (0, 100, 0),
        'JobInvolvement': (1, 4, 1),
        'JobLevel': (1, 5, 1),
        'JobSatisfaction': (1, 4, 1),
        'MonthlyIncome': (0, 100000, 0),
        'MonthlyRate': (0, 30000, 0),
        'NumCompaniesWorked': (0, 10, 0),
        'Over18':(0,1,1),
        'OverTime':(0,1,1),
        'PercentSalaryHike': (0, 1000, 0),
        'PerformanceRating': (1, 4, 1),
        'RelationshipSatisfaction': (1, 4, 1),
        'StockOptionLevel': (1, 4, 1),
        'TotalWorkingYears': (0, 50, 0),
        'TrainingTimesLastYear': (0, 10, 0),
        'WorkLifeBalance': (1, 4, 1),
        'YearsAtCompany': (0, 50, 0),
        'YearsInCurrentRole': (0, 50, 0),
        'YearsSinceLastPromotion': (0, 50, 0),
        'YearsWithCurrManager': (0, 50, 0)
    }

    
    


    # Create sliders for numerical features
    for feature, (min_val, max_val, default_val) in numerical_features.items():
        selected_data[feature] = st.sidebar.slider(feature, min_val, max_val, default_val)

    # Create a DataFrame with the selected inputs
    data_df = pd.DataFrame([selected_data])

    # Display the selected inputs
    st.write("Selected Inputs:")
    st.write(data_df)
    print(data_df)

    if st.button("Predict"):
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            main()
        json_list = json.loads(json.dumps(list(data_df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Predicted Employee Attrition Probability (0 - 1): {:.2f}".format(
                pred[0]
            )
        )

if __name__ == "__main__":
    main()
