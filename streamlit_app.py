import streamlit as st
import pandas as pd
import numpy as np
import json
import mlflow
from mlflow.tracking import MlflowClient

def get_production_model():
    """Get the production model from MLflow"""
    client = MlflowClient()
    try:
        production_model = client.get_latest_versions("employee_attrition_model", stages=["Production"])[0]
        model = mlflow.sklearn.load_model(production_model.source)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure the model is deployed in Production stage.")
        return None

def create_prediction_data(
    age, business_travel, daily_rate, department, distance_from_home,
    education, education_field, environment_satisfaction, gender,
    hourly_rate, job_involvement, job_level, job_role, job_satisfaction,
    marital_status, monthly_income, monthly_rate, num_companies_worked,
    over_time, percent_salary_hike, performance_rating,
    relationship_satisfaction, stock_option_level, total_working_years,
    training_times_last_year, work_life_balance, years_at_company,
    years_in_current_role, years_since_last_promotion, years_with_curr_manager
):
    """Create a DataFrame with the input data in the correct format"""
    
    # Initialize all columns with 0
    data = {col: [0] for col in [
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
        'MaritalStatus_Single'
    ]}
    
    # Set categorical variables
    data[f'BusinessTravel_{business_travel}'] = [1]
    data[f'Department_{department}'] = [1]
    data[f'EducationField_{education_field}'] = [1]
    data[f'Gender_{gender}'] = [1]
    data[f'JobRole_{job_role}'] = [1]
    data[f'MaritalStatus_{marital_status}'] = [1]
    
    # Add numerical variables
    numerical_data = {
        'Age': [age],
        'DailyRate': [daily_rate],
        'DistanceFromHome': [distance_from_home],
        'Education': [education],
        'EnvironmentSatisfaction': [environment_satisfaction],
        'HourlyRate': [hourly_rate],
        'JobInvolvement': [job_involvement],
        'JobLevel': [job_level],
        'JobSatisfaction': [job_satisfaction],
        'MonthlyIncome': [monthly_income],
        'MonthlyRate': [monthly_rate],
        'NumCompaniesWorked': [num_companies_worked],
        'Over18': [1],  # Always 'Y' in the dataset
        'OverTime': [1 if over_time else 0],
        'PercentSalaryHike': [percent_salary_hike],
        'PerformanceRating': [performance_rating],
        'RelationshipSatisfaction': [relationship_satisfaction],
        'StockOptionLevel': [stock_option_level],
        'TotalWorkingYears': [total_working_years],
        'TrainingTimesLastYear': [training_times_last_year],
        'WorkLifeBalance': [work_life_balance],
        'YearsAtCompany': [years_at_company],
        'YearsInCurrentRole': [years_in_current_role],
        'YearsSinceLastPromotion': [years_since_last_promotion],
        'YearsWithCurrManager': [years_with_curr_manager]
    }
    
    data.update(numerical_data)
    return pd.DataFrame(data)

def main():
    st.title("Employee Attrition Prediction")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Load the model
    model = get_production_model()
    if model is None:
        return
    
    # Create form for user input
    with st.form("prediction_form"):
        st.subheader("Employee Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            business_travel = st.selectbox(
                "Business Travel",
                ["Non-Travel", "Travel_Frequently", "Travel_Rarely"]
            )
            daily_rate = st.number_input("Daily Rate", min_value=0, value=1000)
            department = st.selectbox(
                "Department",
                ["Human Resources", "Research & Development", "Sales"]
            )
            
        with col2:
            distance_from_home = st.number_input("Distance From Home", min_value=0, value=10)
            education = st.slider("Education Level", 1, 5, 3)
            education_field = st.selectbox(
                "Education Field",
                ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]
            )
            environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 2)
            
        with col3:
            gender = st.selectbox("Gender", ["Female", "Male"])
            hourly_rate = st.number_input("Hourly Rate", min_value=0, value=50)
            job_involvement = st.slider("Job Involvement", 1, 4, 2)
            job_level = st.slider("Job Level", 1, 5, 2)
            
        st.subheader("Job Details")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            job_role = st.selectbox(
                "Job Role",
                ["Healthcare Representative", "Human Resources", "Laboratory Technician",
                 "Manager", "Manufacturing Director", "Research Director",
                 "Research Scientist", "Sales Executive", "Sales Representative"]
            )
            job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)
            marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
            
        with col5:
            monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
            monthly_rate = st.number_input("Monthly Rate", min_value=0, value=20000)
            num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, value=2)
            over_time = st.checkbox("Works Overtime")
            
        with col6:
            percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, value=15)
            performance_rating = st.slider("Performance Rating", 1, 4, 3)
            relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 2)
            
        st.subheader("Additional Information")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
            total_working_years = st.number_input("Total Working Years", min_value=0, value=5)
            training_times_last_year = st.number_input("Training Times Last Year", min_value=0, value=2)
            
        with col8:
            work_life_balance = st.slider("Work Life Balance", 1, 4, 2)
            years_at_company = st.number_input("Years at Company", min_value=0, value=3)
            years_in_current_role = st.number_input("Years in Current Role", min_value=0, value=2)
            
        with col9:
            years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, value=1)
            years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, value=2)
        
        submitted = st.form_submit_button("Predict Attrition")
        
        if submitted:
            # Create prediction data
            input_data = create_prediction_data(
                age, business_travel, daily_rate, department, distance_from_home,
                education, education_field, environment_satisfaction, gender,
                hourly_rate, job_involvement, job_level, job_role, job_satisfaction,
                marital_status, monthly_income, monthly_rate, num_companies_worked,
                over_time, percent_salary_hike, performance_rating,
                relationship_satisfaction, stock_option_level, total_working_years,
                training_times_last_year, work_life_balance, years_at_company,
                years_in_current_role, years_since_last_promotion, years_with_curr_manager
            )
            
            try:
                # Make prediction
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0]
                
                # Display result
                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    st.error(f"⚠️ High Risk of Attrition (Probability: {probability[1]:.2%})")
                else:
                    st.success(f"✅ Low Risk of Attrition (Probability: {probability[0]:.2%})")
                
                # Display feature importance if available
                if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                    st.subheader("Key Factors")
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    else:
                        importance = abs(model.coef_[0])
                    
                    feature_importance = pd.DataFrame({
                        'feature': input_data.columns,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                    
                    st.bar_chart(feature_importance.head(10).set_index('feature'))
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
