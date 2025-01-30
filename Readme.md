# Employee Attrition Prediction with MLOps

## Problem Statement
Predict whether an employee working in a company will leave based on several factors like income, age, performance, personal details, etc.

## Dataset
IBM HR Analytics Employee Attrition & Performance Dataset
[Source](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## Solution Overview
This project implements an end-to-end MLOps pipeline for employee attrition prediction using:
- [ZenML](https://zenml.io/) for pipeline orchestration
- [MLflow](https://mlflow.org/) for experiment tracking and model deployment
- [Streamlit](https://streamlit.io/) for the web interface

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Employee-attrition-rate-MLOps-Project.git
cd Employee-attrition-rate-MLOps-Project
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. ZenML Setup
```bash
# Install ZenML with server components
pip install "zenml[server]"

# Initialize ZenML
zenml init

# Start ZenML server (dashboard)
zenml up

# Install MLflow integration
zenml integration install mlflow -y

# Configure ZenML stack with MLflow
zenml experiment-tracker register mlflow_tracker_employee --flavor=mlflow
zenml model-deployer register mlflow_employee --flavor=mlflow
zenml stack register mlflow_stack_employee -a default -o default -d mlflow_employee -e mlflow_tracker_employee --set
```

## 🚀 Running the Project

### 1. Start MLflow Server
```bash
# Start MLflow server with SQLite backend
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### 2. Run the Training Pipeline
```bash
python run_pipeline.py
```

### 3. Run the Deployment Pipeline
```bash
python run_deployment.py
```

### 4. Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```

## 📊 Project Structure

```
Employee-attrition-rate-MLOps-Project/
├── data/                      # Dataset directory
├── mlruns/                    # MLflow artifacts
├── pipelines/                 # ZenML pipeline definitions
│   ├── training_pipeline.py
│   └── deployment_pipeline.py
├── steps/                     # Pipeline steps
├── streamlit_app.py          # Streamlit web interface
├── run_pipeline.py           # Training pipeline executor
├── run_deployment.py         # Deployment pipeline executor
└── requirements.txt          # Project dependencies
```

## 🔄 MLflow Components

### Experiment Tracking
- All model training runs are automatically logged using MLflow
- Tracked metrics include:
  - Model accuracy
  - Feature importance
  - Model parameters
  - Input feature distributions

### Model Registry
- Models are registered with MLflow Model Registry
- Production models are automatically transitioned through stages:
  - None → Staging → Production
- Model versions are maintained for rollback capability

### Model Serving
- Production models are served via MLflow's model serving capability
- The Streamlit app automatically fetches the latest production model
- Predictions are made in real-time through the web interface

## 📊 Web Interface Features
- Input form for employee details
- Real-time predictions
- Feature importance visualization
- Prediction confidence scores

## 🔍 Monitoring & Maintenance

### View MLflow Dashboard
```bash
# Access MLflow UI at http://localhost:5000
```

### View ZenML Dashboard
```bash
# Access ZenML UI at http://localhost:8237
```

### Check Model Status
```bash
# List registered models
mlflow models list

# Get model versions
mlflow models versions employee_attrition_model

# Get model stage transitions
mlflow models transitions employee_attrition_model
```

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
