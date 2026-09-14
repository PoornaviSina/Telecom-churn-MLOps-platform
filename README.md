# Telecom Churn MLOps Platform

A real-time AI prediction platform for predicting customer churn using Machine Learning and MLOps practices.

This project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, experiment tracking, real-time prediction, automated retraining, CI/CD, data drift monitoring, and dashboard-based monitoring.

---

## 1. Project Overview

Customer churn occurs when customers stop using a company's services.

Telecommunication companies can use machine learning to identify customers who are likely to leave. This can help companies take early actions to improve customer retention.

This project develops a complete MLOps platform for predicting whether a telecom customer is likely to churn.

The platform includes:

* Data preprocessing
* Machine learning model training
* Model evaluation
* Experiment tracking using MLflow
* Real-time predictions using FastAPI
* Automated model retraining
* Continuous Integration using GitHub Actions
* Data drift monitoring
* Monitoring dashboard using Streamlit
* Docker support

---

## 2. Project Objectives

The main objectives of this project are:

1. Build a machine learning model to predict customer churn.
2. Create a real-time prediction API.
3. Track machine learning experiments using MLflow.
4. Automate model retraining when relevant data or code changes occur.
5. Implement CI/CD using GitHub Actions.
6. Monitor changes in input data distributions.
7. Create a dashboard for visualizing data drift.
8. Dockerize the prediction API.
9. Demonstrate important MLOps practices throughout the machine learning lifecycle.

---

## 3. Dataset

The project uses the **Telco Customer Churn Dataset**.

The dataset contains information about telecom customers and their services.

### Main Features

* Gender
* Senior citizen status
* Partner status
* Dependents
* Tenure
* Phone service
* Multiple lines
* Internet service
* Online security
* Online backup
* Device protection
* Technical support
* Streaming TV
* Streaming movies
* Contract type
* Paperless billing
* Payment method
* Monthly charges
* Total charges
* Churn

### Dataset Details

* Original records: **7,043**
* Original columns: **21**
* Processed records: **7,043**
* Processed columns: **20**
* Target variable: `Churn`

The target variable is converted into binary values:

```text
No  → 0
Yes → 1
```

---

## 4. Technologies Used

### Programming Language

* Python 3.12

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* One-Hot Encoding
* StandardScaler
* Train-Test Split

### Experiment Tracking

* MLflow

### API Development

* FastAPI
* Uvicorn
* Pydantic

### Monitoring

* SciPy
* Kolmogorov-Smirnov Test
* Streamlit

### Testing

* Pytest

### DevOps and Deployment

* Docker
* GitHub Actions
* Git
* GitHub

---

## 5. Project Structure

```text
Telecom-churn-MLOps-platform/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── retraining.yml
│
├── api/
│   └── main.py
│
├── data/
│   ├── current/
│   │   └── .gitkeep
│   │
│   ├── processed/
│   │   ├── .gitkeep
│   │   └── cleaned_telco_churn.csv
│   │
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   │
│   └── reference/
│       └── .gitkeep
│
├── mlruns/
│   └── MLflow experiment files
│
├── models/
│   ├── .gitkeep
│   └── churn_model.joblib
│
├── monitoring/
│   ├── .gitkeep
│   ├── dashboard.py
│   └── drift_monitor.py
│
├── src/
│   ├── preprocess.py
│   └── train.py
│
├── tests/
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── mlflow.db
├── README.md
└── requirements.txt
```

---

## 6. Data Preprocessing

The preprocessing script prepares the raw dataset for machine learning.

### Preprocessing Steps

1. Load the raw dataset.
2. Remove unnecessary spaces from column names.
3. Convert `TotalCharges` into numerical format.
4. Detect missing values in `TotalCharges`.
5. Replace missing `TotalCharges` values with the median.
6. Convert the `Churn` column into binary values.
7. Remove the `customerID` column.
8. Save the cleaned dataset.

### Processed Dataset

The cleaned dataset is saved at:

```text
data/processed/cleaned_telco_churn.csv
```

### Run Preprocessing

```bash
python src/preprocess.py
```

---

## 7. Machine Learning Model

The project uses **Logistic Regression** for customer churn prediction.

Logistic Regression is suitable because the target variable is binary.

```text
0 = Customer does not churn
1 = Customer churns
```

### Feature Preprocessing

Numerical features are standardized using:

```text
StandardScaler
```

Categorical features are converted into numerical values using:

```text
OneHotEncoder
```

The model uses:

```text
handle_unknown="ignore"
```

to handle previously unseen categorical values.

### Model Configuration

```text
Model                : Logistic Regression
Maximum Iterations   : 1000
Class Weight         : Balanced
Random State         : 42
Test Size            : 20%
```

The trained model is saved as:

```text
models/churn_model.joblib
```

---

## 8. Model Training and Evaluation

The processed dataset is divided into:

* **80% training data**
* **20% testing data**

The following metrics are used to evaluate the model:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Evaluation Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.7381 |
| Precision | 0.5043 |
| Recall    | 0.7834 |
| F1 Score  | 0.6136 |
| ROC-AUC   | 0.8413 |

The model achieved a ROC-AUC score of **0.8413**.

The relatively high recall indicates that the model is able to identify a large proportion of customers who are likely to churn.

### Train the Model

```bash
python src/train.py
```

---

## 9. Experiment Tracking with MLflow

MLflow is used to track machine learning experiments.

The project records model parameters, evaluation metrics, and the trained model.

### Parameters Logged

* Model type
* Test size
* Random state
* Class weight

### Metrics Logged

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Experiment Name

```text
Telco-Churn-Prediction
```

### Start MLflow

Run:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

MLflow allows the experiment results to be viewed through a web-based interface.

---

## 10. Real-Time Prediction API

FastAPI is used to provide real-time customer churn predictions.

The API loads the trained machine learning model and accepts customer information as input.

The API returns:

* Churn prediction
* Churn probability

### Start the API

Run:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### API Documentation

FastAPI provides interactive Swagger documentation at:

```text
http://127.0.0.1:8000/docs
```

### API Endpoints

#### GET `/`

Checks whether the API is running.

#### POST `/predict`

Accepts customer information and returns a churn prediction.

### Example Response

```json
{
    "churn_prediction": "Yes",
    "churn_probability": 0.7824
}
```

The prediction API has been tested successfully using the FastAPI Swagger interface.

---

## 11. Automated Model Retraining

The project includes an automated model retraining workflow using GitHub Actions.

The workflow can be triggered when changes occur in:

```text
data/raw/**
src/preprocess.py
src/train.py
```

It can also be started manually using the GitHub Actions interface.

### Retraining Process

The workflow performs the following steps:

1. Checkout the repository.
2. Set up Python 3.12.
3. Install project dependencies.
4. Run data preprocessing.
5. Train the machine learning model.
6. Evaluate the trained model.
7. Verify that the model file was created.

The workflow is located at:

```text
.github/workflows/retraining.yml
```

### Workflow Status

The automated retraining workflow has been successfully tested through GitHub Actions.

---

## 12. Continuous Integration

GitHub Actions is used to implement Continuous Integration.

The CI workflow is triggered when:

* Code is pushed to the `main` branch.
* A pull request is created for the `main` branch.

### CI Pipeline

The pipeline performs:

1. Checkout repository.
2. Set up Python 3.12.
3. Install dependencies.
4. Run automated tests.
5. Build the Docker image.

The workflow is located at:

```text
.github/workflows/ci.yml
```

---

## 13. Data Drift Monitoring

Data drift occurs when the statistical distribution of input data changes over time.

For example, customer tenure or monthly charges may have a different distribution in new customer data compared with the data used for model development.

This project uses the **Kolmogorov-Smirnov (KS) test** to monitor numerical features.

### Monitored Features

* `SeniorCitizen`
* `tenure`
* `MonthlyCharges`
* `TotalCharges`

### Drift Threshold

The project uses:

```text
p-value < 0.05
```

as the drift detection threshold.

If:

```text
p-value < 0.05
```

the feature is considered to have significant drift.

Otherwise:

```text
No significant drift
```

### Run Drift Monitoring

```bash
python monitoring/drift_monitor.py
```

The monitoring script generates a JSON report:

```text
monitoring/drift_report.json
```

### Example Drift Test

For testing purposes, the current dataset was simulated by changing the distributions of:

* `tenure`
* `MonthlyCharges`

The resulting test showed:

| Feature        | Status               |
| -------------- | -------------------- |
| SeniorCitizen  | No significant drift |
| tenure         | Drift detected       |
| MonthlyCharges | Drift detected       |
| TotalCharges   | No significant drift |

This simulated dataset is used only to demonstrate that the monitoring system can detect data drift.

---

## 14. Monitoring Dashboard

A Streamlit dashboard is included to visualize the data drift results.

The dashboard displays:

* Overall drift status
* Feature name
* KS statistic
* P-value
* Drift status

### Start the Dashboard

Run:

```bash
streamlit run monitoring/dashboard.py
```

The dashboard opens in the browser.

### Dashboard Status

The dashboard displays either:

```text
NO SIGNIFICANT DRIFT
```

or:

```text
DRIFT DETECTED
```

depending on the monitoring results.

---

## 15. Docker Support

The project includes a Dockerfile for containerizing the FastAPI prediction service.

The Docker image contains:

* Python 3.12
* Project dependencies
* FastAPI application
* Trained machine learning model

### Dockerfile

The Dockerfile:

1. Uses the Python 3.12 slim image.
2. Sets the working directory.
3. Installs project dependencies.
4. Copies the API application.
5. Copies the trained model.
6. Exposes port 8000.
7. Starts the FastAPI application using Uvicorn.

### Build the Docker Image

```bash
docker build -t telecom-churn-api .
```

### Run the Docker Container

```bash
docker run -p 8000:8000 telecom-churn-api
```

Then open:

```text
http://localhost:8000/docs
```

> **Note:** Docker containerization is included in the project, but local Docker execution depends on Docker Desktop being available and running.

---

## 16. How to Run the Project Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/PoornaviSina/Telecom-churn-MLOps-platform.git
```

### Step 2: Open the Project

```bash
cd Telecom-churn-MLOps-platform
```

### Step 3: Create a Virtual Environment

```bash
py -3.12 -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Preprocess the Dataset

```bash
python src/preprocess.py
```

### Step 7: Train the Model

```bash
python src/train.py
```

### Step 8: Start the FastAPI Service

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Step 9: Start MLflow

Open another terminal:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

### Step 10: Run Drift Monitoring

```bash
python monitoring/drift_monitor.py
```

### Step 11: Start the Monitoring Dashboard

```bash
streamlit run monitoring/dashboard.py
```

---

## 17. MLOps Workflow

The overall workflow of the project is:

```text
                Raw Dataset
                     |
                     v
             Data Preprocessing
                     |
                     v
              Cleaned Dataset
                     |
                     v
              Model Training
                     |
                     v
             Model Evaluation
                     |
                     v
           MLflow Experiment Tracking
                     |
                     v
             Trained Model
                     |
                     v
            FastAPI Prediction API
                     |
                     v
          Real-Time Churn Prediction
                     |
                     v
            Data Drift Monitoring
                     |
                     v
          Automated Model Retraining
                     |
                     v
              CI/CD Pipeline
```

---

## 18. MLOps Components

| Component            | Technology     | Purpose                          |
| -------------------- | -------------- | -------------------------------- |
| Data Processing      | Pandas         | Clean and prepare data           |
| Model Training       | Scikit-learn   | Train churn prediction model     |
| Experiment Tracking  | MLflow         | Track experiments and metrics    |
| Model Storage        | Joblib         | Save trained model               |
| Prediction API       | FastAPI        | Provide real-time predictions    |
| API Server           | Uvicorn        | Run FastAPI application          |
| Automated Retraining | GitHub Actions | Retrain model automatically      |
| CI                   | GitHub Actions | Test project changes             |
| Drift Detection      | SciPy KS Test  | Detect data distribution changes |
| Monitoring Dashboard | Streamlit      | Visualize drift results          |
| Containerization     | Docker         | Package prediction service       |

---

## 19. Testing

Pytest is used for automated testing.

Tests are located in:

```text
tests/test_api.py
```

The CI pipeline automatically runs the tests using:

```bash
pytest
```

This helps ensure that changes to the project do not break the application.

---

## 20. GitHub Actions

The project contains two GitHub Actions workflows.

### CI Workflow

```text
.github/workflows/ci.yml
```

Responsible for:

* Installing dependencies
* Running tests
* Building the Docker image

### Retraining Workflow

```text
.github/workflows/retraining.yml
```

Responsible for:

* Preprocessing data
* Training the model
* Evaluating the model
* Verifying the trained model

Both workflows are stored inside the `.github/workflows` directory.

---

## 21. Current Project Status

### Completed

* [x] GitHub repository created
* [x] Project folder structure created
* [x] Raw dataset added
* [x] Data preprocessing implemented
* [x] Cleaned dataset generated
* [x] Machine learning model trained
* [x] Model evaluation completed
* [x] MLflow experiment tracking implemented
* [x] Trained model saved using Joblib
* [x] FastAPI prediction API implemented
* [x] Real-time prediction tested
* [x] Automated retraining workflow implemented
* [x] GitHub Actions CI workflow implemented
* [x] Data drift monitoring implemented
* [x] Drift detection tested
* [x] Streamlit monitoring dashboard implemented
* [x] Dockerfile created
* [x] Project documentation created

### Remaining / Deployment

* [ ] Complete cloud deployment of the prediction API
* [ ] Deploy monitoring dashboard online
* [ ] Configure persistent cloud-based MLflow tracking
* [ ] Connect production data to automated drift monitoring
* [ ] Deploy automatically retrained models

---

## 22. Future Improvements

The following improvements can be added to make the platform more production-ready:

* Deploy the FastAPI service to a cloud platform.
* Deploy the Streamlit monitoring dashboard online.
* Use a persistent cloud MLflow tracking server.
* Add a model registry.
* Automatically promote better-performing models.
* Add model performance monitoring.
* Add automated alerts when drift is detected.
* Store monitoring results in a database.
* Add authentication and authorization to the API.
* Compare multiple machine learning algorithms.
* Add automated deployment of newly retrained models.
* Improve the monitoring dashboard with historical charts.
* Add production customer data pipelines.

---

## 23. Conclusion

This project demonstrates an end-to-end MLOps workflow for a telecom customer churn prediction system.

A machine learning model was developed using Logistic Regression and connected to a real-time FastAPI prediction service. MLflow was used for experiment tracking, GitHub Actions was used for CI and automated retraining, and the Kolmogorov-Smirnov test was used to monitor data drift.

A Streamlit dashboard provides a visual representation of drift detection results, while Docker support provides a way to package the prediction API for deployment.

The project demonstrates how machine learning can be combined with software engineering, automation, monitoring, and deployment practices to create a more reliable and maintainable AI system.
git status
