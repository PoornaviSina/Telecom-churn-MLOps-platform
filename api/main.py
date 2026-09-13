import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

from pathlib import Path


MODEL_PATH = Path("models/churn_model.joblib")

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Real-time customer churn prediction API",
    version="1.0.0"
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Telecom Churn Prediction API is running"
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    input_data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    churn_prediction = "Yes" if prediction == 1 else "No"

    return {
        "churn_prediction": churn_prediction,
        "churn_probability": round(float(probability), 4)
    }
