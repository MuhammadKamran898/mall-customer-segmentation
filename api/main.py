from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "customer_segmentation_model.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Mall Customer Segmentation API",
    description="KMeans customer segmentation inference API",
    version="1.0.0",
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------


class CustomerInput(BaseModel):
    annual_income: float = Field(
        ..., ge=0, description="Annual income in thousands of dollars"
    )

    spending_score: float = Field(
        ..., ge=1, le=100, description="Spending score from 1 to 100"
    )


# --------------------------------------------------
# Response schema
# --------------------------------------------------


class CustomerPrediction(BaseModel):
    annual_income: float
    spending_score: float
    cluster: int


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------


@app.get("/")
def root():

    return {"message": "Mall Customer Segmentation API", "status": "running"}


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------


@app.get("/health")
def health():

    return {"status": "healthy", "model_loaded": model is not None}


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------


@app.post("/predict", response_model=CustomerPrediction)
def predict(customer: CustomerInput):

    input_data = pd.DataFrame(
        [
            {
                "Annual Income (k$)": customer.annual_income,
                "Spending Score (1-100)": customer.spending_score,
            }
        ]
    )

    prediction = model.predict(input_data)

    cluster = int(prediction[0])

    return CustomerPrediction(
        annual_income=customer.annual_income,
        spending_score=customer.spending_score,
        cluster=cluster,
    )
