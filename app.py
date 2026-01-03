
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(title="Employee Promotion Prediction API")

# --------------------------------------------------
# Load Model Artifact
# --------------------------------------------------
artifact = joblib.load("Emp_Promotion_model.pkl")
model = artifact["model"]
THRESHOLD = artifact["threshold"]

# --------------------------------------------------
# Input Schema
# --------------------------------------------------
class EmployeeData(BaseModel):
    department: str
    region: str
    education: str
    gender: str
    recruitment_channel: str
    no_of_trainings: int
    age: int
    previous_year_rating: int
    length_of_service: int
    KPIs_met_80: int
    awards_won: int
    avg_training_score: float

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.get("/")
def read_root():
  return {"Welcome for the Hackathon - Employee Promotion Prediction"}

@app.post("/predict")
def predict_promotion(data: EmployeeData):
    # Convert to DataFrame
    x_input = pd.DataFrame([data.dict()])

    # ------------------------------
    # Feature Engineering
    # ------------------------------
    x_input['training_efficiency'] = (
        x_input['avg_training_score'] / (x_input['length_of_service'] + 1)
    )

    x_input['performance_index'] = (
        x_input['avg_training_score']
        * (x_input['KPIs_met_80'] + 1)
        * (x_input['awards_won'] + 1)
    )

    x_input['kpi_per_training'] = (
        x_input['KPIs_met_80'] / (x_input['no_of_trainings'] + 1)
    )

    x_input['awards_per_service'] = (
        x_input['awards_won'] / (x_input['length_of_service'] + 1)
    )

    x_input['score_times_awards'] = (
        x_input['avg_training_score'] * (x_input['awards_won'] + 1)
    )

    x_input['score_times_kpi'] = (
        x_input['avg_training_score'] * (x_input['KPIs_met_80'] + 1)
    )

    # ------------------------------
    # Prediction
    # ------------------------------
    prob = model.predict_proba(x_input)[0, 1]
    prediction = int(prob >= THRESHOLD)

    return {
        "promotion_probability": float(prob),
        "threshold_used": float(THRESHOLD),
        "prediction": prediction,
        "promotion_status": "PROMOTED" if prediction == 1 else "NOT PROMOTED"
    }
