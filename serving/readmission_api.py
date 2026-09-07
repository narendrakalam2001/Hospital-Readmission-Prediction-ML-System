# ============================================================
# HOSPITAL READMISSION RISK API — FastAPI Serving
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import logging
import time
import os
import json

from src.model_loader             import load_latest_model
from services.prediction_service  import predict_patient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hospital Readmission Risk Prediction API",
    description="Predicts 30-day hospital readmission risk for diabetic inpatients "
                 "(Diabetes 130-US Hospitals dataset, HRRP-aligned scoring).",
    version="1.0.0",
)

# ── Load model on startup ─────────────────────────────────────
try:
    model, threshold = load_latest_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Model loading failed: %s", e)
    model     = None
    threshold = 0.5


# ============================================================
# INPUT SCHEMA
# ============================================================

class PatientInput(BaseModel):
    race:                      str = "Caucasian"
    gender:                    str = "Female"
    age:                       str = "[70-80)"          # bracketed age group as in source data
    admission_type_id:         int = 1                  # 1=Emergency, 2=Urgent, 3=Elective ...
    discharge_disposition_id:  int = 1                  # 1=Discharged to home ...
    admission_source_id:       int = 7                  # 7=Emergency Room ...
    time_in_hospital:          int = Field(4, ge=1, le=14)
    medical_specialty:         str = "Missing"
    num_lab_procedures:        int = 45
    num_procedures:            int = 1
    num_medications:           int = 15
    number_outpatient:         int = 0
    number_emergency:          int = 0
    number_inpatient:          int = 0
    diag_1:                    str = "250.83"           # ICD-9 code
    diag_2:                    str = "401"
    diag_3:                    str = "272"
    number_diagnoses:          int = 7
    max_glu_serum:             str = "None"             # None / Norm / >200 / >300
    A1Cresult:                 str = "None"             # None / Norm / >7 / >8
    metformin:                 str = "No"
    repaglinide:                str = "No"
    nateglinide:                str = "No"
    chlorpropamide:             str = "No"
    glimepiride:                 str = "No"
    acetohexamide:               str = "No"
    glipizide:                   str = "No"
    glyburide:                   str = "No"
    tolbutamide:                 str = "No"
    pioglitazone:                str = "No"
    rosiglitazone:                str = "No"
    acarbose:                     str = "No"
    miglitol:                     str = "No"
    troglitazone:                  str = "No"
    tolazamide:                    str = "No"
    insulin:                       str = "Steady"
    glyburide_metformin:           str = Field("No", alias="glyburide-metformin")
    glipizide_metformin:           str = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone:      str = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone:       str = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone:        str = Field("No", alias="metformin-pioglitazone")
    change:                        str = "No"           # Ch / No
    diabetesMed:                   str = "Yes"           # Yes / No

    class Config:
        populate_by_name = True


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Hospital Readmission Risk Prediction API is live",
        "docs":    "/docs",
        "health":  "/health"
    }

@app.get("/health")
def health():
    return {"status": "running", "model_loaded": model is not None}

@app.get("/model_info")
def model_info():
    registry_path = "readmission_models/latest_model.json"
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            return json.load(f)
    return {"error": "Model registry not found"}


# ── Prediction endpoint ───────────────────────────────────────

@app.post("/predict")
def predict(patient: PatientInput):

    start      = time.time()
    input_data = patient.dict(by_alias=True)

    result = predict_patient(model, input_data, threshold)

    # ── Log prediction ────────────────────────────────────────
    log_record = {
        "timestamp":                 time.time(),
        "age":                       input_data["age"],
        "time_in_hospital":          input_data["time_in_hospital"],
        "number_inpatient":          input_data["number_inpatient"],
        "readmission_probability":   result["readmission_probability"],
        "risk_band":                 result["risk_band"],
        "decision":                  result["decision"],
        "rule_triggered":            result.get("rule_triggered"),
    }

    log_path = "logs/prediction_logs.csv"
    os.makedirs("logs", exist_ok=True)

    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    result["latency_seconds"] = round(time.time() - start, 4)

    return result
