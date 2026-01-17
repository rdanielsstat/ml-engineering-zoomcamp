""" 
Length of Stay Prediction API
Uses a LinearRegression model with target encoding and standardization.
The pipeline includes:
    - Preprocessing (missing value imputation, category mappings)
    - Target encoding for categorical features
    - StandardScaler for feature scaling
    - LinearRegression model predicting log(length_of_stay)
"""
from pydantic import BaseModel
import sys
import pickle
from typing import Dict, Any, Optional
from importlib.metadata import version
from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
from pipeline import LOSPipeline

print("Pydantic version: " + str(version("pydantic")))
print("Python version: " + str(sys.version))
print("fastapi version: " + str(version("fastapi")))
print("uvicorn version: " + str(uvicorn.__version__))

class PatientStay(BaseModel):
    health_service_area: str
    hospital_county: str
    operating_certificate_number: str
    permanent_facility_id: str
    facility_name: str
    age_group: str
    zip_code: str
    gender: str
    race: str
    ethnicity: str
    length_of_stay: int
    type_of_admission: str
    patient_disposition: str
    discharge_year: int
    ccsr_diagnosis_code: str
    ccsr_diagnosis_description: str
    ccsr_procedure_code: Optional[str] = None
    ccsr_procedure_description: Optional[str] = None
    apr_drg_code: str
    apr_drg_description: str
    apr_mdc_code: str
    apr_mdc_description: str
    apr_severity_of_illness_code: str
    apr_severity_of_illness_description: str
    apr_risk_of_mortality: str
    apr_medical_surgical_description: str
    payment_typology_1: str
    payment_typology_2: str
    payment_typology_3: Optional[str] = None
    birth_weight: Optional[float] = None
    emergency_department_indicator: str
    total_charges: float
    total_costs: float

app = FastAPI(title = "length-of-stay-prediction")

# now load the pipeline
with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)


def predict_single(patientstay):
    df = pd.DataFrame([patientstay])
    result = pipeline.predict(df)[0]
    return float(result)

@app.post("/predict")
def predict(patientstay: PatientStay):
    predicted_los = predict_single(patientstay.dict())
    return {"predicted_length_of_stay": predicted_los}

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 9696)