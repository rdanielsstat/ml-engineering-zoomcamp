import requests

url = 'http://localhost:9696/predict'

patientstay = {
  "health_service_area": "New York City",
  "hospital_county": "New York",
  "operating_certificate_number": "7002032",
  "permanent_facility_id": "001469",
  "facility_name": "MOUNT SINAI MORNINGSIDE",
  "age_group": "70 or Older",
  "zip_code": "100",
  "gender": "F",
  "race": "Other Race",
  "ethnicity": "Spanish/Hispanic",
  "length_of_stay": 3,
  "type_of_admission": "Emergency",
  "patient_disposition": "Home w/ Home Health Services",
  "discharge_year": 2024,
  "ccsr_diagnosis_code": "RSP003",
  "ccsr_diagnosis_description": "INFLUENZA",
  "ccsr_procedure_code": None,
  "ccsr_procedure_description": None,
  "apr_drg_code": "113",
  "apr_drg_description": "INFECTIONS OF UPPER RESPIRATORY TRACT",
  "apr_mdc_code": "03",
  "apr_mdc_description": "EAR, NOSE, MOUTH, THROAT AND CRANIOFACIAL DISEASES",
  "apr_severity_of_illness_code": "3",
  "apr_severity_of_illness_description": "Major",
  "apr_risk_of_mortality": "Major",
  "apr_medical_surgical_description": "Medical",
  "payment_typology_1": "Medicare",
  "payment_typology_2": "Medicare",
  "payment_typology_3": None,
  "birth_weight": None,
  "emergency_department_indicator": "Y",
  "total_charges": 52009.26,
  "total_costs": 11007.46
}

response = requests.post(url, json = patientstay)
result = response.json()
print('Response:', result)
print('Predicted length of stay:', result['predicted_length_of_stay'])