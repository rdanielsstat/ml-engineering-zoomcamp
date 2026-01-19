# Length of Stay Prediction for Flu Hospitalizations in New York

This project predicts the **length of hospital stay (LOS)** for patients hospitalized with influenza in New York using publicly available SPARCS data (2024).

A short walkthrough video is available in the `midterm_project/` folder.

The model pipeline includes:

- Data preprocessing and cleaning  
- Target encoding for categorical features  
- Feature standardization  
- Predictive modeling with multiple models (OLS, Random Forest, CatBoost)  
- Linear Regression used for the deployed API  
- Model deployment via FastAPI and Docker  

The dataset is from [NY State Health Data: Hospital Inpatient Discharges (SPARCS De-Identified) 2024](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/sf4k-39ay/about_data).

The model is deployed via FastAPI and Docker at [https://los-prediction.fly.dev/docs](https://los-prediction.fly.dev/docs)

## Quick Test

You can manually try predictions via the Swagger docs at the above URL, or programmatically using the script in `app/test-fly.io-deployment.py`.

Project Structure
- app/ – Deployment files, FastAPI service, model pipeline, and test script
- data/ – Raw and processed datasets
- models/ – Saved models
- python/ – Jupyter notebooks for data preparation, EDA, feature engineering, model training, and evaluation

Note

The deployed API uses the pipeline_v1.bin model pipeline for predictions. Full exploration, feature importance, and model tuning are in the notebooks.