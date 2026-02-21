""" 
We have prepared a pipeline with a dictionary vectorizer and a model.

It was trained (roughly) using this code:

    categorical = ['lead_source']
    numeric = ['number_of_courses_viewed', 'annual_income']

    df[categorical] = df[categorical].fillna('NA')
    df[numeric] = df[numeric].fillna(0)

    train_dict = df[categorical + numeric].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(solver='liblinear')
    )

    pipeline.fit(train_dict, y_train)

Note: You don't need to train the model. This code is just for your reference.

And then saved with Pickle. Download it here.

With wget:

wget https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin
"""

import sys
import pickle
from typing import Dict, Any
from importlib.metadata import version
from fastapi import FastAPI
import uvicorn

print("Python version: " + str(sys.version))
print("Fastapi version: " + str(version("fastapi")))
print("Uvicorn version: " + str(uvicorn.__version__))

app = FastAPI(title = "convert-prediction")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(client):
    result = pipeline.predict_proba(client)[0, 1]
    return float(result)

@app.post("/predict")
def predict(client: Dict[str, Any]):
    prob = predict_single(client)

    return {
        "convert_probability": prob,
        "convert": bool(prob >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host = "127.0.0.1", port = 9696)