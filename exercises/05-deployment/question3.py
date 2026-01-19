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

print("Python version: " + str(sys.version))

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

convert = pipeline.predict_proba(client)[0, 1]

print('Probability of converting: ', convert)

if convert > 0.5:
    print('Prediction: will convert')
else:
    print('Prediction: will not convert')