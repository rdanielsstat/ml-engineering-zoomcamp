import time
import requests

url = "http://localhost:9696/predict"

client = {"job": "management", "duration": 400, "poutcome": "success"}
# response = requests.post(url, json=client).json()
# print(response)

while True:
    response = requests.post(url, json=client).json()
    print(response)
    time.sleep(0.1)
