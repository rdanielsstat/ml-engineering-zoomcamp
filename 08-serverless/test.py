import requests

url = "http://localhost:8080"

request_data = {
    "url": 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'
}

response = requests.post(url, json=request_data)
print(response.text)  # print raw response