
import requests
import json

url = "http://127.0.0.1:8000/session/"
headers = {"Content-Type": "application/json"}
data = {"role": "developer"}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.json())
except requests.exceptions.RequestException as e:
    with open("error.txt", "w") as f:
        f.write(str(e))
