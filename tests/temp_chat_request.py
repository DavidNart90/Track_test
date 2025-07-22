

import requests
import json

url = "http://127.0.0.1:8000/agents/developer/chat"
headers = {"Content-Type": "application/json"}
data = {
    "message": "Hello, developer agent!",
    "session_id": "2aee780c-dc52-4dda-acb3-3fcc82dc13a5"
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    print(response.json())
except requests.exceptions.RequestException as e:
    with open("error.txt", "w") as f:
        f.write(str(e))
except json.JSONDecodeError as e:
    with open("error.txt", "w") as f:
        f.write(f"JSON Decode Error: {e}\nResponse content: {response.text}")
