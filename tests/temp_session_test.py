import requests
import json

try:
    response = requests.post("http://127.0.0.1:8000/session/", json={"role": "investor"})
    response.raise_for_status()
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    if e.response:
        print(f"Response content: {e.response.text}")
