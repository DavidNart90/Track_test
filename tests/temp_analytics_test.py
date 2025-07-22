import requests
import json

property_id = "333-Florida-St,-San-Antonio,-TX-78210"

try:
    response = requests.get(f"http://127.0.0.1:8000/analytics/cma/{property_id}")
    response.raise_for_status()
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    if e.response:
        print(f"Response content: {e.response.text}")
