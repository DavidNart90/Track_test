import requests
import json

session_id = "b51555cf-8f68-49d9-b017-ff1e7e140db3"  # From the previous test
prompt = "What is the median price per square foot in Dallas, TX?"

try:
    response = requests.post(
        "http://127.0.0.1:8000/chat/",
        json={"session_id": session_id, "message": prompt}
    )
    response.raise_for_status()
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    if e.response:
        print(f"Response content: {e.response.text}")
