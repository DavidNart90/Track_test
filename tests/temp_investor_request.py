

import requests
import json

# Define the URL and the payload for session creation
session_url = "http://127.0.0.1:8000/session/"
session_payload = {"role": "investor", "user_id": "debug-user"}
headers = {"Content-Type": "application/json"}

try:
    # Create the session
    session_response = requests.post(session_url, data=json.dumps(session_payload), headers=headers)
    session_response.raise_for_status()  # Raise an exception for bad status codes
    session_data = session_response.json()
    session_id = session_data.get("session_id")

    if not session_id:
        print("Error: 'session_id' not found in the response.")
        print("Response:", session_data)
        exit()

    print(f"Successfully created session with ID: {session_id}")

    # Define the URL and payload for the chat request
    chat_url = f"http://127.0.0.1:8000/agents/investor/chat"
    chat_payload = {
        "message": "What is the current market trend for commercial real estate in New York?",
        "session_id": session_id
    }

    # Send the chat message
    chat_response = requests.post(chat_url, data=json.dumps(chat_payload), headers=headers)
    chat_response.raise_for_status()
    chat_data = chat_response.json()

    print("\nChat Response:")
    print(json.dumps(chat_data, indent=2))

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    if e.response:
        print(f"Status Code: {e.response.status_code}")
        try:
            print("Response Body:", e.response.json())
        except json.JSONDecodeError:
            print("Response Body:", e.response.text)


