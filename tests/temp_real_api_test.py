import pytest
import requests
import json
import sys

pytest.skip("Skipping real API test", allow_module_level=True)

def test_agent(role, prompt):
    print(f"Testing {role} agent with prompt: '{prompt}'")
    try:
        session_response = requests.post("http://127.0.0.1:8000/session/", json={"role": role})
        session_response.raise_for_status()
        session_id = session_response.json()["session_id"]

        response = requests.post(
            f"http://127.0.0.1:8000/agents/{role}/chat",
            json={"session_id": session_id, "message": prompt}
        )
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        if e.response:
            print(f"Response content: {e.response.text}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        role = sys.argv[1]
        prompt = " ".join(sys.argv[2:])
        test_agent(role, prompt)
    else:
        print("Running default test for investor agent...")
        test_agent("investor", "Compare the real estate market in Los Angeles County, CA to Dallas, TX.")
