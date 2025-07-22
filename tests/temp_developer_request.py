import asyncio
import httpx

async def main():
    """Sends a request to the developer agent and prints the response."""
    async with httpx.AsyncClient() as client:
        try:
            # Create a session first
            session_response = await client.post("http://127.0.0.1:8000/session/", json={"role": "developer"})
            session_response.raise_for_status()
            session_id = session_response.json()["session_id"]
            print(f"Successfully created session: {session_id}")

            # Send a message to the agent
            request_data = {
                "message": "What is the median price and inventory count in Dallas, TX?",
                "session_id": session_id
            }
            response = await client.post("http://127.0.0.1:8000/agents/developer/chat", json=request_data)
            response.raise_for_status()

            print("Successfully received response from the developer agent:")
            print(response.json())

        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            print("Please ensure the API server is running.")
        except httpx.HTTPStatusError as e:
            print(f"Error response {e.response.status_code} while requesting {e.request.url!r}.")
            print(f"Response content: {e.response.text}")

if __name__ == "__main__":
    asyncio.run(main())