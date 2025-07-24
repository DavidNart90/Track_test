import asyncio
import httpx
import json

async def test_greeting_handling():
    """Test that greetings are properly handled by the LLM"""
    
    test_cases = [
        ("Hi agent, how are you?", "investor"),
        ("Hello there!", "buyer"),
        ("Good morning", "developer"),
        ("What's up?", "agent"),
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for greeting, role in test_cases:
            print(f"\n{'='*50}")
            print(f"Testing: '{greeting}' as {role}")
            print('='*50)
            
            # Create session
            session_resp = await client.post(
                "http://localhost:8000/session/",
                json={"role": role}
            )
            session_data = session_resp.json()
            session_id = session_data["session_id"]
            
            # Send greeting
            chat_resp = await client.post(
                f"http://localhost:8000/agents/{role}/chat",
                json={
                    "session_id": session_id,
                    "message": greeting
                }
            )
            
            if chat_resp.status_code == 200:
                response_data = chat_resp.json()
                print(f"✅ Success!")
                print(f"Response: {response_data['message']}")
                print(f"Tools used: {response_data.get('tools_used', [])}")
            else:
                print(f"❌ Error: {chat_resp.status_code}")
                print(chat_resp.text)

if __name__ == "__main__":
    asyncio.run(test_greeting_handling())