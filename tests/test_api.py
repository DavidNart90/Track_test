import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from src.trackrealties.api.main import app
from src.trackrealties.agents.base import AgentResponse

@pytest.mark.anyio
@pytest.mark.parametrize("role", ["investor", "developer", "buyer", "agent"])
async def test_agent_chat_endpoint(role: str):
    """
    Tests the /agents/{role}/chat endpoint for all roles.
    """
    with patch('src.trackrealties.api.main.lifespan', new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # Mock the agent's run method
            with patch('src.trackrealties.agents.base.BaseAgent.run', new_callable=AsyncMock) as mock_agent_run:
                mock_agent_run.return_value = AgentResponse(content=f"Response from {role} agent")

                # Create a session first
                session_response = await ac.post("/session/", json={"role": role})
                assert session_response.status_code == 200
                session_id = session_response.json()["session_id"]

                request_data = {
                    "message": "Hello",
                    "session_id": session_id
                }

                # Act
                response = await ac.post(f"/agents/{role}/chat", json=request_data)

                # Assert
                assert response.status_code == 200
                json_response = response.json()
                assert json_response["message"] == f"Response from {role} agent"
                assert "session_id" in json_response
                assert "assistant_message_id" in json_response

