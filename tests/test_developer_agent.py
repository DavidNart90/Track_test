"""
Unit tests for the Developer Agent.
"""
import pytest
import pytest
from src.trackrealties.agents.developer import DeveloperAgent
from src.trackrealties.agents.base import AgentResponse, AgentDependencies
from unittest.mock import patch, create_autospec
from src.trackrealties.rag.pipeline import RAGPipeline

@pytest.mark.asyncio
async def test_developer_agent_run():
    """
    Tests the run method of the DeveloperAgent.
    """
    # Arrange
    mock_rag_pipeline = create_autospec(RAGPipeline)
    mock_rag_pipeline.generate_response.return_value = "This is a test response from the developer agent."

    agent = DeveloperAgent(deps=AgentDependencies(rag_pipeline=mock_rag_pipeline))
    query = "What are the zoning regulations for a property in Texas, TX?"
    session_id = "test_session"

    # Act
    response = await agent.run(query, session_id)

    # Assert
    assert isinstance(response, AgentResponse)
    assert response.content == "This is a test response from the developer agent."
    mock_rag_pipeline.generate_response.assert_called_once()
