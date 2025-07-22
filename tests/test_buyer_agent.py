"""
Unit tests for the Buyer Agent.
"""
import pytest
import pytest
from src.trackrealties.agents.buyer import BuyerAgent
from src.trackrealties.agents.base import AgentResponse, AgentDependencies
from unittest.mock import patch, create_autospec
from src.trackrealties.rag.pipeline import RAGPipeline

@pytest.mark.asyncio
async def test_buyer_agent_run():
    """
    Tests the run method of the BuyerAgent.
    """
    # Arrange
    mock_rag_pipeline = create_autospec(RAGPipeline)
    mock_rag_pipeline.generate_response.return_value = "This is a test response from the buyer agent."

    agent = BuyerAgent(deps=AgentDependencies(rag_pipeline=mock_rag_pipeline))
    query = "I'm looking for a 3 bedroom house in Texas, TX."
    session_id = "test_session"

    # Act
    response = await agent.run(query, session_id)

    # Assert
    assert isinstance(response, AgentResponse)
    assert response.content == "This is a test response from the buyer agent."
    mock_rag_pipeline.generate_response.assert_called_once()
