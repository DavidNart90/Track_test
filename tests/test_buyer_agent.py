"""
Unit tests for the Buyer Agent.
"""
import pytest
import pytest
from src.trackrealties.agents.buyer import BuyerAgent
import uuid
from src.trackrealties.agents.base import AgentResponse, AgentDependencies
from unittest.mock import patch, create_autospec
from src.trackrealties.rag.pipeline import RAGPipeline
from src.trackrealties.models.agent import ValidationResult

@pytest.mark.asyncio
async def test_buyer_agent_run():
    """
    Tests the run method of the BuyerAgent.
    """
    # Arrange
    mock_rag_pipeline = create_autospec(RAGPipeline)
    mock_rag_pipeline.generate_response.return_value = (
        "This is a test response from the buyer agent.",
        ValidationResult(
            is_valid=True,
            confidence_score=1.0,
            issues=[],
            validation_type="hallucination",
            validator_version="1.0",
        ),
    )

    agent = BuyerAgent(deps=AgentDependencies(rag_pipeline=mock_rag_pipeline))
    query = "I'm looking for a 3 bedroom house in Texas, TX."
    session_id = str(uuid.uuid4())

    # Act
    response = await agent.run(query, session_id)

    # Assert
    assert isinstance(response, AgentResponse)
    assert response.content == "This is a test response from the buyer agent."
    assert response.validation_result is not None
    mock_rag_pipeline.generate_response.assert_called_once()
