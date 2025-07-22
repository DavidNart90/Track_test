"""
Unit tests for the Investor Agent.
"""
import pytest
import pytest
from src.trackrealties.agents.investor import InvestorAgent
from src.trackrealties.agents.base import AgentResponse, AgentDependencies
from unittest.mock import patch, create_autospec
from src.trackrealties.rag.pipeline import RAGPipeline
from src.trackrealties.models.agent import ValidationResult

@pytest.mark.asyncio
async def test_investor_agent_run():
    """
    Tests the run method of the InvestorAgent.
    """
    # Arrange
    mock_rag_pipeline = create_autospec(RAGPipeline)
    mock_rag_pipeline.generate_response.return_value = (
        "This is a test response from the investor agent.",
        ValidationResult(
            is_valid=True,
            confidence_score=1.0,
            issues=[],
            validation_type="hallucination",
            validator_version="1.0",
        ),
    )

    agent = InvestorAgent(deps=AgentDependencies(rag_pipeline=mock_rag_pipeline))
    query = "What is the ROI on a property in Texas, TX?"
    session_id = "test_session"

    # Act
    response = await agent.run(query, session_id)

    # Assert
    assert isinstance(response, AgentResponse)
    assert response.content == "This is a test response from the investor agent."
    assert response.validation_result is not None
    mock_rag_pipeline.generate_response.assert_called_once()
