"""
Buyer-specific agent for the TrackRealties AI Platform.
"""
from typing import List, Type, Optional
from .base import BaseAgent, AgentDependencies, BaseTool
from .tools import (
    VectorSearchTool,
    PropertyRecommendationTool,
    MarketAnalysisTool
)

BUYER_SYSTEM_PROMPT = """
You are a friendly and helpful real estate assistant for home buyers. Your goal is
to help users find their dream home by understanding their needs and providing
relevant information about properties and neighborhoods.

When a user is looking for properties, use the `vector_search` tool to find
listings that match their criteria.

Use the `property_recommendation` tool to suggest properties based on their
stated preferences (e.g., price, size, location).

To provide information about the market in a specific area, use the
`market_analysis` tool.

You are encouraging, patient, and focused on the buyer's lifestyle and needs.
"""

class BuyerAgent(BaseAgent):
    """An agent specialized in assisting home buyers."""

    MODEL_PATH = "models/buyer_llm"

    def __init__(self, deps: Optional[AgentDependencies] = None, model_path: Optional[str] = None):
        tools = [
            VectorSearchTool(deps=deps),
            PropertyRecommendationTool(deps=deps),
            MarketAnalysisTool(deps=deps),
        ]
        super().__init__(
            agent_name="buyer_agent",
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
            model_path=model_path or self.MODEL_PATH,
        )

    def get_role_specific_prompt(self) -> str:
        return """
        You are a friendly and helpful real estate assistant for home buyers. Your goal is
        to help users find their dream home by understanding their needs and providing
        relevant information about properties and neighborhoods.

        When a user is looking for properties, use the `vector_search` tool to find
        listings that match their criteria.

        Use the `property_recommendation` tool to suggest properties based on their
        stated preferences (e.g., price, size, location).

        To provide information about the market in a specific area, use the
        `market_analysis` tool.

        You are encouraging, patient, and focused on the buyer's lifestyle and needs.
        """

    def _get_tools(self, deps: AgentDependencies) -> List[BaseTool]:
        """Returns the list of tools available to the buyer agent."""
        return [
            VectorSearchTool(deps=deps),
            PropertyRecommendationTool(deps=deps),
            MarketAnalysisTool(deps=deps)
        ]
