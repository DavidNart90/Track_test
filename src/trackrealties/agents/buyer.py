"""
Buyer-specific agent for the TrackRealties AI Platform.
"""
from typing import List, Type, Optional
from .base import BaseAgent, AgentDependencies, BaseTool
from .tools import (
    VectorSearchTool,
    PropertyRecommendationTool,
    MarketAnalysisTool,
)
from .prompts import BASE_SYSTEM_CONTEXT, BUYER_SYSTEM_PROMPT



class BuyerAgent(BaseAgent):
    """Agent specialized in assisting home buyers."""

    def __init__(self, deps: Optional[AgentDependencies] = None):
        tools = self._get_tools(deps)

        role_models = getattr(deps.rag_pipeline, "role_models", {}) if deps else {}
        model = role_models.get("buyer") if role_models else None

        super().__init__(
            agent_name="buyer_agent",
            model=model,
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
        )

    def get_role_specific_prompt(self) -> str:
        return f"{BASE_SYSTEM_CONTEXT}\n{BUYER_SYSTEM_PROMPT}"

    def _get_tools(self, deps: Optional[AgentDependencies] = None) -> List[BaseTool]:
        """Returns the list of tools available to the buyer agent."""
        return [
            VectorSearchTool(deps=deps),
            PropertyRecommendationTool(deps=deps),
            MarketAnalysisTool(deps=deps)
        ]
