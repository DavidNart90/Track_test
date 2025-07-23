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
    """An agent specialized in assisting home buyers."""

    MODEL_PATH = "models/buyer_llm"

    def __init__(self, deps: Optional[AgentDependencies] = None, model_path: Optional[str] = None):
        role_models = getattr(deps.rag_pipeline, "role_models", {}) if deps else {}
        model = role_models.get("buyer") if role_models else None
        tools = [
            VectorSearchTool(deps=deps),
            PropertyRecommendationTool(deps=deps),
            MarketAnalysisTool(deps=deps),
        ]
        super().__init__(
            agent_name="buyer_agent",
            model=model,
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
            model_path=model_path or self.MODEL_PATH,
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
