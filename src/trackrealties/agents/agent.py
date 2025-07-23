"""
Agent-specific agent for the TrackRealties AI Platform.
"""
from typing import List, Optional

from .base import BaseAgent, AgentDependencies, BaseTool
from .tools import (
    VectorSearchTool,
    GraphSearchTool,
    MarketAnalysisTool,
    PropertyRecommendationTool,
    InvestmentOpportunityAnalysisTool,
    ROIProjectionTool,
    RiskAssessmentTool,
    ZoningAnalysisTool,
    ConstructionCostEstimationTool,
    FeasibilityAnalysisTool,
    SiteAnalysisTool,
)
from .prompts import BASE_SYSTEM_CONTEXT, AGENT_SYSTEM_PROMPT


class AgentAgent(BaseAgent):
    """Agent specializing in real estate agent tasks."""

    MODEL_PATH = "models/agent_llm"

    def __init__(self, deps: Optional[AgentDependencies] = None, model_path: Optional[str] = None):
        role_models = getattr(deps.rag_pipeline, "role_models", {}) if deps else {}
        model = role_models.get("agent") if role_models else None

        tools = [
            VectorSearchTool(deps=deps),
            GraphSearchTool(deps=deps),
            MarketAnalysisTool(deps=deps),
            PropertyRecommendationTool(deps=deps),
            InvestmentOpportunityAnalysisTool(deps=deps),
            ROIProjectionTool(deps=deps),
            RiskAssessmentTool(deps=deps),
            ZoningAnalysisTool(deps=deps),
            ConstructionCostEstimationTool(deps=deps),
            FeasibilityAnalysisTool(deps=deps),
            SiteAnalysisTool(deps=deps),
        ]
        super().__init__(
            agent_name="agent_agent",
            model=model,
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
            model_path=model_path or self.MODEL_PATH,
        )

    def get_role_specific_prompt(self) -> str:
        return f"{BASE_SYSTEM_CONTEXT}\n{AGENT_SYSTEM_PROMPT}"

    def _get_tools(self, deps: Optional[AgentDependencies] = None) -> List[BaseTool]:
        """Returns the list of all tools available to the agent."""
        return [
            VectorSearchTool(deps=deps),
            GraphSearchTool(deps=deps),
            MarketAnalysisTool(deps=deps),
            PropertyRecommendationTool(deps=deps),
            InvestmentOpportunityAnalysisTool(deps=deps),
            ROIProjectionTool(deps=deps),
            RiskAssessmentTool(deps=deps),
            ZoningAnalysisTool(deps=deps),
            ConstructionCostEstimationTool(deps=deps),
            FeasibilityAnalysisTool(deps=deps),
            SiteAnalysisTool(deps=deps),
        ]
