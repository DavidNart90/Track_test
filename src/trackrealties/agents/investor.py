"""
Investor-specific agent for the TrackRealties AI Platform.
"""
from typing import List, Type, Optional
from .base import BaseAgent, AgentDependencies, BaseTool
from .tools import (
    VectorSearchTool,
    GraphSearchTool,
    MarketAnalysisTool,
    InvestmentOpportunityAnalysisTool,
    ROIProjectionTool,
    RiskAssessmentTool,
)
from .prompts import BASE_SYSTEM_CONTEXT, INVESTOR_SYSTEM_PROMPT

class InvestorAgent(BaseAgent):
    """Agent specializing in real estate investor tasks."""

    MODEL_PATH = "models/investor_llm"

    def __init__(self, deps: Optional[AgentDependencies] = None, model_path: Optional[str] = None):
        role_models = getattr(deps.rag_pipeline, "role_models", {}) if deps else {}
        model = role_models.get("investor") if role_models else None

        tools = [
            VectorSearchTool(deps=deps),
            GraphSearchTool(deps=deps),
            MarketAnalysisTool(deps=deps),
            InvestmentOpportunityAnalysisTool(deps=deps),
            ROIProjectionTool(deps=deps),
            RiskAssessmentTool(deps=deps),
        ]
        super().__init__(
            agent_name="investor_agent",
            model=model,
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
            model_path=model_path or self.MODEL_PATH,
        )

    def get_role_specific_prompt(self) -> str:
        return f"{BASE_SYSTEM_CONTEXT}\n{INVESTOR_SYSTEM_PROMPT}"

    def _get_tools(self, deps: Optional[AgentDependencies] = None) -> List[BaseTool]:
        """Returns the list of tools available to the investor agent."""
        return [
            VectorSearchTool(deps=deps),
            GraphSearchTool(deps=deps),
            MarketAnalysisTool(deps=deps),
            InvestmentOpportunityAnalysisTool(deps=deps),
            ROIProjectionTool(deps=deps),
            RiskAssessmentTool(deps=deps),
        ]