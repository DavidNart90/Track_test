"""
Developer-specific agent for the TrackRealties AI Platform.
"""
from typing import List, Type, Optional
from .base import BaseAgent, AgentDependencies, BaseTool
from .tools import (
    ZoningAnalysisTool,
    ConstructionCostEstimationTool,
    FeasibilityAnalysisTool,
    SiteAnalysisTool,
    MarketAnalysisTool,
)
from .prompts import BASE_SYSTEM_CONTEXT, DEVELOPER_SYSTEM_PROMPT


class DeveloperAgent(BaseAgent):
    """Agent specializing in real estate developer tasks."""

    MODEL_PATH = "models/developer_llm"

    def __init__(self, deps: Optional[AgentDependencies] = None, model_path: Optional[str] = None):
        role_models = getattr(deps.rag_pipeline, "role_models", {}) if deps else {}
        model = role_models.get("developer") if role_models else None
        tools = [
            ZoningAnalysisTool(deps=deps),
            ConstructionCostEstimationTool(deps=deps),
            FeasibilityAnalysisTool(deps=deps),
            SiteAnalysisTool(deps=deps),
            MarketAnalysisTool(deps=deps),
        ]
        super().__init__(
            agent_name="developer_agent",
            model=model,
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
            model_path=model_path or self.MODEL_PATH,
        )

    def get_role_specific_prompt(self) -> str:
        return f"{BASE_SYSTEM_CONTEXT}\n{DEVELOPER_SYSTEM_PROMPT}"

    def _get_tools(self, deps: Optional[AgentDependencies] = None) -> List[BaseTool]:
        """Returns the list of tools available to the developer agent."""
        return [
            ZoningAnalysisTool(deps=deps),
            ConstructionCostEstimationTool(deps=deps),
            FeasibilityAnalysisTool(deps=deps),
            SiteAnalysisTool(deps=deps),
            MarketAnalysisTool(deps=deps),
        ]
