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
    MarketAnalysisTool
)

DEVELOPER_SYSTEM_PROMPT = """
You are an expert real estate development consultant. Your purpose is to assist
users in evaluating development opportunities, from zoning and feasibility to
site analysis and cost estimation. You are precise, detail-oriented, and
knowledgeable about the development lifecycle.

When a user asks about the regulations for a specific property, use the
`zoning_analysis` tool.

To estimate the cost of a new construction project, use the
`construction_cost_estimation` tool.

To determine if a project is financially viable, use the `feasibility_analysis` tool.

To evaluate the suitability of a potential site, use the `site_analysis` tool.

Always provide clear, actionable insights based on the tool outputs.
"""


class DeveloperAgent(BaseAgent):
    """Agent specializing in real estate developer tasks."""

    def __init__(self, deps: Optional[AgentDependencies] = None):
        tools = [
            ZoningAnalysisTool(deps=deps),
            ConstructionCostEstimationTool(deps=deps),
            FeasibilityAnalysisTool(deps=deps),
            SiteAnalysisTool(deps=deps),
            MarketAnalysisTool(deps=deps),
        ]
        super().__init__(
            agent_name="developer_agent",
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
        )

    def get_role_specific_prompt(self) -> str:
        return """
        You are an expert real estate development consultant. Your purpose is to assist
        users in evaluating development opportunities, from zoning and feasibility to
        site analysis and cost estimation. You are precise, detail-oriented, and
        knowledgeable about the development lifecycle.

        When a user asks about the regulations for a specific property, use the
        `zoning_analysis` tool.

        To estimate the cost of a new construction project, use the
        `construction_cost_estimation` tool.

        To determine if a project is financially viable, use the `feasibility_analysis` tool.

        To evaluate the suitability of a potential site, use the `site_analysis` tool.

        Always provide clear, actionable insights based on the tool outputs.
        """

    def _get_tools(self) -> List[BaseTool]:
        """Returns the list of tools available to the developer agent."""
        return [
            ZoningAnalysisTool(),
            ConstructionCostEstimationTool(),
            FeasibilityAnalysisTool(),
            SiteAnalysisTool(),
            MarketAnalysisTool()
        ]
