"""
Agent-specific agent for the TrackRealties AI Platform.
"""
from typing import List, Type, Optional
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
    SiteAnalysisTool
)

AGENT_SYSTEM_PROMPT = """
You are an expert real estate agent's assistant. You have access to a complete
set of tools for market analysis, investment evaluation, and development assessment.
Your purpose is to provide comprehensive, accurate, and actionable information to
help real estate agents serve their clients effectively.

You can perform any task from searching for properties, analyzing investments,
evaluating development sites, to providing market intelligence. Be ready to
switch between different roles (buyer's agent, listing agent, etc.) as needed.

Always be professional, thorough, and proactive in your responses.
"""


class AgentAgent(BaseAgent):
    """Agent specializing in real estate agent tasks."""

    def __init__(self, deps: Optional[AgentDependencies] = None):
        tools = [
            VectorSearchTool(deps=deps),
            GraphSearchTool(deps=deps),
            MarketAnalysisTool(deps=deps),
            PropertyRecommendationTool(deps=deps),
            InvestmentOpportunityAnalysisTool(deps=deps),
            ROIProjectionTool(deps=deps),
            RiskAssessmentTool(deps=deps),
        ]

        super().__init__(
            agent_name="agent_agent",
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
        )

    def get_role_specific_prompt(self) -> str:
        return """
        You are an expert real estate agent's assistant. You have access to a complete
        set of tools for market analysis, investment evaluation, and development assessment.
        Your purpose is to provide comprehensive, accurate, and actionable information to
        help real estate agents serve their clients effectively.

        You can perform any task from searching for properties, analyzing investments,
        evaluating development sites, to providing market intelligence. Be ready to
        switch between different roles (buyer's agent, listing agent, etc.) as needed.

        Always be professional, thorough, and proactive in your responses.
        """

    def _get_tools(self) -> List[BaseTool]:
        """Returns the list of all tools available to the agent."""
        return [
            VectorSearchTool(),
            GraphSearchTool(),
            MarketAnalysisTool(),
            PropertyRecommendationTool(),
            InvestmentOpportunityAnalysisTool(),
            ROIProjectionTool(),
            RiskAssessmentTool(),
            ZoningAnalysisTool(),
            ConstructionCostEstimationTool(),
            FeasibilityAnalysisTool(),
            SiteAnalysisTool()
        ]
