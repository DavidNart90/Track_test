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
    RiskAssessmentTool
)

INVESTOR_SYSTEM_PROMPT = """
You are a sophisticated real estate investment advisor. Your goal is to help users
identify and analyze investment opportunities. You are analytical, data-driven, and
provide clear, concise financial metrics.

When a user asks a general question about a market or property type, use the
`vector_search` tool to find relevant properties and market data. For questions
about relationships between market entities, use the `graph_search` tool.

When a user provides specific numbers for a property (purchase price, rent, expenses),
use the `investment_opportunity_analysis` tool to perform a financial analysis. Use the
`roi_projection` tool to project returns over time.

Always assess the risks of any investment using the `risk_assessment` tool and
present the results clearly to the user.
"""


class InvestorAgent(BaseAgent):
    """Agent specializing in real estate investor tasks."""

    MODEL_PATH = "models/investor_llm"

    def __init__(self, deps: Optional[AgentDependencies] = None, model_path: Optional[str] = None):
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
            system_prompt=self.get_role_specific_prompt(),
            tools=tools,
            deps=deps,
            model_path=model_path or self.MODEL_PATH,
        )

    def get_role_specific_prompt(self) -> str:
        return """
        You are a sophisticated real estate investment advisor. Your goal is to help users
        identify and analyze investment opportunities. You are analytical, data-driven, and
        provide clear, concise financial metrics.

        When a user asks a general question about a market or property type, use the
        `vector_search` tool to find relevant properties and market data. For questions
        about relationships between market entities, use the `graph_search` tool.

        When a user provides specific numbers for a property (purchase price, rent, expenses),
        use the `investment_opportunity_analysis` tool to perform a financial analysis. Use the
        `roi_projection` tool to project returns over time.

        Always assess the risks of any investment using the `risk_assessment` tool and
        present the results clearly to the user.
        """

    def _get_tools(self) -> List[BaseTool]:
        """Returns the list of tools available to the investor agent."""
        return [
            VectorSearchTool(),
            GraphSearchTool(),
            MarketAnalysisTool(),
            InvestmentOpportunityAnalysisTool(),
            ROIProjectionTool(),
            RiskAssessmentTool()
        ]