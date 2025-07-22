"""
Factory for creating agent instances.
"""
from typing import Type
from .base import BaseAgent
from .roles import UserRole
from .investor import InvestorAgent
from .developer import DeveloperAgent
from .buyer import BuyerAgent
from .agent import AgentAgent

def get_agent_class(role: UserRole) -> Type[BaseAgent]:
    """Returns the agent class corresponding to the user role."""
    if role == UserRole.INVESTOR:
        return InvestorAgent
    elif role == UserRole.DEVELOPER:
        return DeveloperAgent
    elif role == UserRole.BUYER:
        return BuyerAgent
    elif role == UserRole.AGENT:
        return AgentAgent
    else:
        # Fallback to a general agent if no specific role matches
        # For now, we raise an error as per the original logic.
        raise NotImplementedError(f"No agent implemented for role: {role.value}")