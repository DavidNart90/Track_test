"""
Base classes for all agents in the TrackRealties AI Platform.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from pydantic_ai.agent import Agent as PydanticAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider as OpenAI

from ..core.config import settings
from ..models.agent import ValidationResult
from pydantic_ai.models.openai import OpenAIModel
from ..rag.pipeline import RAGPipeline
from ..rag.optimized_pipeline import EnhancedRAGPipeline, GreetingDetector
from ..models.db import ConversationMessage, MessageRole
from ..rag.pipeline import RAGPipeline
from ..rag.rag_pipeline_integration import EnhancedRAGPipeline
from ..validation.base import ResponseValidator
from .context import ContextManager
from .prompts import GREETINGS_PROMPT 


logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    """Standard response format for all agents."""

    content: str
    tools_used: list[dict[str, Any]] = Field(default_factory=list)
    validation_result: dict[str, Any] | None = None
    confidence_score: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(
        self, name: str, description: str, deps: Optional["AgentDependencies"] = None
    ):
        self.name = name
        self.description = description
        self.dependencies = deps or AgentDependencies()

    @abstractmethod
    async def execute(self, **kwargs) -> dict[str, Any]:
        pass

    def as_function(self):
        """Returns the execute method with a unique name for PydanticAI."""
        from functools import wraps

        @wraps(self.execute)
        async def wrapper(**kwargs):
            return await self.execute(**kwargs)

        wrapper.__name__ = self.name
        return wrapper
    



class AgentDependencies(BaseModel):
    """Dependencies needed by agents and tools."""

    context_manager: ContextManager = Field(default_factory=ContextManager)
    rag_pipeline: RAGPipeline = Field(default_factory=EnhancedRAGPipeline)

    class Config:
        arbitrary_types_allowed = True

    # Add other dependencies like database connections, etc.


class BaseAgent(ABC):
    """
    The BaseAgent class provides the foundational functionality that all
    role-specific agents inherit.
    """

    def __init__(
        self,
        agent_name: str,
        model: str | OpenAI = None,
        *,
        model_path: str | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        tools: list[type[BaseTool]] = None,
        deps: AgentDependencies | None = None,
        validator: ResponseValidator | None = None,
        **kwargs,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.dependencies = deps or AgentDependencies()
        self.validator = validator
        self.tools = {tool.name: tool for tool in (tools or [])}

        self.model_path = model_path

        _model = model or settings.DEFAULT_MODEL
        if isinstance(_model, str):
            # Assuming format "provider:model_name" e.g., "openai:gpt-4o"
            provider, model_name = _model.split(":")
            if provider == "openai":
                provider = OpenAI(api_key=settings.OPENAI_API_KEY)
                self.llm = OpenAIModel(model_name, provider=provider)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        else:
            self.llm = _model

        self.agent = PydanticAI(
            llm=self.llm,
            system_prompt=self.get_role_specific_prompt(),
            tools=[tool.as_function() for tool in self.tools.values()],
        )

    @abstractmethod
    def get_role_specific_prompt(self) -> str:
        """Returns the role-specific part of the system prompt."""
        pass

    def add_tool(self, tool: type[BaseTool]):
        """Adds a tool to the agent."""
        self.tools[tool.name] = tool
        # Re-initialize PydanticAI with the new toolset
        self.agent = PydanticAI(
            llm=self.llm,
            system_prompt=self.get_role_specific_prompt(),
            tools=[t.as_function() for t in self.tools.values()],
        )

    async def run(
        self,
        message: str,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Process a message with greeting detection."""
        try:
            # Get or create context
            context = self.dependencies.context_manager.get_or_create_context(
                session_id=session_id or "default",
                user_id=user_id,
                user_role=user_role,
            )
            user_message = ConversationMessage(
                role=MessageRole.USER,
                content=message,
                session_id=session_id or "default",
                created_at=datetime.now(timezone.utc),
            )
            context.add_message(user_message)

            # Check for greeting first
            if GreetingDetector.is_greeting(message):
                logger.info(f"Processing greeting from {user_role or 'user'}")

                # If greeting detected, respond with the greetings prompt
                greeting_prompt = GREETINGS_PROMPT
                # Add role-specific context to the greeting prompt
                role_context = {
                    "investor": "\n\nThe user is an investor, so mention how you can help with ROI analysis, market trends, and investment opportunities.",
                    "developer": "\n\nThe user is a developer, so mention how you can assist with site analysis, feasibility studies, and development opportunities.",
                    "buyer": "\n\nThe user is a home buyer, so mention how you can help find the perfect property, analyze neighborhoods, and guide through the buying process.",
                    "agent": "\n\nThe user is a real estate agent, so mention how you can provide market intelligence, lead insights, and business growth strategies."
                }
                greeting_prompt += role_context.get(user_role, "")

                # Create temporary agent with greeting prompt
                # Extract model name from the default model string (e.g., "openai:gpt-4")
                model_name = settings.DEFAULT_MODEL.split(":")[-1] if isinstance(settings.DEFAULT_MODEL, str) else "gpt-4"
                
                greeting_agent = PydanticAI(
                    model=model_name,  # Set the model name explicitly
                    llm=self.llm,
                    system_prompt=greeting_prompt,
                    tools=[tool.as_function() for tool in self.tools.values()]
                )

                # Generate greeting response using LLM
                result = await greeting_agent.run(message)
                response_content = str(result.output)

                # Create agent response
                return AgentResponse(
                    content=response_content,
                    tools_used=[{
                        "tool_name": "greeting_handler",
                        "args": {"message": message}
                    }],
                    confidence_score=1.0,
                    metadata={"response_type": "greeting", "user_role": user_role}
                )
            else:
                try:
                    # Use the RAG pipeline to generate a response
                    response_content, validation = (
                        await self.dependencies.rag_pipeline.generate_response(
                            query=message, context=context.dict()
                        )
                    )

                    agent_response = AgentResponse(content=response_content)
                    """codex/implement-realestatehallucinationdetector-and-tests"""
                    combined_result = validation

                    # Append assistant message to history
                    assistant_msg = ConversationMessage(
                        session_id=session_id,
                        role=MessageRole.ASSISTANT,
                        content=agent_response.content,
                    )
                    context.add_message(assistant_msg)

                    # Persist updated context
                    self.dependencies.context_manager.update_context(session_id, context)
                    # Planning-for-Phase-2-Docs

                    if self.validator:
                        extra = await self.validator.validate_response(
                            response=agent_response.content, context=context, user_query=message
                        )
                        combined_result = ValidationResult(
                            is_valid=validation.is_valid and extra.is_valid,
                            confidence_score=min(
                                validation.confidence_score, extra.confidence_score
                            ),
                            issues=validation.issues + extra.issues,
                            validation_type="combined",
                            validator_version="1.0",
                            correction_needed=validation.correction_needed or extra.correction_needed,
                        )

                    agent_response.validation_result = combined_result.dict()
                    agent_response.confidence_score = combined_result.confidence_score

                    # Mock tool usage for now
                    agent_response.tools_used = []

                    return agent_response

                except Exception as e:
                    logger.error(f"Agent {self.agent_name} failed to run: {e}", exc_info=True)
                    return AgentResponse(
                        content=f"I'm sorry, but I encountered an error: {e}",
                        confidence_score=0.0,
                    )
        except Exception as e:
            logger.error(f"Unexpected error in Agent {self.agent_name}: {e}", exc_info=True)
            return AgentResponse(
                content=f"I'm sorry, but I encountered an unexpected error: {e}",
                confidence_score=0.0,
            )

    async def stream(
        self,
        message: str,
        session_id: str,
        user_id: str | None = None,
        user_role: str | None = None,
    ):
        """
        Streams the response from the agent.
        """
        self.dependencies.context_manager.get_or_create_context(
            session_id, user_id, user_role
        )

        # Placeholder for streaming implementation
        response = await self.run(message, session_id, user_id, user_role)
        for char in response.content:
            yield char

    def get_context(self, session_id: str):
        """Gets the conversation context for a session."""
        return self.dependencies.context_manager.get_context(session_id)

    def clear_context(self, session_id: str):
        """Clears the context for a session."""
        self.dependencies.context_manager.clear_context(session_id)
