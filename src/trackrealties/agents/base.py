"""
Base classes for all agents in the TrackRealties AI Platform.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Type
import logging
from pydantic import BaseModel, Field
from pydantic_ai.agent import Agent as PydanticAI
from pydantic_ai.providers.openai import OpenAIProvider as OpenAI

from ..core.config import settings
from .context import ContextManager
from ..validation.base import ResponseValidator
from ..models.agent import ValidationResult
from pydantic_ai.models.openai import OpenAIModel
from ..rag.pipeline import RAGPipeline
from rag_pipeline_integration import EnhancedRAGPipeline
from ..models.db import ConversationMessage, MessageRole

logger = logging.getLogger(__name__)

class AgentResponse(BaseModel):
    """Standard response format for all agents."""
    content: str
    tools_used: List[Dict[str, Any]] = Field(default_factory=list)
    validation_result: Optional[Dict[str, Any]] = None
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

from functools import partial

class BaseTool(ABC):
    """Abstract base class for all tools."""
    def __init__(self, name: str, description: str, deps: Optional['AgentDependencies'] = None):
        self.name = name
        self.description = description
        self.dependencies = deps or AgentDependencies()

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
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
        model: Union[str, OpenAI] = None,
        *,
        model_path: Optional[str] = None,
        system_prompt: str = "You are a helpful AI assistant.",
        tools: List[Type[BaseTool]] = None,
        deps: Optional[AgentDependencies] = None,
        validator: Optional[ResponseValidator] = None,
        **kwargs
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
            tools=[tool.as_function() for tool in self.tools.values()]
        )

    @abstractmethod
    def get_role_specific_prompt(self) -> str:
        """Returns the role-specific part of the system prompt."""
        pass

    def add_tool(self, tool: Type[BaseTool]):
        """Adds a tool to the agent."""
        self.tools[tool.name] = tool
        # Re-initialize PydanticAI with the new toolset
        self.agent = PydanticAI(
            llm=self.llm,
            system_prompt=self.get_role_specific_prompt(),
            tools=[t.as_function() for t in self.tools.values()]
        )

    async def run(self, message: str, session_id: str, user_id: Optional[str] = None, user_role: Optional[str] = None) -> AgentResponse:
        """
        Runs the agent for a single turn.
        """
        context = self.dependencies.context_manager.get_or_create_context(session_id, user_id, user_role)

        # Record the user message in the conversation history
        user_msg = ConversationMessage(
            session_id=session_id,
            role=MessageRole.USER,
            content=message,
        )
        context.add_message(user_msg)
        
        try:
            # Use the RAG pipeline to generate a response
            response_content, validation = await self.dependencies.rag_pipeline.generate_response(
                query=message,
                context=context.dict()
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
        """Planning-for-Phase-2-Docs"""

            if self.validator:
                extra = await self.validator.validate_response(
                    response=agent_response.content,
                    context=context,
                    user_query=message
                )
                combined_result = ValidationResult(
                    is_valid=validation.is_valid and extra.is_valid,
                    confidence_score=min(validation.confidence_score, extra.confidence_score),
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
                confidence_score=0.0
            )

    async def stream(self, message: str, session_id: str, user_id: Optional[str] = None, user_role: Optional[str] = None):
        """
        Streams the response from the agent.
        """
        context = self.dependencies.context_manager.get_or_create_context(session_id, user_id, user_role)
        
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
