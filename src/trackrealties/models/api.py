
"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from uuid import UUID
from datetime import datetime

class UserRole(str, Enum):
    INVESTOR = "investor"
    DEVELOPER = "developer"
    BUYER = "buyer"
    AGENT = "agent"
    GENERAL = "general"

class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None
    role: UserRole

class SessionCreateResponse(BaseModel):
    session_id: str
    role: UserRole
    created_at: datetime

class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: Optional[str] = None

class ToolCall(BaseModel):
    tool_name: str
    args: Dict[str, Any]

class ChatResponse(BaseModel):
    message: str
    session_id: str
    assistant_message_id: str
    tools_used: List[ToolCall] = []
    metadata: Dict[str, Any] = {}

class StreamDelta(BaseModel):
    type: str  # "text", "tool", "session", "end", "error"
    content: Any

class HealthStatus(BaseModel):
    status: str
    database: bool
    graph_database: bool
    llm_connection: bool
    version: str
    timestamp: datetime
