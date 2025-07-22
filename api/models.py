"""
Pydantic models for the TrackRealties API.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

class UserSession(BaseModel):
    """
    Pydantic model for a user session.
    """
    id: UUID = Field(default_factory=uuid4)
    user_role: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True

class ConversationMessage(BaseModel):
    """
    Pydantic model for a conversation message.
    """
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True

class ChatRequest(BaseModel):
    """
    Pydantic model for a chat request.
    """
    query: str
    session_id: Optional[UUID] = None

class ChatResponse(BaseModel):
    """
    Pydantic model for a chat response.
    """
    message: str
    session_id: UUID