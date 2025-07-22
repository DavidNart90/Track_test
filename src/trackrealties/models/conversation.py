from sqlalchemy import (
    Column, String, DateTime, ForeignKey, Text, Integer, Numeric, CheckConstraint, text
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.trackrealties.models.base import Base
from .session import ChatSession


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey('user_sessions.id', ondelete='CASCADE'), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    tools_used = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    validation_result = Column(JSONB, nullable=True)
    confidence_score = Column(Numeric(3, 2), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    message_metadata = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name='conversation_messages_role_check'),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name='conversation_messages_confidence_score_check'),
    )


# Pydantic Models for API Interaction

class MessageCreateRequest(BaseModel):
    session_id: uuid.UUID
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    tools_used: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    validation_result: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    processing_time_ms: Optional[int] = None
    token_count: Optional[int] = None
    message_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MessageUpdateRequest(BaseModel):
    content: Optional[str] = None
    tools_used: Optional[List[Dict[str, Any]]] = None
    validation_result: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    message_metadata: Optional[Dict[str, Any]] = None


class ConversationMessageResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    tools_used: List[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    confidence_score: Optional[float]
    processing_time_ms: Optional[int]
    token_count: Optional[int]
    message_metadata: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationHistory(BaseModel):
    session: ChatSession
    messages: List[ConversationMessageResponse]
    total_messages: int
    has_more: bool

    class Config:
        from_attributes = True
        arbitrary_types_allowed = True


class ConversationSummary(BaseModel):
    session_id: uuid.UUID
    user_id: str
    title: str
    last_message_at: datetime
    message_count: int
    topics: List[str]
    preview: str


class ConversationFilter(BaseModel):
    user_id: Optional[uuid.UUID] = None
    role_filter: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    contains_text: Optional[str] = None
    property_reference: Optional[str] = None
    market_reference: Optional[str] = None
    limit: int = 50
    offset: int = 0