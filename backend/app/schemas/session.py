from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.chat import ChatResponse
from app.schemas.telemetry import TelemetrySummary


class MessageItem(BaseModel):
    id: str = Field(..., description="Unique message UUID")
    session_id: str = Field(..., description="ID of parent session")
    sender: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text content")
    response: Optional[ChatResponse] = Field(default=None, description="Full structured response if assistant")
    created_at: str = Field(..., description="ISO formatted creation timestamp")


class SessionItem(BaseModel):
    id: str = Field(..., description="Unique session ID")
    title: str = Field(..., description="User-facing title of session")
    active_document: Optional[str] = Field(default=None, description="Associated document filename")
    message_count: int = Field(default=0, description="Total messages in session")
    total_tokens: int = Field(default=0, description="Cumulative tokens consumed in session")
    total_cost_usd: float = Field(default=0.0, description="Total USD cost incurred in session")
    created_at: str = Field(..., description="ISO formatted creation timestamp")
    updated_at: str = Field(..., description="ISO formatted last activity timestamp")


class SessionDetail(SessionItem):
    messages: List[MessageItem] = Field(default_factory=list, description="Ordered conversation history")
    telemetry: TelemetrySummary = Field(..., description="Session-specific telemetry metrics")


class SessionCreate(BaseModel):
    title: Optional[str] = Field(default=None, description="Optional custom session title")
    active_document: Optional[str] = Field(default=None, description="Document filename to bind")


class SessionUpdate(BaseModel):
    title: Optional[str] = Field(default=None, description="Updated session title")
    active_document: Optional[str] = Field(default=None, description="Updated bound document")
