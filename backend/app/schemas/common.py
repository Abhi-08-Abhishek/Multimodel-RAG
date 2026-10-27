from enum import Enum
from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class ModalityEnum(str, Enum):
    TEXT = "text"
    TABLE = "table"
    VISUAL = "visual"


class BaseResponse(BaseModel, Generic[DataT]):
    success: bool = Field(default=True, description="Indicates if the operation succeeded")
    message: str = Field(default="Operation completed successfully", description="Status message")
    data: Optional[DataT] = Field(default=None, description="Payload data")


class HealthStatus(BaseModel):
    status: str = Field(default="ok", description="Overall service status")
    version: str = Field(default="1.0.0", description="Application version")
    groq_configured: bool = Field(..., description="Whether Groq API key is valid")
    pinecone_configured: bool = Field(..., description="Whether Pinecone API key is valid")
    active_index: str = Field(..., description="Connected Pinecone index")
    active_namespace: str = Field(..., description="Active Pinecone namespace")
    text_model: str = Field(..., description="Current text LLM")
    vision_model: str = Field(..., description="Current vision LLM")
    embedding_model: str = Field(..., description="Current embedding model")
