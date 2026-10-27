from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.common import ModalityEnum
from app.schemas.telemetry import TokenUsage


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question or query")
    session_id: Optional[str] = Field(default=None, description="Optional conversation session ID")
    namespace: Optional[str] = Field(default=None, description="Target Pinecone namespace (defaults to setting)")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context passages to retrieve")
    stream: bool = Field(default=False, description="Whether to stream response tokens")


class CitationSource(BaseModel):
    id: str = Field(..., description="Document / Chunk ID")
    page: int = Field(..., description="Page number in original PDF")
    modality: ModalityEnum = Field(..., description="Document modality (text, table, visual)")
    content_preview: str = Field(..., description="Preview snippet of the chunk")
    full_content: str = Field(..., description="Full text or table or visual description")
    image_url: Optional[str] = Field(default=None, description="Public URL to extracted visual image")
    table_number: Optional[int] = Field(default=None, description="Table index on the page")
    source_document: str = Field(default="", description="Source PDF filename")
    score: Optional[float] = Field(default=None, description="Similarity score from Pinecone")


class ChatResponse(BaseModel):
    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer from RAG pipeline")
    session_id: Optional[str] = Field(default=None, description="Associated session ID")
    sources: List[CitationSource] = Field(default_factory=list, description="Retrieved evidence chunks")
    has_visuals: bool = Field(default=False, description="Whether retrieved context contained visual evidence")
    visual_images: List[str] = Field(default_factory=list, description="List of image URLs used in multimodal reasoning")
    model_used: str = Field(..., description="Model that produced the final synthesis")
    latency_seconds: float = Field(..., description="Processing time in seconds")
    token_usage: Optional[TokenUsage] = Field(default=None, description="Detailed token metrics and cost in USD")

