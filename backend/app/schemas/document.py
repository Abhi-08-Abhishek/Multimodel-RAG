from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    document_name: str = Field(
        ...,
        min_length=1,
        description="Name of the PDF file in storage/uploads/ to ingest and index"
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Target Pinecone namespace (defaults to setting)"
    )
    clean_namespace_first: bool = Field(
        default=True,
        description="Whether to clear existing vectors in the namespace prior to indexing"
    )


class ExtractedStats(BaseModel):
    total_pages: int = Field(default=0, description="Total pages processed")
    text_chunks: int = Field(default=0, description="Extracted text document count")
    tables_extracted: int = Field(default=0, description="Extracted table document count")
    visuals_extracted: int = Field(default=0, description="Extracted visual/image document count")
    total_documents: int = Field(default=0, description="Total LangChain documents indexed")


class IngestResponse(BaseModel):
    document_name: str = Field(..., description="Ingested PDF document name")
    namespace: str = Field(..., description="Pinecone namespace where vectors are indexed")
    stats: ExtractedStats = Field(..., description="Ingestion statistics breakdown")
    duration_seconds: float = Field(..., description="Time taken to extract, summarize, and index")


class DocumentItem(BaseModel):
    filename: str = Field(..., description="PDF filename")
    size_bytes: int = Field(..., description="File size in bytes")
    modified_at: str = Field(..., description="Last modification timestamp")
    is_indexed: bool = Field(default=False, description="Whether document is currently indexed")
    image_count: int = Field(default=0, description="Number of extracted visuals on disk")
