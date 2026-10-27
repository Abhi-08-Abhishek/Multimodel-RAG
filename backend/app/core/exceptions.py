from typing import Any, Dict, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from app.core.logging import logger


class RAGException(Exception):
    """Base exception for Multimodal RAG application."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(RAGException):
    """Raised when critical configuration or API keys are missing."""
    pass


class ExtractionError(RAGException):
    """Raised when PDF extraction (text, tables, or images) fails."""
    pass


class VisionProcessingError(RAGException):
    """Raised when Groq Vision API fails to process an image."""
    pass


class VectorStoreError(RAGException):
    """Raised when Pinecone index operations or upserts fail."""
    pass


class RetrievalError(RAGException):
    """Raised during similarity search or query embedding."""
    pass


async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    """Global exception handler converting domain exceptions to structured JSON."""
    logger.error(f"Domain error processing {request.method} {request.url.path}: {exc.message} | Details: {exc.details}")
    
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if isinstance(exc, ConfigurationError):
        status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, ExtractionError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(exc, VectorStoreError):
        status_code = status.HTTP_502_BAD_GATEWAY
        
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error_type": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
            "path": request.url.path,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for unhandled internal exceptions."""
    logger.exception(f"Unexpected error on {request.method} {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "InternalServerError",
            "message": str(exc),
            "path": request.url.path,
        },
    )
