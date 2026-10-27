from fastapi import APIRouter
from app.core.config import settings
from app.schemas.common import BaseResponse, HealthStatus

router = APIRouter()


@router.get("/health", response_model=BaseResponse[HealthStatus])
async def health_check():
    """Returns connectivity status for Groq, Pinecone, and configured models."""
    keys_valid = settings.validate_api_keys()

    health_data = HealthStatus(
        status="ok",
        version="1.0.0",
        groq_configured=keys_valid["groq_configured"],
        pinecone_configured=keys_valid["pinecone_configured"],
        active_index=settings.PINECONE_INDEX_NAME,
        active_namespace=settings.PINECONE_NAMESPACE,
        text_model=settings.TEXT_MODEL,
        vision_model=settings.VISION_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
    )

    return BaseResponse(
        success=True,
        message="System health verified",
        data=health_data
    )
