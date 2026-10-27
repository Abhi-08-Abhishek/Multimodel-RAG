from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.api_router import api_router
from app.core.config import settings
from app.core.exceptions import RAGException, generic_exception_handler, rag_exception_handler
from app.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown initialization."""
    logger.info("Initializing NovaCore Multimodal RAG Backend...")
    settings.ensure_directories()
    keys = settings.validate_api_keys()
    if not keys["groq_configured"]:
        logger.warning("GROQ_API_KEY is not set or using placeholder in .env. RAG inference will require valid keys.")
    if not keys["pinecone_configured"]:
        logger.warning("PINECONE_API_KEY is not set or using placeholder in .env. Vector search will require valid keys.")
    yield
    logger.info("Shutting down NovaCore Multimodal RAG Backend.")


def create_application() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="NovaCore Multimodal RAG API",
        description="Enterprise Multimodal RAG combining Text, Tables, and Charts via Groq Vision & Pinecone",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handlers
    app.add_exception_handler(RAGException, rag_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Static File Mounts for extracted charts and uploaded PDFs
    app.mount(
        "/static/extracted_images",
        StaticFiles(directory=str(settings.EXTRACTED_IMAGES_DIR)),
        name="extracted_images"
    )
    app.mount(
        "/static/uploads",
        StaticFiles(directory=str(settings.UPLOADS_DIR)),
        name="uploads"
    )

    # Include API Routers
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {
            "app": "NovaCore Multimodal RAG API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
            "status": "online",
        }

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
