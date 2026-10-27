from fastapi import APIRouter
from app.api.v1.endpoints import chat, documents, health, ingest, sessions, telemetry

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(documents.router, tags=["Documents"])
api_router.include_router(ingest.router, tags=["Ingestion"])
api_router.include_router(chat.router, tags=["Chat"])
api_router.include_router(telemetry.router, tags=["Telemetry"])
api_router.include_router(sessions.router, tags=["Sessions"])
