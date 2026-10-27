from pathlib import Path
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings dynamically loaded from environment variables and .env file.
    No hardcoded fallbacks for API credentials.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    GROQ_API_KEY: str = Field(default="", description="Groq API Key for LLM and Vision models")
    PINECONE_API_KEY: str = Field(default="", description="Pinecone Vector Database API Key")

    # Pinecone Index Config
    PINECONE_INDEX_NAME: str = Field(default="novacore-multimodal-rag", description="Pinecone index name")
    PINECONE_NAMESPACE: str = Field(default="fy2026-demo", description="Pinecone partition namespace")
    PINECONE_CLOUD: str = Field(default="aws", description="Pinecone cloud provider")
    PINECONE_REGION: str = Field(default="us-east-1", description="Pinecone cloud region")

    # LLM & Embedding Models
    TEXT_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Groq text generation model")
    VISION_MODEL: str = Field(default="llama-3.2-11b-vision-preview", description="Groq multimodal vision model")
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="HuggingFace text embedding model")

    # Server & Networking
    HOST: str = Field(default="0.0.0.0", description="Server bind host")
    PORT: int = Field(default=8000, description="Server port")
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins for Next.js frontend"
    )
    DEBUG: bool = Field(default=True, description="Debug mode")

    # Storage Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    STORAGE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "storage")
    UPLOADS_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "storage" / "uploads")
    EXTRACTED_IMAGES_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "storage" / "extracted_images")

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                return json.loads(v)
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    def ensure_directories(self) -> None:
        """Ensure all storage directories exist on disk."""
        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    def validate_api_keys(self) -> dict[str, bool]:
        """Validate if required API keys are configured and not placeholders."""
        is_groq_valid = bool(self.GROQ_API_KEY and "your_groq" not in self.GROQ_API_KEY.lower())
        is_pinecone_valid = bool(self.PINECONE_API_KEY and "your_pinecone" not in self.PINECONE_API_KEY.lower())
        return {
            "groq_configured": is_groq_valid,
            "pinecone_configured": is_pinecone_valid,
        }


settings = Settings()
settings.ensure_directories()

import os
if settings.GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
if settings.PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
