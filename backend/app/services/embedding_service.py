from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.core.logging import logger


class EmbeddingService:
    """
    Manages text embeddings using HuggingFace Sentence Transformers.
    Caches model weights in memory and verifies dimension compatibility.
    """

    _instance: Optional["EmbeddingService"] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Initializing embedding model: {self.model_name}")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        # Compute dimension dynamically
        test_vector = self._embeddings.embed_query("dimension check")
        self.dimension = len(test_vector)
        logger.info(f"Embedding model ready. Vector dimension: {self.dimension}")

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = EmbeddingService(model_name)
        return cls._instance

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)
