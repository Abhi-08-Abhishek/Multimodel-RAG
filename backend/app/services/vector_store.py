import time
from typing import Any, Dict, List, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from app.core.config import settings
from app.core.exceptions import ConfigurationError, VectorStoreError
from app.core.logging import logger
from app.services.embedding_service import EmbeddingService


class VectorStoreService:
    """
    Manages Pinecone serverless vector database lifecycle, namespace isolation,
    document indexing, and similarity searches.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.api_key = api_key or settings.PINECONE_API_KEY
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.cloud = settings.PINECONE_CLOUD
        self.region = settings.PINECONE_REGION
        self.embedding_service = embedding_service or EmbeddingService.get_instance()
        self._pc: Optional[Pinecone] = None

    @property
    def client(self) -> Pinecone:
        if self._pc is None:
            if not self.api_key or "your_pinecone" in self.api_key.lower():
                raise ConfigurationError(
                    message="Pinecone API key is not configured. Please supply PINECONE_API_KEY in .env.",
                    details={"field": "PINECONE_API_KEY"}
                )
            try:
                self._pc = Pinecone(api_key=self.api_key)
            except Exception as e:
                raise VectorStoreError(
                    message=f"Failed to initialize Pinecone client: {str(e)}",
                    details={"error": str(e)}
                )
        return self._pc

    def ensure_index(self) -> None:
        """
        Creates serverless Pinecone index if it does not already exist,
        matching the vector dimension of the active embedding model.
        """
        pc = self.client
        dimension = self.embedding_service.dimension

        try:
            if not pc.has_index(self.index_name):
                logger.info(f"Creating Pinecone serverless index '{self.index_name}' (dim={dimension})...")
                pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region,
                    ),
                )

                # Wait for readiness
                while not pc.describe_index(self.index_name).status["ready"]:
                    logger.info("Waiting for Pinecone index to become ready...")
                    time.sleep(2)

                logger.info(f"Pinecone index '{self.index_name}' is ready.")
            else:
                logger.info(f"Pinecone index '{self.index_name}' already exists.")
        except Exception as e:
            logger.error(f"Error ensuring Pinecone index: {e}")
            raise VectorStoreError(
                message=f"Pinecone index initialization failed: {str(e)}",
                details={"index": self.index_name, "error": str(e)}
            )

    def clean_namespace(self, namespace: Optional[str] = None) -> None:
        """Deletes all vectors inside a specific namespace."""
        ns = namespace or settings.PINECONE_NAMESPACE
        try:
            index = self.client.Index(self.index_name)
            index.delete(delete_all=True, namespace=ns)
            logger.info(f"Cleared namespace: {ns}")
        except Exception as e:
            # 404 or Namespace not found is expected if brand new
            logger.info(f"Namespace '{ns}' was empty or not found: {e}")

    def add_documents(
        self,
        documents: List[Document],
        namespace: Optional[str] = None
    ) -> None:
        """
        Uploads LangChain documents to Pinecone vector store under the namespace.
        """
        self.ensure_index()
        ns = namespace or settings.PINECONE_NAMESPACE

        if not documents:
            logger.warning("No documents provided for Pinecone indexing.")
            return

        try:
            logger.info(f"Upserting {len(documents)} documents to Pinecone (index={self.index_name}, ns={ns})")
            vectorstore = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embedding_service.embeddings,
                namespace=ns,
                pinecone_api_key=self.api_key,
            )
            # Batch upsert
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                vectorstore.add_documents(batch)
            logger.info("Documents successfully indexed into Pinecone.")
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {e}")
            raise VectorStoreError(
                message=f"Failed to upsert documents to Pinecone: {str(e)}",
                details={"index": self.index_name, "namespace": ns, "error": str(e)}
            )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        namespace: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieves top-k most similar documents along with similarity scores.
        """
        self.ensure_index()
        ns = namespace or settings.PINECONE_NAMESPACE

        try:
            vectorstore = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embedding_service.embeddings,
                namespace=ns,
                pinecone_api_key=self.api_key,
            )
            results = vectorstore.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search in Pinecone: {e}")
            raise VectorStoreError(
                message=f"Similarity retrieval failed: {str(e)}",
                details={"query": query, "namespace": ns, "error": str(e)}
            )

    def get_stats(self) -> Dict[str, Any]:
        """Returns vector count and namespace information."""
        try:
            index = self.client.Index(self.index_name)
            stats = index.describe_index_stats()
            return stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        except Exception as e:
            logger.warning(f"Could not retrieve Pinecone stats: {e}")
            return {"status": "unavailable", "error": str(e)}
