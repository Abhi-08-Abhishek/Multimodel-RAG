import time
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document

from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.core.logging import logger
from app.schemas.chat import ChatResponse, CitationSource
from app.schemas.common import ModalityEnum
from app.schemas.telemetry import TokenUsage
from app.services.llm_gateway import LLMGateway, ModelRole
from app.services.vector_store import VectorStoreService
from app.services.vision_service import VisionService


class RAGOrchestrator:
    """
    Coordinates the Multimodal RAG pipeline:
    1. Pinecone retrieval based on dense semantic query embeddings.
    2. Modality classification and citation assembly.
    3. Unified LiteLLM Gateway routing (Vision vs Text).
    4. Real-time token usage and cost recording.
    5. Formatting of rich citation objects and public image URLs.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        vision_service: Optional[VisionService] = None,
        llm_gateway: Optional[LLMGateway] = None,
    ):
        self.vector_store = vector_store or VectorStoreService()
        self.gateway = llm_gateway or LLMGateway()
        self.vision_service = vision_service or VisionService(gateway=self.gateway)

    @staticmethod
    def _format_context(retrieved_docs: List[Document]) -> str:
        """Assembles structured context with clean integer page numbers and modality tags."""
        parts = []
        for doc in retrieved_docs:
            raw_page = doc.metadata.get("page", 1)
            try:
                page_num = int(float(raw_page))
            except Exception:
                page_num = raw_page
            modality = doc.metadata.get("modality", "text").upper()
            content = doc.page_content.strip()
            parts.append(f"[Page {page_num} | {modality}]\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _to_public_image_url(local_image_path: str) -> str:
        """Converts local file path to public FastAPI static endpoint URL."""
        if not local_image_path:
            return ""
        filename = Path(local_image_path).name
        return f"/static/extracted_images/{filename}"

    def ask(
        self,
        query: str,
        namespace: Optional[str] = None,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        Executes end-to-end multimodal retrieval, dynamic routing, and synthesis.
        """
        start_time = time.time()
        ns = namespace or settings.PINECONE_NAMESPACE

        logger.info(f"RAG query received: '{query}' (ns={ns}, top_k={top_k})")

        # 1. Retrieve from Pinecone
        try:
            results_with_scores = self.vector_store.similarity_search(
                query=query,
                k=top_k,
                namespace=ns,
            )
        except Exception as e:
            raise RetrievalError(
                message=f"Failed to query vector database: {str(e)}",
                details={"query": query, "namespace": ns}
            )

        if not results_with_scores:
            logger.info("No documents retrieved for query.")
            return ChatResponse(
                query=query,
                answer="No relevant information was found in the indexed documents.",
                sources=[],
                has_visuals=False,
                visual_images=[],
                model_used="none",
                latency_seconds=round(time.time() - start_time, 2),
                token_usage=None,
            )

        retrieved_docs = [doc for doc, score in results_with_scores]
        formatted_context = self._format_context(retrieved_docs)

        # 2. Inspect for visual evidence and assemble citations
        image_paths: List[str] = []
        citations: List[CitationSource] = []

        for i, (doc, score) in enumerate(results_with_scores, start=1):
            modality = doc.metadata.get("modality", ModalityEnum.TEXT.value)
            img_path = doc.metadata.get("image_path")
            public_url = self._to_public_image_url(img_path) if img_path else None

            if modality == ModalityEnum.VISUAL.value and img_path:
                if Path(img_path).exists():
                    image_paths.append(img_path)

            page_val = 1
            try:
                page_val = int(float(doc.metadata.get("page", 1)))
            except Exception:
                page_val = 1

            citations.append(
                CitationSource(
                    id=doc.metadata.get("id", f"chunk-{i}"),
                    page=page_val,
                    modality=ModalityEnum(modality),
                    content_preview=doc.page_content[:180].replace("\n", " ") + "...",
                    full_content=doc.page_content,
                    image_url=public_url,
                    table_number=doc.metadata.get("table_number"),
                    source_document=doc.metadata.get("source", ""),
                    score=round(float(score), 4) if score is not None else None,
                )
            )

        # 3. Dynamic Model Routing through unified LiteLLM gateway
        has_visuals = len(image_paths) > 0
        token_usage: Optional[TokenUsage] = None

        if has_visuals:
            logger.info(f"Routing to LiteLLM Vision model with {len(image_paths)} visual assets.")
            try:
                answer, token_usage = self.vision_service.answer_with_vision(
                    question=query,
                    context=formatted_context,
                    image_paths=image_paths,
                    session_id=session_id,
                )
                model_used = token_usage.model if token_usage else settings.VISION_MODEL
            except Exception as vision_err:
                logger.warning(
                    f"Vision model invocation failed ({vision_err}). Falling back gracefully to text synthesis."
                )
                # Resilient fallback to text model using retrieved context & tables
                system_prompt = """
You are a knowledgeable and precise enterprise business analyst answering questions about the provided enterprise reports.

Strict Grounding Rules:
- Rely strictly on the retrieved document context below (which includes text chunks, extracted tables, and figure descriptions).
- Answer the user's question accurately using all available factual metrics, tables, and notes.
- If the exact visual graph was rate-limited on the OCR service, summarize from the extracted context and indicate page references.
- Always cite relevant page numbers for each finding.
"""
                user_message = f"""
RETRIEVED CONTEXT:
{formatted_context}

QUESTION:
{query}
"""
                answer, token_usage = self.gateway.complete(
                    role=ModelRole.TEXT,
                    messages=[
                        {"role": "system", "content": system_prompt.strip()},
                        {"role": "user", "content": user_message.strip()},
                    ],
                    temperature=0.0,
                    max_tokens=800,
                    operation="rag_text_qa (vision_fallback)",
                    query_preview=query,
                    session_id=session_id,
                )
                model_used = f"{token_usage.model if token_usage else settings.TEXT_MODEL} (grounded fallback)"
        else:
            logger.info("Routing to LiteLLM Text model (no visual chunks retrieved).")
            system_prompt = """
You are a knowledgeable and precise enterprise business analyst answering questions about the provided enterprise reports.

Strict Grounding Rules:
- Rely ONLY on the retrieved context below.
- If the exact answer is not present in the retrieved context, say: "I could not find that information in the provided document."
- Always mention relevant page numbers for each factual point.
- When tables are present, read metrics and percentages accurately.
"""
            user_message = f"""
RETRIEVED CONTEXT:
{formatted_context}

QUESTION:
{query}
"""
            answer, token_usage = self.gateway.complete(
                role=ModelRole.TEXT,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_message.strip()},
                ],
                temperature=0.0,
                max_tokens=800,
                operation="rag_text_qa",
                query_preview=query,
                session_id=session_id,
            )
            model_used = token_usage.model if token_usage else settings.TEXT_MODEL

        latency = round(time.time() - start_time, 2)
        logger.info(f"Query completed in {latency}s via {model_used} | Tokens: {token_usage.total_tokens if token_usage else 0}")

        return ChatResponse(
            query=query,
            answer=answer,
            session_id=session_id,
            sources=citations,
            has_visuals=has_visuals,
            visual_images=[self._to_public_image_url(p) for p in image_paths],
            model_used=model_used,
            latency_seconds=latency,
            token_usage=token_usage,
        )
