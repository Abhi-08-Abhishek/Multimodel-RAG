import time
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, status
from langchain_core.documents import Document

from app.core.config import settings
from app.core.exceptions import ExtractionError
from app.core.logging import logger
from app.schemas.common import BaseResponse, ModalityEnum
from app.schemas.document import ExtractedStats, IngestRequest, IngestResponse
from app.services.pdf_extractor import MultimodalPDFExtractor
from app.services.vector_store import VectorStoreService
from app.services.vision_service import VisionService

router = APIRouter()


@router.post("/ingest", response_model=BaseResponse[IngestResponse])
def ingest_document(request: IngestRequest):
    """
    Executes the multimodal ingestion pipeline:
    1. Extracts narrative text, markdown tables, and visual assets from the target PDF.
    2. Uses Groq Vision to generate descriptive business summaries for all visuals.
    3. Transforms items into LangChain Document records with metadata.
    4. Optionally clears the target namespace.
    5. Embeds content and indexes into Pinecone serverless vector store.
    """
    start_time = time.time()
    target_doc = request.document_name.strip()
    if not target_doc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid document_name is required for ingestion."
        )

    pdf_path = settings.UPLOADS_DIR / target_doc
    if not pdf_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{target_doc}' not found in uploads directory. Please upload it first."
        )

    namespace = request.namespace or settings.PINECONE_NAMESPACE

    logger.info(f"Starting ingestion for {target_doc} into namespace '{namespace}'")

    # Step 1: Multimodal Extraction
    extractor = MultimodalPDFExtractor()
    extracted_items = extractor.extract(pdf_path)

    # Step 2: Vision Summarization
    vision_service = VisionService()
    documents: List[Document] = []

    text_count = 0
    table_count = 0
    visual_count = 0
    pages_seen = set()

    for item in extracted_items:
        pages_seen.add(item.page)
        metadata = {
            "page": item.page,
            "modality": item.modality.value,
            "source": item.source,
        }

        if item.modality == ModalityEnum.TEXT:
            text_count += 1
            documents.append(
                Document(
                    page_content=item.content,
                    metadata=metadata
                )
            )

        elif item.modality == ModalityEnum.TABLE:
            table_count += 1
            metadata["table_number"] = item.table_number
            documents.append(
                Document(
                    page_content=item.content,
                    metadata=metadata
                )
            )

        elif item.modality == ModalityEnum.VISUAL:
            visual_count += 1
            metadata["image_path"] = item.image_path

            # Summarize image with Groq Vision
            try:
                summary = vision_service.summarize_visual(
                    image_path=item.image_path,
                    page_number=item.page,
                    doc_name=item.source,
                )
                documents.append(
                    Document(
                        page_content=summary,
                        metadata=metadata
                    )
                )
                time.sleep(1.5)  # Smooth out token distribution for Vision API
            except Exception as e:
                logger.error(f"Failed to summarize visual on page {item.page}: {e}")
                # Still index item with basic fallback description if vision API throttles
                documents.append(
                    Document(
                        page_content=f"Visual asset extracted from page {item.page} of {item.source}.",
                        metadata=metadata
                    )
                )

    # Step 3: Vector Storage
    vector_store = VectorStoreService()

    if request.clean_namespace_first:
        logger.info(f"Clearing namespace '{namespace}' prior to fresh indexing")
        vector_store.clean_namespace(namespace)

    # Index into Pinecone
    vector_store.add_documents(documents, namespace=namespace)

    duration = round(time.time() - start_time, 2)
    stats = ExtractedStats(
        total_pages=len(pages_seen),
        text_chunks=text_count,
        tables_extracted=table_count,
        visuals_extracted=visual_count,
        total_documents=len(documents),
    )

    logger.info(f"Ingestion finished in {duration}s: {stats}")

    return BaseResponse(
        success=True,
        message=f"Successfully indexed {len(documents)} multimodal documents",
        data=IngestResponse(
            document_name=target_doc,
            namespace=namespace,
            stats=stats,
            duration_seconds=duration,
        )
    )
