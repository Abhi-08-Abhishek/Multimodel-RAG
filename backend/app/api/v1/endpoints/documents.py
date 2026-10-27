import os
import shutil
import time
from pathlib import Path
from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.logging import logger
from app.schemas.common import BaseResponse
from app.schemas.document import DocumentItem

router = APIRouter()


@router.get("/documents", response_model=BaseResponse[List[DocumentItem]])
async def list_documents():
    """Lists all stored PDF documents and extracted image counts."""
    docs: List[DocumentItem] = []
    uploads_dir = settings.UPLOADS_DIR

    if uploads_dir.exists():
        for file_path in uploads_dir.glob("*.pdf"):
            stats = file_path.stat()
            modified_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(stats.st_mtime)
            )

            # Count only actual extracted image files (png/jpg)
            stem = file_path.stem.replace(" ", "_")
            image_files = [
                img for img in settings.EXTRACTED_IMAGES_DIR.glob(f"{stem}_*")
                if img.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
            ]
            image_count = len(image_files)

            docs.append(
                DocumentItem(
                    filename=file_path.name,
                    size_bytes=stats.st_size,
                    modified_at=modified_time,
                    is_indexed=image_count > 0,
                    image_count=image_count,
                )
            )

    return BaseResponse(
        success=True,
        message=f"Found {len(docs)} documents",
        data=docs
    )


@router.post("/documents/upload", response_model=BaseResponse[DocumentItem])
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF file into storage/uploads."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported."
        )

    destination = settings.UPLOADS_DIR / file.filename
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file: {str(e)}"
        )

    file_stat = destination.stat()
    return BaseResponse(
        success=True,
        message=f"Uploaded {file.filename} successfully",
        data=DocumentItem(
            filename=destination.name,
            size_bytes=file_stat.st_size,
            modified_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_stat.st_mtime)),
            is_indexed=False,
            image_count=0,
        )
    )


@router.delete("/documents/{filename}", response_model=BaseResponse[bool])
async def delete_document(filename: str):
    """Deletes an uploaded document and its associated extracted images and caches."""
    file_path = settings.UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found."
        )

    try:
        file_path.unlink()

        # Clean up any associated extracted images and summaries
        stem = file_path.stem.replace(" ", "_")
        for asset in settings.EXTRACTED_IMAGES_DIR.glob(f"{stem}_*"):
            try:
                asset.unlink()
            except Exception as e:
                logger.warning(f"Could not delete asset {asset}: {e}")

        logger.info(f"Deleted document '{filename}' and associated assets.")
        return BaseResponse(
            success=True,
            message=f"Document '{filename}' deleted successfully.",
            data=True
        )
    except Exception as e:
        logger.error(f"Failed to delete {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )
