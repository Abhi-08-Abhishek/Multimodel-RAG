import io
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pymupdf
import pandas as pd
from PIL import Image

from app.core.config import settings
from app.core.exceptions import ExtractionError
from app.core.logging import logger
from app.schemas.common import ModalityEnum


class RawExtractedItem:
    """Represents a piece of extracted content before vectorization."""
    def __init__(
        self,
        content: str,
        page: int,
        modality: ModalityEnum,
        source: str,
        image_path: str = None,
        table_number: int = None,
    ):
        self.content = content
        self.page = page
        self.modality = modality
        self.source = source
        self.image_path = image_path
        self.table_number = table_number


class MultimodalPDFExtractor:
    """
    Extracts text, structured markdown tables, and visual assets (charts, diagrams, images)
    from PDF documents using PyMuPDF and Pandas.
    """

    def __init__(self, output_image_dir: Path = None):
        self.output_image_dir = output_image_dir or settings.EXTRACTED_IMAGES_DIR
        self.output_image_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, pdf_path: Path) -> List[RawExtractedItem]:
        """
        Processes all pages in the PDF and extracts multimodal content.
        """
        if not pdf_path.exists():
            raise ExtractionError(
                message=f"PDF document not found at: {pdf_path}",
                details={"path": str(pdf_path)}
            )

        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception as e:
            raise ExtractionError(
                message=f"Failed to open PDF document: {str(e)}",
                details={"path": str(pdf_path), "error": str(e)}
            )

        extracted_items: List[RawExtractedItem] = []
        extracted_xrefs: Set[int] = set()

        logger.info(f"Beginning multimodal extraction for {pdf_path.name} ({len(doc)} pages)")

        try:
            for page_index in range(len(doc)):
                page_number = page_index + 1
                page = doc[page_index]

                # -------------------------------------------------------------
                # 1. TEXT EXTRACTION
                # -------------------------------------------------------------
                try:
                    text = page.get_text("text").strip()
                    if text:
                        extracted_items.append(
                            RawExtractedItem(
                                content=text,
                                page=page_number,
                                modality=ModalityEnum.TEXT,
                                source=pdf_path.name,
                            )
                        )
                except Exception as text_err:
                    logger.warning(f"Error extracting text from page {page_number}: {text_err}")

                # -------------------------------------------------------------
                # 2. TABLE EXTRACTION
                # -------------------------------------------------------------
                try:
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        for table_index, table in enumerate(tabs.tables, start=1):
                            df = table.to_pandas()
                            if df is not None and not df.empty:
                                markdown_table = df.to_markdown(index=False)
                                extracted_items.append(
                                    RawExtractedItem(
                                        content=markdown_table,
                                        page=page_number,
                                        modality=ModalityEnum.TABLE,
                                        source=pdf_path.name,
                                        table_number=table_index,
                                    )
                                )
                except Exception as table_err:
                    logger.warning(f"Table extraction notice on page {page_number}: {table_err}")

                # -------------------------------------------------------------
                # 3. VISUAL ASSETS (CHARTS, DIAGRAMS, IMAGES)
                # -------------------------------------------------------------
                try:
                    page_images = page.get_images(full=True)
                    for image_index, img_info in enumerate(page_images, start=1):
                        xref = img_info[0]
                        if xref in extracted_xrefs:
                            continue
                        extracted_xrefs.add(xref)

                        img_dict = doc.extract_image(xref)
                        img_bytes = img_dict.get("image")
                        img_ext = img_dict.get("ext", "png")

                        if not img_bytes or len(img_bytes) < 1024:
                            # Skip micro-icons or empty artifacts (< 1KB)
                            continue

                        # Clean filename
                        stem = pdf_path.stem.replace(" ", "_")
                        saved_filename = f"{stem}_p{page_number}_img{image_index}.{img_ext}"
                        saved_path = self.output_image_dir / saved_filename

                        saved_path.write_bytes(img_bytes)

                        # Validate and normalize using PIL
                        try:
                            with Image.open(saved_path) as pil_img:
                                # Ensure valid image format
                                pil_img.verify()
                        except Exception as pil_err:
                            logger.warning(f"Skipping corrupt extracted image {saved_filename}: {pil_err}")
                            if saved_path.exists():
                                saved_path.unlink()
                            continue

                        extracted_items.append(
                            RawExtractedItem(
                                content="",  # To be populated by VisionService summary
                                page=page_number,
                                modality=ModalityEnum.VISUAL,
                                source=pdf_path.name,
                                image_path=str(saved_path),
                            )
                        )

                except Exception as img_err:
                    logger.warning(f"Image extraction notice on page {page_number}: {img_err}")

        finally:
            doc.close()

        logger.info(
            f"Extraction complete for {pdf_path.name}: "
            f"{len([i for i in extracted_items if i.modality == ModalityEnum.TEXT])} text, "
            f"{len([i for i in extracted_items if i.modality == ModalityEnum.TABLE])} tables, "
            f"{len([i for i in extracted_items if i.modality == ModalityEnum.VISUAL])} visuals"
        )
        return extracted_items
