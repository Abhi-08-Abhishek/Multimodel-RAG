import base64
import io
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image

from app.core.config import settings
from app.core.exceptions import VisionProcessingError
from app.core.logging import logger
from app.schemas.telemetry import TokenUsage
from app.services.llm_gateway import LLMGateway, ModelRole


class VisionService:
    """
    Manages multimodal visual interactions via unified LiteLLM gateway:
    1. Image to Base64 Data URI conversion with downsampled token optimization.
    2. Deep visual summarization of charts, diagrams, and business graphics during ingestion.
    3. Multimodal question-answering combining text context, user query, and actual visual evidence.
    """

    def __init__(self, gateway: Optional[LLMGateway] = None):
        self.gateway = gateway or LLMGateway()

    @staticmethod
    def image_to_data_uri(image_path: Path, max_dimension: int = 512) -> str:
        """
        Loads an image, converts to RGB, resizes proportionally to max_dimension,
        and returns a standard base64 data URI string with optimized token weight.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise VisionProcessingError(
                message=f"Image file does not exist at {image_path}",
                details={"image_path": str(image_path)}
            )

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.thumbnail((max_dimension, max_dimension))

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            raise VisionProcessingError(
                message=f"Failed to encode image to data URI: {str(e)}",
                details={"image_path": str(image_path), "error": str(e)}
            )

    def summarize_visual(self, image_path: str, page_number: int, doc_name: str) -> str:
        """
        Invokes LiteLLM Vision model to generate a structured, factual text description
        of an extracted chart, graph, diagram, or business visual.
        """
        # Check persistent on-disk cache for previously summarized visual
        cache_path = Path(image_path).with_suffix(".summary.txt")
        if cache_path.exists():
            cached_text = cache_path.read_text(encoding="utf-8").strip()
            if cached_text:
                logger.info(f"Loaded cached visual summary for {Path(image_path).name}")
                return cached_text

        data_uri = self.image_to_data_uri(Path(image_path))

        prompt = f"""
This visual was extracted from page {page_number} of the enterprise document '{doc_name}'.

Describe the useful business, technical, and quantitative information visible in the visual:

If it is a chart or graph (line graph, bar chart, pie chart):
- Identify the chart title, metric measured, and axes.
- Mention specific values, data points, or percentages visible.
- Highlight maximum, minimum, and noticeable trends.

If it is a diagram or flow chart:
- Identify key components, labels, and workflow stages.
- Explain relationships and any critical control points or highlights.

If it is a facility or business photo:
- Detail factual elements, equipment, infrastructure, or operational indicators.

Keep the description concise, factual, and data-dense so it can be indexed for semantic retrieval.
"""

        try:
            logger.info(f"Summarizing visual from page {page_number}: {Path(image_path).name}")
            summary, _ = self.gateway.complete(
                role=ModelRole.SUMMARY,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt.strip()},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_uri},
                            },
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=350,
                operation="visual_summarization",
                query_preview=f"{Path(image_path).name} (p.{page_number})",
            )

            try:
                cache_path.write_text(summary, encoding="utf-8")
            except Exception as cache_err:
                logger.warning(f"Could not write cache file {cache_path}: {cache_err}")

            return summary

        except Exception as e:
            logger.error(f"LiteLLM Vision error summarizing {image_path}: {e}")
            raise VisionProcessingError(
                message=f"Failed to summarize visual with Vision LLM: {str(e)}",
                details={"image_path": str(image_path), "error": str(e)}
            )

    def answer_with_vision(
        self,
        question: str,
        context: str,
        image_paths: List[str],
        session_id: Optional[str] = None,
    ) -> Tuple[str, TokenUsage]:
        """
        Performs multimodal reasoning by attaching the actual extracted visual images
        alongside the retrieved context and user query. Returns (answer, token_usage).
        """
        valid_paths = [Path(p) for p in image_paths if Path(p).exists()][:2]

        system_prompt = f"""
You are an expert enterprise business intelligence analyst answering questions about company reports and documents.

Grounding Rules:
- Rely strictly on the retrieved context and attached visual assets.
- If data is visible in the attached visuals (charts, graphs, diagrams), extract exact numbers, labels, and trends.
- Cite the relevant page numbers whenever making factual claims.
- If the required information is not supported by the context or visuals, state clearly: "I could not find that information in the provided document."

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}
"""

        user_content = [{"type": "text", "text": system_prompt.strip()}]

        for img_path in valid_paths:
            try:
                uri = self.image_to_data_uri(img_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": uri}
                })
            except Exception as e:
                logger.warning(f"Could not append image {img_path} to vision answer payload: {e}")

        try:
            logger.info(f"Invoking LiteLLM Vision QA with {len(valid_paths)} visual assets")
            answer, token_usage = self.gateway.complete(
                role=ModelRole.VISION,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.1,
                max_tokens=450,
                operation="rag_visual_qa",
                query_preview=question,
                session_id=session_id,
            )
            return answer, token_usage

        except Exception as e:
            logger.error(f"LiteLLM Vision QA failed: {e}")
            raise VisionProcessingError(
                message=f"Multimodal vision generation failed: {str(e)}",
                details={"error": str(e)}
            )
