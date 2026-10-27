import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import litellm

from app.core.config import settings
from app.core.exceptions import ConfigurationError, VisionProcessingError
from app.core.logging import logger
from app.schemas.telemetry import TokenUsage
from app.services.telemetry_service import TelemetryService


class ModelRole(str, Enum):
    TEXT = "text"
    VISION = "vision"
    SUMMARY = "summary"


class LLMGateway:
    """
    Unified LiteLLM model router and execution gateway.
    Handles dynamic model selection, multimodal payloads, real-time cost calculation,
    and automatic telemetry dispatching.
    """

    def __init__(self, telemetry_service: Optional[TelemetryService] = None):
        self.telemetry = telemetry_service or TelemetryService()
        # Suppress noisy LiteLLM default debug outputs
        litellm.suppress_debug_info = True

    def resolve_model(self, role: ModelRole, custom_model: Optional[str] = None) -> Tuple[str, str]:
        """
        Dynamically maps logical role to LiteLLM-compatible provider string and api key.
        Returns (litellm_model_string, api_key).
        """
        if custom_model:
            raw_model = custom_model
        elif role == ModelRole.TEXT:
            raw_model = settings.TEXT_MODEL
        elif role in (ModelRole.VISION, ModelRole.SUMMARY):
            raw_model = settings.VISION_MODEL
        else:
            raw_model = settings.TEXT_MODEL

        # Normalize Groq provider prefix for LiteLLM
        # Note: Groq hosts models named openai/gpt-oss-20b and qwen/qwen3.8-27b, so ensure groq/ prefix
        if not raw_model.startswith("groq/") and not raw_model.startswith("anthropic/") and not raw_model.startswith("ollama/"):
            litellm_model = f"groq/{raw_model}"
        else:
            litellm_model = raw_model

        api_key = settings.GROQ_API_KEY
        if not api_key or "your_groq" in api_key.lower():
            raise ConfigurationError(
                message="Groq API key is not configured for LLM execution.",
                details={"field": "GROQ_API_KEY"}
            )

        return litellm_model, api_key

    def complete(
        self,
        role: ModelRole,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        operation: str = "llm_completion",
        query_preview: Optional[str] = None,
        custom_model: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Tuple[str, TokenUsage]:
        """
        Executes a completion through LiteLLM, calculates tokens and USD costs,
        and registers the execution with TelemetryService.
        """
        model_name, api_key = self.resolve_model(role, custom_model)
        start_time = time.time()
        modality = "visual" if role in (ModelRole.VISION, ModelRole.SUMMARY) else "text"

        logger.info(f"[LLMGateway] Invoking {model_name} for role '{role.value}' (op={operation})")

        try:
            kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "api_key": api_key,
                "num_retries": 1,
                "timeout": 45,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = litellm.completion(**kwargs)

            latency = round(time.time() - start_time, 3)
            content = response.choices[0].message.content or ""

            # Extract token usage
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

            # Calculate cost via LiteLLM cost calculator
            try:
                cost_usd = litellm.completion_cost(completion_response=response)
            except Exception as cost_err:
                logger.debug(f"Cost calculation fallback: {cost_err}")
                cost_usd = 0.0

            # Dispatch trace to TelemetryService
            trace = self.telemetry.record_trace(
                operation=operation,
                model=model_name,
                modality=modality,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                latency_seconds=latency,
                query_preview=query_preview,
                status="success",
                session_id=session_id,
            )

            return content.strip(), trace.token_usage

        except Exception as e:
            latency = round(time.time() - start_time, 3)
            logger.error(f"[LLMGateway] LiteLLM execution failed on {model_name}: {e}")

            # Record failure in telemetry
            self.telemetry.record_trace(
                operation=operation,
                model=model_name,
                modality=modality,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                latency_seconds=latency,
                query_preview=query_preview,
                status=f"error: {str(e)[:100]}",
                session_id=session_id,
            )
            raise e
