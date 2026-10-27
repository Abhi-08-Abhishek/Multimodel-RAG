import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional
from datetime import datetime

from app.core.logging import logger
from app.schemas.telemetry import TelemetrySummary, TelemetryTrace, TokenUsage


class TelemetryService:
    """
    Centralized, thread-safe telemetry and cost tracker for all LLM calls.
    Maintains real-time counters, per-model consumption, and recent execution traces.
    """

    _instance: Optional["TelemetryService"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TelemetryService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, max_traces: int = 200):
        if self._initialized:
            return
        self._max_traces = max_traces
        self._traces: deque = deque(maxlen=max_traces)
        self._total_requests: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._latencies: List[float] = []
        self._model_breakdown: Dict[str, Dict[str, Any]] = {}
        self._service_lock = threading.Lock()
        self._initialized = True
        logger.info("Initialized TelemetryService.")

    def record_trace(
        self,
        operation: str,
        model: str,
        modality: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float,
        latency_seconds: float,
        query_preview: Optional[str] = None,
        status: str = "success",
        session_id: Optional[str] = None,
    ) -> TelemetryTrace:
        """Records an individual execution trace and updates aggregate counters."""
        with self._service_lock:
            self._total_requests += 1
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._total_tokens += total_tokens
            self._total_cost_usd += cost_usd
            self._latencies.append(latency_seconds)

            # Update per-model stats
            if model not in self._model_breakdown:
                self._model_breakdown[model] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "total_latency": 0.0,
                }
            m_stat = self._model_breakdown[model]
            m_stat["calls"] += 1
            m_stat["prompt_tokens"] += prompt_tokens
            m_stat["completion_tokens"] += completion_tokens
            m_stat["total_tokens"] += total_tokens
            m_stat["cost_usd"] = round(m_stat["cost_usd"] + cost_usd, 6)
            m_stat["total_latency"] += latency_seconds

            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=round(cost_usd, 6),
                latency_seconds=round(latency_seconds, 3),
                model=model,
            )

            trace = TelemetryTrace(
                trace_id=f"tr-{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                operation=operation,
                model=model,
                modality=modality,
                token_usage=usage,
                status=status,
                query_preview=query_preview[:120] if query_preview else None,
            )
            self._traces.appendleft(trace)

            if session_id:
                try:
                    from app.db.session_repo import SessionRepository
                    SessionRepository().record_session_trace(session_id, trace)
                except Exception as sess_err:
                    logger.debug(f"Could not link trace to session {session_id}: {sess_err}")

            logger.info(
                f"[Telemetry] {operation} ({model} | {modality}): "
                f"{total_tokens} tokens (${cost_usd:.6f}) in {latency_seconds:.2f}s"
            )
            return trace

    def get_summary(self) -> TelemetrySummary:
        """Returns the high-level aggregated telemetry summary."""
        with self._service_lock:
            avg_lat = (
                sum(self._latencies) / len(self._latencies)
                if self._latencies
                else 0.0
            )

            # Format model breakdown averages
            formatted_breakdown = {}
            for m_name, m_data in self._model_breakdown.items():
                calls = m_data["calls"]
                formatted_breakdown[m_name] = {
                    "calls": calls,
                    "total_tokens": m_data["total_tokens"],
                    "cost_usd": round(m_data["cost_usd"], 6),
                    "avg_latency": round(m_data["total_latency"] / calls, 3) if calls > 0 else 0.0,
                }

            return TelemetrySummary(
                total_requests=self._total_requests,
                total_prompt_tokens=self._total_prompt_tokens,
                total_completion_tokens=self._total_completion_tokens,
                total_tokens=self._total_tokens,
                total_cost_usd=round(self._total_cost_usd, 6),
                average_latency_seconds=round(avg_lat, 3),
                model_breakdown=formatted_breakdown,
            )

    def get_traces(self, limit: int = 50) -> List[TelemetryTrace]:
        """Returns the recent execution traces."""
        with self._service_lock:
            return list(self._traces)[:limit]

    def reset(self) -> None:
        """Clears all telemetry counters and traces."""
        with self._service_lock:
            self._traces.clear()
            self._total_requests = 0
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._total_tokens = 0
            self._total_cost_usd = 0.0
            self._latencies.clear()
            self._model_breakdown.clear()
            logger.info("Telemetry counters reset.")
