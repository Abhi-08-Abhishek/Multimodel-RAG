from typing import List
from fastapi import APIRouter, Query

from app.schemas.common import BaseResponse
from app.schemas.telemetry import TelemetrySummary, TelemetryTrace
from app.services.telemetry_service import TelemetryService

router = APIRouter()


@router.get("/telemetry/summary", response_model=BaseResponse[TelemetrySummary])
def get_telemetry_summary():
    """
    Returns cumulative token consumption, estimated USD cost,
    mean latency, and per-model usage breakdown.
    """
    telemetry = TelemetryService()
    summary = telemetry.get_summary()
    return BaseResponse(
        success=True,
        message="Telemetry summary retrieved",
        data=summary
    )


@router.get("/telemetry/traces", response_model=BaseResponse[List[TelemetryTrace]])
def get_telemetry_traces(limit: int = Query(default=50, ge=1, le=200)):
    """
    Returns recent execution traces with token counts, costs, and response times.
    """
    telemetry = TelemetryService()
    traces = telemetry.get_traces(limit=limit)
    return BaseResponse(
        success=True,
        message=f"Retrieved {len(traces)} execution traces",
        data=traces
    )


@router.post("/telemetry/reset", response_model=BaseResponse[bool])
def reset_telemetry():
    """
    Resets all cumulative counters and trace buffers.
    """
    telemetry = TelemetryService()
    telemetry.reset()
    return BaseResponse(
        success=True,
        message="Telemetry metrics reset successfully",
        data=True
    )
