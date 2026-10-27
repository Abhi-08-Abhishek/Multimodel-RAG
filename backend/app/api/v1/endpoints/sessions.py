from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.db.session_repo import SessionRepository
from app.schemas.common import BaseResponse
from app.schemas.session import SessionCreate, SessionDetail, SessionItem, SessionUpdate
from app.schemas.telemetry import TelemetrySummary, TelemetryTrace

router = APIRouter(prefix="/sessions", tags=["Sessions"])
repo = SessionRepository()


@router.get("", response_model=BaseResponse[List[SessionItem]])
async def list_sessions():
    """Returns all conversation sessions ordered by most recent activity."""
    sessions = repo.list_sessions()
    return BaseResponse(
        success=True,
        message=f"Retrieved {len(sessions)} conversation sessions",
        data=sessions,
    )


@router.post("", response_model=BaseResponse[SessionItem], status_code=status.HTTP_201_CREATED)
async def create_session(request: SessionCreate):
    """Creates a new conversation session."""
    session = repo.create_session(
        title=request.title,
        active_document=request.active_document,
    )
    return BaseResponse(
        success=True,
        message="Conversation session created",
        data=session,
    )


@router.get("/{session_id}", response_model=BaseResponse[SessionDetail])
async def get_session(session_id: str):
    """Retrieves full conversation history and telemetry for a specific session."""
    session = repo.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found",
        )
    return BaseResponse(
        success=True,
        message="Session details retrieved",
        data=session,
    )


@router.patch("/{session_id}", response_model=BaseResponse[bool])
async def update_session(session_id: str, request: SessionUpdate):
    """Updates session title or active document."""
    success = repo.update_session(
        session_id=session_id,
        title=request.title,
        active_document=request.active_document,
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found",
        )
    return BaseResponse(
        success=True,
        message="Session updated successfully",
        data=True,
    )


@router.delete("/{session_id}", response_model=BaseResponse[bool])
async def delete_session(session_id: str):
    """Permanently deletes a conversation session and all associated messages."""
    success = repo.delete_session(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with ID '{session_id}' not found",
        )
    return BaseResponse(
        success=True,
        message="Session deleted successfully",
        data=True,
    )


@router.get("/{session_id}/telemetry", response_model=BaseResponse[TelemetrySummary])
async def get_session_telemetry(session_id: str):
    """Returns token consumption and cost breakdown for a specific session."""
    summary = repo.get_session_telemetry_summary(session_id)
    return BaseResponse(
        success=True,
        message="Session telemetry summary retrieved",
        data=summary,
    )


@router.get("/{session_id}/traces", response_model=BaseResponse[List[TelemetryTrace]])
async def get_session_traces(
    session_id: str,
    limit: int = Query(default=20, ge=1, le=100),
):
    """Returns recent execution traces logged in a specific session."""
    traces = repo.get_session_traces(session_id, limit=limit)
    return BaseResponse(
        success=True,
        message=f"Retrieved {len(traces)} session traces",
        data=traces,
    )
