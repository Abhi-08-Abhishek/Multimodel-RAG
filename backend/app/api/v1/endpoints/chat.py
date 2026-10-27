from fastapi import APIRouter
from app.db.session_repo import SessionRepository
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.common import BaseResponse
from app.services.rag_orchestrator import RAGOrchestrator

router = APIRouter()
repo = SessionRepository()


@router.post("/chat", response_model=BaseResponse[ChatResponse])
def ask_question(request: ChatRequest):
    """
    Answers questions using Multimodal RAG with automatic session tracking:
    - Resolves or creates conversation session in SQLite.
    - Persists user query message.
    - Queries Pinecone serverless vector index and LiteLLM models.
    - Attaches session_id to telemetry traces.
    - Persists assistant response and citation payload.
    - Returns answer with page citations, public image URLs, and session ID.
    """
    session_id = request.session_id
    if not session_id or not repo.get_session(session_id):
        # Auto-create session titled after query snippet
        clean_title = request.query.strip().split("\n")[0][:45]
        if len(request.query) > 45:
            clean_title += "..."
        new_session = repo.create_session(title=clean_title)
        session_id = new_session.id

    # 1. Record incoming user message
    repo.add_message(
        session_id=session_id,
        sender="user",
        content=request.query,
    )

    # 2. Execute Multimodal RAG synthesis with session_id
    orchestrator = RAGOrchestrator()
    response = orchestrator.ask(
        query=request.query,
        namespace=request.namespace,
        top_k=request.top_k,
        session_id=session_id,
    )

    # 3. Record assistant response with full citations payload
    repo.add_message(
        session_id=session_id,
        sender="assistant",
        content=response.answer,
        response_payload=response.model_dump(),
    )

    return BaseResponse(
        success=True,
        message="Query processed successfully",
        data=response
    )
