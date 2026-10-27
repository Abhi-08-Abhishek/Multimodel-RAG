import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.schemas.chat import ChatResponse
from app.schemas.session import MessageItem, SessionDetail, SessionItem
from app.schemas.telemetry import TelemetrySummary, TelemetryTrace, TokenUsage


class SessionRepository:
    """
    Lightweight, thread-safe SQLite persistence layer for:
    1. Multi-turn conversation sessions.
    2. Message and citation history.
    3. Per-session token and cost telemetry tracing.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SessionRepository, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return

        self.db_path = db_path or (settings.STORAGE_DIR / "sessions.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        self._initialized = True
        logger.info(f"Initialized SessionRepository at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=20.0,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        """Creates tables and indexes if they do not exist."""
        conn = self._get_connection()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    active_document TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    content TEXT NOT NULL,
                    response_payload TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    model TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    latency_seconds REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'success',
                    query_preview TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_session ON session_traces(session_id);")

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_session(
        self,
        title: Optional[str] = None,
        active_document: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> SessionItem:
        """Creates a new chat session."""
        sid = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        now = self._now_iso()
        session_title = title.strip() if title and title.strip() else "New Conversation"

        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO sessions (id, title, active_document, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sid, session_title, active_document, now, now),
            )

        return SessionItem(
            id=sid,
            title=session_title,
            active_document=active_document,
            message_count=0,
            total_tokens=0,
            total_cost_usd=0.0,
            created_at=now,
            updated_at=now,
        )

    def list_sessions(self) -> List[SessionItem]:
        """Lists all conversation sessions ordered by most recent activity."""
        conn = self._get_connection()
        query = """
            SELECT 
                s.id,
                s.title,
                s.active_document,
                s.created_at,
                s.updated_at,
                COUNT(DISTINCT m.id) AS msg_count,
                COALESCE(SUM(t.total_tokens), 0) AS total_tokens,
                COALESCE(SUM(t.cost_usd), 0.0) AS total_cost_usd
            FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            LEFT JOIN session_traces t ON s.id = t.session_id
            GROUP BY s.id
            ORDER BY s.updated_at DESC;
        """
        rows = conn.execute(query).fetchall()
        items = []
        for r in rows:
            items.append(
                SessionItem(
                    id=r["id"],
                    title=r["title"],
                    active_document=r["active_document"],
                    message_count=r["msg_count"],
                    total_tokens=r["total_tokens"],
                    total_cost_usd=round(r["total_cost_usd"], 5),
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
            )
        return items

    def get_session(self, session_id: str) -> Optional[SessionDetail]:
        """Retrieves session details, ordered messages, and session telemetry summary."""
        conn = self._get_connection()
        s_row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not s_row:
            return None

        # Fetch messages
        m_rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        ).fetchall()

        messages: List[MessageItem] = []
        for mr in m_rows:
            resp_obj = None
            if mr["response_payload"]:
                try:
                    resp_dict = json.loads(mr["response_payload"])
                    resp_obj = ChatResponse(**resp_dict)
                except Exception as parse_err:
                    logger.debug(f"Could not parse stored response_payload: {parse_err}")

            messages.append(
                MessageItem(
                    id=mr["id"],
                    session_id=mr["session_id"],
                    sender=mr["sender"],
                    content=mr["content"],
                    response=resp_obj,
                    created_at=mr["created_at"],
                )
            )

        telemetry = self.get_session_telemetry_summary(session_id)

        return SessionDetail(
            id=s_row["id"],
            title=s_row["title"],
            active_document=s_row["active_document"],
            message_count=len(messages),
            total_tokens=telemetry.total_tokens,
            total_cost_usd=telemetry.total_cost_usd,
            created_at=s_row["created_at"],
            updated_at=s_row["updated_at"],
            messages=messages,
            telemetry=telemetry,
        )

    def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        active_document: Optional[str] = None,
    ) -> bool:
        """Updates session title or bound document."""
        conn = self._get_connection()
        fields = []
        params = []
        if title is not None:
            fields.append("title = ?")
            params.append(title.strip())
        if active_document is not None:
            fields.append("active_document = ?")
            params.append(active_document)

        if not fields:
            return False

        fields.append("updated_at = ?")
        params.append(self._now_iso())
        params.append(session_id)

        with conn:
            cursor = conn.execute(
                f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?",
                tuple(params)
            )
            return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """Deletes session and its cascading child records."""
        conn = self._get_connection()
        with conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    def add_message(
        self,
        session_id: str,
        sender: str,
        content: str,
        response_payload: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
    ) -> MessageItem:
        """Appends a new message to the session."""
        mid = message_id or f"msg_{uuid.uuid4().hex[:8]}"
        now = self._now_iso()
        payload_json = json.dumps(response_payload) if response_payload else None

        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO messages (id, session_id, sender, content, response_payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (mid, session_id, sender, content, payload_json, now)
            )
            # Update parent session's updated_at
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id)
            )

        resp_obj = ChatResponse(**response_payload) if response_payload else None
        return MessageItem(
            id=mid,
            session_id=session_id,
            sender=sender,
            content=content,
            response=resp_obj,
            created_at=now,
        )

    def record_session_trace(
        self,
        session_id: Optional[str],
        trace: TelemetryTrace,
    ) -> None:
        """Associates an execution trace with a session."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_traces (
                    trace_id, session_id, timestamp, operation, model, modality,
                    prompt_tokens, completion_tokens, total_tokens, cost_usd,
                    latency_seconds, status, query_preview
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    session_id,
                    trace.timestamp,
                    trace.operation,
                    trace.model,
                    trace.modality,
                    trace.token_usage.prompt_tokens,
                    trace.token_usage.completion_tokens,
                    trace.token_usage.total_tokens,
                    trace.token_usage.estimated_cost_usd,
                    trace.token_usage.latency_seconds,
                    trace.status,
                    trace.query_preview,
                )
            )

    def get_session_telemetry_summary(self, session_id: str) -> TelemetrySummary:
        """Calculates aggregated token and cost metrics for a single session."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT 
                model,
                COUNT(*) as calls,
                SUM(prompt_tokens) as p_tokens,
                SUM(completion_tokens) as c_tokens,
                SUM(total_tokens) as tot_tokens,
                SUM(cost_usd) as tot_cost,
                AVG(latency_seconds) as avg_lat
            FROM session_traces
            WHERE session_id = ?
            GROUP BY model
            """,
            (session_id,)
        ).fetchall()

        total_requests = 0
        total_p = 0
        total_c = 0
        total_tok = 0
        total_cost = 0.0
        total_lat_sum = 0.0
        breakdown: Dict[str, Dict[str, Any]] = {}

        for r in rows:
            calls = r["calls"] or 0
            p_tok = r["p_tokens"] or 0
            c_tok = r["c_tokens"] or 0
            t_tok = r["tot_tokens"] or 0
            c_cost = r["tot_cost"] or 0.0
            avg_l = round(r["avg_lat"] or 0.0, 3)

            total_requests += calls
            total_p += p_tok
            total_c += c_tok
            total_tok += t_tok
            total_cost += c_cost
            total_lat_sum += avg_l * calls

            breakdown[r["model"]] = {
                "calls": calls,
                "total_tokens": t_tok,
                "cost_usd": round(c_cost, 5),
                "avg_latency": avg_l,
            }

        avg_latency = round(total_lat_sum / total_requests, 3) if total_requests > 0 else 0.0

        return TelemetrySummary(
            total_requests=total_requests,
            total_prompt_tokens=total_p,
            total_completion_tokens=total_c,
            total_tokens=total_tok,
            total_cost_usd=round(total_cost, 5),
            average_latency_seconds=avg_latency,
            model_breakdown=breakdown,
        )

    def get_session_traces(self, session_id: str, limit: int = 20) -> List[TelemetryTrace]:
        """Fetches recent execution traces scoped to a specific session."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT * FROM session_traces
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (session_id, limit)
        ).fetchall()

        traces = []
        for r in rows:
            traces.append(
                TelemetryTrace(
                    trace_id=r["trace_id"],
                    timestamp=r["timestamp"],
                    operation=r["operation"],
                    model=r["model"],
                    modality=r["modality"],
                    token_usage=TokenUsage(
                        prompt_tokens=r["prompt_tokens"],
                        completion_tokens=r["completion_tokens"],
                        total_tokens=r["total_tokens"],
                        estimated_cost_usd=r["cost_usd"],
                        latency_seconds=r["latency_seconds"],
                        model=r["model"],
                    ),
                    status=r["status"],
                    query_preview=r["query_preview"],
                )
            )
        return traces
