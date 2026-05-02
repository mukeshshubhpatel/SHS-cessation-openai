import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SessionRecord:
    session_id: str
    user_age: int
    age_tier: int


class MemoryStore:
    """Local SQLite store for session memory, summaries, and telemetry."""

    def __init__(self, db_path: Path):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_age INTEGER NOT NULL,
                    age_tier INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    session_id TEXT PRIMARY KEY,
                    summary_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS user_memory (
                    session_id TEXT PRIMARY KEY,
                    memory_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS retrieval_cache (
                    cache_key TEXT PRIMARY KEY,
                    docs_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS llm_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tokens_in INTEGER NOT NULL,
                    tokens_out INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    summary_used INTEGER NOT NULL,
                    rag_docs_used INTEGER NOT NULL,
                    trimmed_context INTEGER NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def upsert_session(self, session_id: str, user_age: int, age_tier: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(session_id, user_age, age_tier)
                VALUES(?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  user_age=excluded.user_age,
                  age_tier=excluded.age_tier,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (session_id, user_age, age_tier),
            )

    def append_message(self, session_id: str, role: str, content: str, turn_index: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO messages(session_id, role, content, turn_index) VALUES(?, ?, ?, ?)",
                (session_id, role, content, turn_index),
            )

    def get_messages(self, session_id: str, limit: int = 200) -> List[Dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "text": r["content"]} for r in rows]

    def get_summary(self, session_id: str) -> Dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary_json FROM summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["summary_json"])
        except Exception:
            return {}

    def upsert_summary(self, session_id: str, summary: Dict) -> None:
        summary_json = json.dumps(summary, ensure_ascii=True)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO summaries(session_id, summary_json)
                VALUES(?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  summary_json=excluded.summary_json,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (session_id, summary_json),
            )

    def get_user_memory(self, session_id: str) -> Dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT memory_json FROM user_memory WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["memory_json"])
        except Exception:
            return {}

    def upsert_user_memory(self, session_id: str, memory: Dict) -> None:
        memory_json = json.dumps(memory, ensure_ascii=True)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_memory(session_id, memory_json)
                VALUES(?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  memory_json=excluded.memory_json,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (session_id, memory_json),
            )

    def get_retrieval_cache(self, cache_key: str) -> Optional[List[Dict]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT docs_json FROM retrieval_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["docs_json"])
        except Exception:
            return None

    def upsert_retrieval_cache(self, cache_key: str, docs: List[Dict]) -> None:
        docs_json = json.dumps(docs, ensure_ascii=True)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_cache(cache_key, docs_json)
                VALUES(?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                  docs_json=excluded.docs_json,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (cache_key, docs_json),
            )

    def log_metric(
        self,
        session_id: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: int,
        summary_used: bool,
        rag_docs_used: int,
        trimmed_context: bool,
        cache_hit: bool,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_metrics(
                    session_id, tokens_in, tokens_out, latency_ms,
                    summary_used, rag_docs_used, trimmed_context, cache_hit
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    int(tokens_in),
                    int(tokens_out),
                    int(latency_ms),
                    1 if summary_used else 0,
                    int(rag_docs_used),
                    1 if trimmed_context else 0,
                    1 if cache_hit else 0,
                ),
            )

