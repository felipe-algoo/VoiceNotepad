import pathlib
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from src.security import content_fingerprint

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    language TEXT NOT NULL DEFAULT 'pt',
    title TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    is_final INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    fingerprint TEXT NOT NULL DEFAULT '',
    UNIQUE(session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_segments_session ON segments(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_segments_fp ON segments(fingerprint);

CREATE TABLE IF NOT EXISTS episodic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodic_session ON episodic(session_id, created_at);
"""


@dataclass(frozen=True)
class Segment:
    id: int
    session_id: str
    seq: int
    text: str
    confidence: float
    is_final: bool
    created_at: float
    fingerprint: str


class EpisodicMemory:
    def __init__(self, path) -> None:
        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self._path),
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with self._write_lock:
            self._conn().executescript(_SCHEMA)

    def create_session(self, session_id: Optional[str] = None, language: str = "pt", title: Optional[str] = None) -> str:
        sid = session_id if session_id else uuid.uuid4().hex
        now = time.time()
        with self._write_lock:
            self._conn().execute(
                "INSERT OR IGNORE INTO sessions(id, created_at, updated_at, language, title) VALUES (?,?,?,?,?)",
                (sid, now, now, language, title),
            )
        return sid

    def session_exists(self, session_id: str) -> bool:
        row = self._conn().execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
        return row is not None

    def set_session_language(self, session_id: str, language: str) -> None:
        with self._write_lock:
            self._conn().execute(
                "UPDATE sessions SET language=?, updated_at=? WHERE id=?",
                (language, time.time(), session_id),
            )

    def next_seq(self, session_id: str) -> int:
        row = self._conn().execute(
            "SELECT COALESCE(MAX(seq), 0) AS m FROM segments WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return int(row["m"]) + 1

    def add_segment(self, session_id: str, text: str, confidence: float, is_final: bool = True, seq: Optional[int] = None) -> Optional[int]:
        if not text:
            return None
        now = time.time()
        fp = content_fingerprint(text)
        with self._write_lock:
            conn = self._conn()
            if seq is None:
                seq = self.next_seq(session_id)
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO segments(session_id, seq, text, confidence, is_final, created_at, fingerprint) VALUES (?,?,?,?,?,?,?)",
                    (session_id, seq, text, float(confidence), 1 if is_final else 0, now, fp),
                )
                conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, session_id))
                conn.execute("COMMIT")
            except sqlite3.IntegrityError:
                conn.execute("ROLLBACK")
                return None
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return seq

    def recent_context(self, session_id: str, limit: int = 5) -> list[str]:
        rows = self._conn().execute(
            "SELECT text FROM segments WHERE session_id=? AND is_final=1 ORDER BY seq DESC LIMIT ?",
            (session_id, int(limit)),
        ).fetchall()
        return [r["text"] for r in reversed(rows)]

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT id, created_at, updated_at, language, title FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        sess = self._conn().execute(
            "SELECT id, created_at, updated_at, language, title FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        if not sess:
            return None
        segs = self._conn().execute(
            "SELECT id, seq, text, confidence, is_final, created_at, fingerprint FROM segments WHERE session_id=? ORDER BY seq ASC",
            (session_id,),
        ).fetchall()
        return {"session": dict(sess), "segments": [dict(s) for s in segs]}

    def delete_session(self, session_id: str) -> None:
        with self._write_lock:
            conn = self._conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM segments WHERE session_id=?", (session_id,))
                conn.execute("DELETE FROM episodic WHERE session_id=?", (session_id,))
                conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def record_event(self, session_id: str, kind: str, payload: str) -> None:
        with self._write_lock:
            self._conn().execute(
                "INSERT INTO episodic(session_id, kind, payload, created_at) VALUES (?,?,?,?)",
                (session_id, kind, payload[:1024], time.time()),
            )

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None