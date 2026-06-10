"""SQLite-backed conversation history + core memory.

Two concerns, one file:

* **history** — every (user, assistant) turn, per session, with the intent
  that produced it. Read back as short-term context for the next prompt.
* **core memory** — a small persistent key/value scratchpad per session
  (durable facts the user stated, e.g. their name or what they are
  investigating). This is the "core memory" the thesis pipeline carries
  across turns; it is plain data, the model decides nothing about storage.

The store is process-wide and thread-safe (FastAPI serves requests from a
threadpool). SQLite handles the concurrency; a lock serialises writers so
the pipeline stays simple.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    ts         REAL    NOT NULL,
    role       TEXT    NOT NULL,          -- 'user' | 'assistant'
    content    TEXT    NOT NULL,
    intent     TEXT
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id);

CREATE TABLE IF NOT EXISTS core_memory (
    session_id TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    ts         REAL NOT NULL,
    PRIMARY KEY (session_id, key)
);
"""


class ConversationStore:
    def __init__(self, db_path: str) -> None:
        self._path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the connection is shared across the
        # FastAPI threadpool; every write goes through self._lock.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # --------------------------------------------------------------- history
    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (session_id, ts, role, content, intent) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, time.time(), role, content, intent),
            )
            self._conn.commit()

    def recent_turns(self, session_id: str, limit: int) -> list[dict]:
        """Last ``limit`` turns for the session, oldest -> newest."""
        if limit <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, intent FROM turns "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ----------------------------------------------------------- core memory
    def set_core_memory(self, session_id: str, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO core_memory (session_id, key, value, ts) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id, key) DO UPDATE SET "
                "value = excluded.value, ts = excluded.ts",
                (session_id, key, value, time.time()),
            )
            self._conn.commit()

    def get_core_memory(self, session_id: str) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM core_memory WHERE session_id = ? "
                "ORDER BY key",
                (session_id,),
            ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def format_core_memory(self, session_id: str) -> str:
        # ``__``-prefixed keys are internal blobs (e.g. the last image's
        # base64) — never dump them into a prompt.
        items = {
            k: v for k, v in self.get_core_memory(session_id).items()
            if not k.startswith("__")
        }
        if not items:
            return ""
        return "\n".join(f"- {k}: {v}" for k, v in items.items())

    # ----------------------------------- last image (for re-analysis turns)
    _LAST_IMAGE_KEY = "__last_image__"

    def set_last_image(self, session_id: str, data_uri: str) -> None:
        if data_uri:
            self.set_core_memory(session_id, self._LAST_IMAGE_KEY, data_uri)

    def get_last_image(self, session_id: str) -> str:
        return self.get_core_memory(session_id).get(self._LAST_IMAGE_KEY, "")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
