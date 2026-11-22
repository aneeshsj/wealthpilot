"""Persistent SQLite-backed memory subsystem for WealthPilot SME loan underwriting.

Implements four isolated, restart-safe memory layers:
1. WorkingMemory: Per-(applicant_id, thread_id) rolling conversation window (last 8 messages + summary).
2. EpisodicMemory: Historical underwriting decision log seeded from CSV and updated with new events.
3. SemanticMemory: Upsert-clean stable applicant facts keyed by (applicant_id, fact_key).
4. ProceduralMemory: Versioned, retrievable WealthPilot underwriting SOP rules.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

try:
    from src.config import settings as _app_settings  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - supports running with src on PYTHONPATH
    try:
        from config import settings as _app_settings  # type: ignore[import-not-found,no-redef]
    except ImportError:
        _app_settings = None

DEFAULT_SQLITE_ENV_VAR: str = "SQLITE_DB_PATH"
DEFAULT_FALLBACK_DB_PATH: str = "data/wealthpilot/wealthpilot_memory.db"
DEFAULT_PAST_DECISIONS_CSV: str = "data/wealthpilot/mock_api/past_decisions.csv"
DEFAULT_WORKING_WINDOW_SIZE: int = 8
DEFAULT_MAX_SUMMARY_CHARS: int = 1600
MAX_RETRIEVE_LIMIT: int = 50

_TOKEN_RE: re.Pattern[str] = re.compile(r"[a-z0-9]+")
_VALID_ROLES: frozenset[str] = frozenset({"user", "assistant", "system", "tool"})


def _utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _validate_identifier(value: str, field_name: str) -> str:
    """Validate and normalize a required string identifier."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a str, got {type(value).__name__}")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _validate_limit(limit: int, default_limit: int = 10) -> int:
    """Clamp retrieval limits to a safe positive range."""
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    return min(limit, MAX_RETRIEVE_LIMIT) if limit > 0 else default_limit


def _tokenize(text: str) -> set[str]:
    """Extract normalized alphanumeric tokens for lightweight relevance scoring."""
    if not text:
        return set()
    return {match.group(0) for match in _TOKEN_RE.finditer(text.lower())}


def _relevance_score(query_tokens: set[str], candidate_text: str) -> float:
    """Compute token overlap ratio between a query and candidate text."""
    if not query_tokens or not candidate_text:
        return 0.0
    candidate_tokens = _tokenize(candidate_text)
    if not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    return overlap / len(query_tokens)


def _resolve_db_path(db_path: str | Path | None) -> Path:
    """Resolve the SQLite database path from argument, settings, or environment."""
    if db_path is not None:
        return Path(db_path)
    if _app_settings is not None and hasattr(_app_settings, "SQLITE_DB_PATH"):
        configured = getattr(_app_settings, "SQLITE_DB_PATH")
        if configured:
            return Path(configured)
    env_path = os.environ.get(DEFAULT_SQLITE_ENV_VAR)
    if env_path:
        return Path(env_path)
    logger.debug(
        "settings.SQLITE_DB_PATH unavailable; using fallback SQLite path %s",
        DEFAULT_FALLBACK_DB_PATH,
    )
    return Path(DEFAULT_FALLBACK_DB_PATH)


@dataclass(frozen=True, slots=True)
class WorkingMessage:
    """Single message stored in WorkingMemory."""

    message_id: int
    applicant_id: str
    thread_id: str
    role: str
    content: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkingContext:
    """Bounded working memory snapshot for a single (applicant_id, thread_id)."""

    applicant_id: str
    thread_id: str
    summary: str
    summarized_message_count: int
    messages: tuple[WorkingMessage, ...]


@dataclass(frozen=True, slots=True)
class EpisodicEvent:
    """Timestamped underwriting event stored in EpisodicMemory."""

    event_id: str
    applicant_id: str
    thread_id: str
    timestamp: str
    summary: str
    outcome: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SemanticFact:
    """Stable applicant attribute stored in SemanticMemory."""

    applicant_id: str
    fact_key: str
    fact_value: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ProceduralRule:
    """Versioned WealthPilot underwriting SOP rule stored in ProceduralMemory."""

    rule_id: str
    step_order: int
    task_type: str
    instruction: str
    version: int
    is_active: bool
    created_at: str


SummarizerFn = Callable[[Sequence[WorkingMessage], str], str]


def _default_rolling_summarizer(
    evicted_messages: Sequence[WorkingMessage],
    prior_summary: str,
    max_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
) -> str:
    """Deterministically fold evicted messages into a bounded rolling summary."""
    lines: list[str] = []
    if prior_summary.strip():
        for raw_line in prior_summary.strip().splitlines():
            stripped = raw_line.strip()
            if stripped:
                lines.append(stripped)

    for msg in evicted_messages:
        collapsed = " ".join(msg.content.split())
        if len(collapsed) > 220:
            collapsed = f"{collapsed[:217]}..."
        entry = f"[{msg.role}] {collapsed}"
        if not lines or lines[-1] != entry:
            lines.append(entry)

    combined = "\n".join(lines)
    if len(combined) <= max_chars:
        return combined

    # Retain the earliest context anchor if possible, plus the most recent lines.
    kept_tail: list[str] = []
    budget = max_chars - 24
    for line in reversed(lines):
        cost = len(line) + 1
        if budget - cost < 0 and kept_tail:
            break
        kept_tail.append(line)
        budget -= cost
    kept_tail.reverse()
    return "...[earlier context trimmed]\n" + "\n".join(kept_tail)


class _SQLiteBackend:
    """Thread-safe SQLite connection manager with WAL persistence."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path: Path = _resolve_db_path(db_path)
        self._is_in_memory: bool = str(self.db_path) == ":memory:"
        self._lock: threading.RLock = threading.RLock()

        if not self._is_in_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON;")
            if not self._is_in_memory:
                self._conn.execute("PRAGMA journal_mode = WAL;")
                self._conn.execute("PRAGMA synchronous = NORMAL;")
            self._conn.execute("PRAGMA busy_timeout = 5000;")

    @property
    def lock(self) -> threading.RLock:
        """Return the re-entrant lock protecting database access."""
        return self._lock

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the active SQLite connection."""
        return self._conn

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()


class WorkingMemory:
    """Session working memory isolated by (applicant_id, thread_id) with a rolling window."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        window_size: int = DEFAULT_WORKING_WINDOW_SIZE,
        summarizer: SummarizerFn | None = None,
        backend: _SQLiteBackend | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._backend: _SQLiteBackend = backend or _SQLiteBackend(db_path)
        self._owns_backend: bool = backend is None
        self.window_size: int = window_size
        self._summarizer: SummarizerFn = summarizer or _default_rolling_summarizer
        self._init_schema()

    def _init_schema(self) -> None:
        with self._backend.lock:
            conn = self._backend.connection
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS working_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        applicant_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_working_messages_partition
                        ON working_messages (applicant_id, thread_id, id);

                    CREATE TABLE IF NOT EXISTS working_summaries (
                        applicant_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        summary_text TEXT NOT NULL,
                        summarized_message_count INTEGER NOT NULL DEFAULT 0,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (applicant_id, thread_id)
                    );
                    """
                )

    def save_message(
        self,
        applicant_id: str,
        thread_id: str,
        role: str,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkingMessage:
        """Persist a message for (applicant_id, thread_id) and trim beyond window_size."""
        clean_applicant = _validate_identifier(applicant_id, "applicant_id")
        clean_thread = _validate_identifier(thread_id, "thread_id")
        clean_role = _validate_identifier(role, "role").lower()
        if clean_role not in _VALID_ROLES:
            raise ValueError(
                f"role must be one of {sorted(_VALID_ROLES)}, got {clean_role!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")

        created_at = _utc_now_iso()
        metadata_json = json.dumps(dict(metadata) if metadata else {}, sort_keys=True)

        with self._backend.lock:
            conn = self._backend.connection
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO working_messages (
                        applicant_id, thread_id, role, content, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clean_applicant,
                        clean_thread,
                        clean_role,
                        content,
                        metadata_json,
                        created_at,
                    ),
                )
                inserted_id = int(cursor.lastrowid)
                self._trim_window_locked(conn, clean_applicant, clean_thread)

        logger.debug(
            "Saved working message id=%d for applicant_id=%s thread_id=%s",
            inserted_id,
            clean_applicant,
            clean_thread,
        )
        return WorkingMessage(
            message_id=inserted_id,
            applicant_id=clean_applicant,
            thread_id=clean_thread,
            role=clean_role,
            content=content,
            created_at=created_at,
            metadata=dict(metadata) if metadata else {},
        )

    def _trim_window_locked(
        self,
        conn: sqlite3.Connection,
        applicant_id: str,
        thread_id: str,
    ) -> None:
        """Summarize and evict messages older than the most recent window_size items."""
        rows = conn.execute(
            """
            SELECT id, applicant_id, thread_id, role, content, metadata_json, created_at
            FROM working_messages
            WHERE applicant_id = ? AND thread_id = ?
            ORDER BY id ASC
            """,
            (applicant_id, thread_id),
        ).fetchall()

        excess = len(rows) - self.window_size
        if excess <= 0:
            return

        evicted_rows = rows[:excess]
        evicted_messages = [
            WorkingMessage(
                message_id=int(row["id"]),
                applicant_id=str(row["applicant_id"]),
                thread_id=str(row["thread_id"]),
                role=str(row["role"]),
                content=str(row["content"]),
                created_at=str(row["created_at"]),
                metadata=json.loads(str(row["metadata_json"])),
            )
            for row in evicted_rows
        ]

        summary_row = conn.execute(
            """
            SELECT summary_text, summarized_message_count
            FROM working_summaries
            WHERE applicant_id = ? AND thread_id = ?
            """,
            (applicant_id, thread_id),
        ).fetchone()

        prior_summary = str(summary_row["summary_text"]) if summary_row else ""
        prior_count = int(summary_row["summarized_message_count"]) if summary_row else 0

        updated_summary = self._summarizer(evicted_messages, prior_summary)
        updated_count = prior_count + len(evicted_messages)
        now_iso = _utc_now_iso()

        conn.execute(
            """
            INSERT INTO working_summaries (
                applicant_id, thread_id, summary_text, summarized_message_count, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(applicant_id, thread_id) DO UPDATE SET
                summary_text = excluded.summary_text,
                summarized_message_count = excluded.summarized_message_count,
                updated_at = excluded.updated_at
            """,
            (applicant_id, thread_id, updated_summary, updated_count, now_iso),
        )

        evict_ids = [(int(row["id"]),) for row in evicted_rows]
        conn.executemany(
            "DELETE FROM working_messages WHERE id = ? AND applicant_id = ? AND thread_id = ?",
            [(row_id, applicant_id, thread_id) for (row_id,) in evict_ids],
        )
        logger.debug(
            "Trimmed %d working messages into rolling summary for applicant_id=%s thread_id=%s",
            len(evicted_messages),
            applicant_id,
            thread_id,
        )

    def retrieve_context(
        self,
        applicant_id: str,
        thread_id: str,
        *,
        limit: int | None = None,
        query: str | None = None,
    ) -> WorkingContext:
        """Retrieve bounded rolling summary and recent messages for (applicant_id, thread_id)."""
        clean_applicant = _validate_identifier(applicant_id, "applicant_id")
        clean_thread = _validate_identifier(thread_id, "thread_id")
        effective_limit = (
            min(_validate_limit(limit), self.window_size)
            if limit is not None
            else self.window_size
        )

        with self._backend.lock:
            conn = self._backend.connection
            summary_row = conn.execute(
                """
                SELECT summary_text, summarized_message_count
                FROM working_summaries
                WHERE applicant_id = ? AND thread_id = ?
                """,
                (clean_applicant, clean_thread),
            ).fetchone()

            msg_rows = conn.execute(
                """
                SELECT id, applicant_id, thread_id, role, content, metadata_json, created_at
                FROM working_messages
                WHERE applicant_id = ? AND thread_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (clean_applicant, clean_thread, effective_limit),
            ).fetchall()

        messages = [
            WorkingMessage(
                message_id=int(row["id"]),
                applicant_id=str(row["applicant_id"]),
                thread_id=str(row["thread_id"]),
                role=str(row["role"]),
                content=str(row["content"]),
                created_at=str(row["created_at"]),
                metadata=json.loads(str(row["metadata_json"])),
            )
            for row in reversed(msg_rows)
        ]

        if query and query.strip() and len(messages) > 1:
            query_tokens = _tokenize(query)
            # Keep exact chronological order among relevant messages when filtering,
            # always preserving the most recent message for conversational continuity.
            latest_id = messages[-1].message_id
            scored = [
                msg
                for msg in messages
                if msg.message_id == latest_id or _relevance_score(query_tokens, msg.content) > 0.0
            ]
            if scored:
                messages = scored

        return WorkingContext(
            applicant_id=clean_applicant,
            thread_id=clean_thread,
            summary=str(summary_row["summary_text"]) if summary_row else "",
            summarized_message_count=(
                int(summary_row["summarized_message_count"]) if summary_row else 0
            ),
            messages=tuple(messages),
        )

    def clear_thread(self, applicant_id: str, thread_id: str) -> int:
        """Remove all working messages and summary for a single (applicant_id, thread_id)."""
        clean_applicant = _validate_identifier(applicant_id, "applicant_id")
        clean_thread = _validate_identifier(thread_id, "thread_id")
        with self._backend.lock:
            conn = self._backend.connection
            with conn:
                msg_cur = conn.execute(
                    "DELETE FROM working_messages WHERE applicant_id = ? AND thread_id = ?",
                    (clean_applicant, clean_thread),
                )
                conn.execute(
                    "DELETE FROM working_summaries WHERE applicant_id = ? AND thread_id = ?",
                    (clean_applicant, clean_thread),
                )
        return int(msg_cur.rowcount)

    def close(self) -> None:
        """Close SQLite connection if owned by this instance."""
        if self._owns_backend:
            self._backend.close()


class EpisodicMemory:
    """Applicant-isolated historical underwriting event memory seeded from CSV."""

    _SEED_META_KEY: str = "episodic_past_decisions_seeded"

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        seed_csv_path: str | Path = DEFAULT_PAST_DECISIONS_CSV,
        backend: _SQLiteBackend | None = None,
    ) -> None:
        self._backend: _SQLiteBackend = backend or _SQLiteBackend(db_path)
        self._owns_backend: bool = backend is None
        self.seed_csv_path: Path = Path(seed_csv_path)
        self._init_schema()
        self._seed_on_first_run()

    def _init_schema(self) -> None:
        with self._backend.lock:
            conn = self._backend.connection
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS memory_metadata (
                        meta_key TEXT PRIMARY KEY,
                        meta_value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS episodic_events (
                        event_id TEXT PRIMARY KEY,
                        applicant_id TEXT NOT NULL,
                        thread_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}'
                    );

                    CREATE INDEX IF NOT EXISTS idx_episodic_applicant_ts
                        ON episodic_events (applicant_id, timestamp DESC);
                    """
                )

    def _seed_on_first_run(self) -> None:
        with self._backend.lock:
            conn = self._backend.connection
            marker = conn.execute(
                "SELECT meta_value FROM memory_metadata WHERE meta_key = ?",
                (self._SEED_META_KEY,),
            ).fetchone()
            if marker is not None:
                return

            if not self.seed_csv_path.exists():
                logger.warning(
                    "Episodic seed CSV not found at %s; skipping initial seed",
                    self.seed_csv_path,
                )
                return

            seeded_count = 0
            records_to_insert: list[tuple[str, str, str, str, str, str, str]] = []
            try:
                with self.seed_csv_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row_idx, raw_row in enumerate(reader):
                        parsed = self._parse_csv_row(raw_row, row_idx)
                        if parsed is not None:
                            records_to_insert.append(parsed)
            except OSError as exc:
                logger.exception(
                    "Failed reading episodic seed CSV at %s: %s",
                    self.seed_csv_path,
                    exc,
                )
                return

            with conn:
                if records_to_insert:
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO episodic_events (
                            event_id, applicant_id, thread_id, timestamp,
                            summary, outcome, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        records_to_insert,
                    )
                    seeded_count = len(records_to_insert)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_metadata (meta_key, meta_value, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (self._SEED_META_KEY, str(seeded_count), _utc_now_iso()),
                )

            logger.info(
                "Seeded %d historical episodic records from %s",
                seeded_count,
                self.seed_csv_path,
            )

    @staticmethod
    def _parse_csv_row(
        raw_row: Mapping[str, str | None],
        row_idx: int,
    ) -> tuple[str, str, str, str, str, str, str] | None:
        """Normalize a CSV row into an episodic_events tuple."""
        normalized: dict[str, str] = {
            (k or "").strip().lower(): (v or "").strip()
            for k, v in raw_row.items()
            if k is not None
        }

        applicant_id = (
            normalized.get("applicant_id")
            or normalized.get("sme_id")
            or normalized.get("borrower_id")
            or normalized.get("company_id")
            or ""
        )
        if not applicant_id:
            logger.warning(
                "Skipping past_decisions.csv row %d due to missing applicant_id",
                row_idx,
            )
            return None

        thread_id = (
            normalized.get("thread_id")
            or normalized.get("session_id")
            or f"historical-{applicant_id}"
        )
        timestamp = (
            normalized.get("timestamp")
            or normalized.get("decision_date")
            or normalized.get("created_at")
            or normalized.get("date")
            or _utc_now_iso()
        )
        outcome = (
            normalized.get("outcome")
            or normalized.get("decision")
            or normalized.get("status")
            or "UNKNOWN"
        )
        summary = (
            normalized.get("summary")
            or normalized.get("rationale")
            or normalized.get("notes")
            or normalized.get("reason")
            or normalized.get("description")
            or ""
        )

        consumed_keys = {
            "event_id",
            "decision_id",
            "id",
            "applicant_id",
            "sme_id",
            "borrower_id",
            "company_id",
            "thread_id",
            "session_id",
            "timestamp",
            "decision_date",
            "created_at",
            "date",
            "summary",
            "rationale",
            "notes",
            "reason",
            "description",
            "outcome",
            "decision",
            "status",
        }
        extra_metadata = {
            k: v for k, v in normalized.items() if k not in consumed_keys and v != ""
        }

        if not summary:
            details = ", ".join(f"{k}={v}" for k, v in sorted(extra_metadata.items()))
            summary = f"Historical decision {outcome}" + (f" ({details})" if details else "")

        event_id = (
            normalized.get("event_id")
            or normalized.get("decision_id")
            or normalized.get("id")
            or str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{applicant_id}:{thread_id}:{timestamp}:{outcome}:{summary}:{row_idx}",
                )
            )
        )

        return (
            event_id,
            applicant_id,
            thread_id,
            timestamp,
            summary,
            outcome,
            json.dumps(extra_metadata, sort_keys=True),
        )

    def save_event(
        self,
        applicant_id: str,
        thread_id: str,
        summary: str,
        outcome: str,
        *,
        event_id: str | None = None,
        timestamp: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EpisodicEvent:
        """Log a new timestamped underwriting event strictly under applicant_id."""
        clean_applicant = _validate_identifier(applicant_id, "applicant_id")
        clean_thread = _validate_identifier(thread_id, "thread_id")
        clean_summary = _validate_identifier(summary, "summary")
        clean_outcome = _validate_identifier(outcome, "outcome")
        clean_event_id = (
            _validate_identifier(event_id, "event_id")
            if event_id is not None
            else str(uuid.uuid4())
        )
        clean_timestamp = (
            _validate_identifier(timestamp, "timestamp")
            if timestamp is not None
            else _utc_now_iso()
        )
        metadata_dict = dict(metadata) if metadata else {}
        metadata_json = json.dumps(metadata_dict, sort_keys=True)

        with self._backend.lock:
            conn = self._backend.connection
            with conn:
                conn.execute(
                    """
                    INSERT INTO episodic_events (
                        event_id, applicant_id, thread_id, timestamp,
                        summary, outcome, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET
                        summary = excluded.summary,
                        outcome = excluded.outcome,
                        metadata_json = excluded.metadata_json,
                        timestamp = excluded.timestamp
                    WHERE episodic_events.applicant_id = excluded.applicant_id
                    """,
                    (
                        clean_event_id,
                        clean_applicant,
                        clean_thread,
                        clean_timestamp,
                        clean_summary,
                        clean_outcome,
                        metadata_json,
                    ),
                )

        logger.info(
            "Logged episodic event event_id=%s applicant_id=%s outcome=%s",
            clean_event_id,
            clean_applicant,
            clean_outcome,
        )
        return EpisodicEvent(
            event_id=clean_event_id,
            applicant_id=clean_applicant,
            thread_id=clean_thread,
            timestamp=clean_timestamp,
            summary=clean_summary,
            outcome=clean_outcome,
            metadata=metadata_dict,
        )

    def retrieve_events(
        self,
        applicant_id: str,
        *,
        query: str | None = None,
        outcome: str | None = None,
        limit: int = 5,
    ) -> list[EpisodicEvent]:
        """Retrieve relevant, bounded episodic events strictly for applicant_id."""
        clean_applicant = _validate_identifier(applicant_id, "applicant_id")
        safe_limit = _validate_limit(limit, default_limit=5)
        clean_outcome = outcome.strip() if isinstance(outcome, str) and outcome.strip() else None

        sql = """
            SELECT event_id, applicant_id, thread_id, timestamp, summary, outcome, metadata_json
            FROM episodic_events
            WHERE applicant_id = ?
        """
        params: list[Any] = [clean_applicant]
        if clean_outcome is not None:
            sql += " AND UPPER(outcome) = UPPER(?)"
            params.append(clean_outcome)
        sql += " ORDER BY timestamp DESC"

        # If a relevance query is provided, inspect a bounded candidate window for that applicant.
        fetch_limit = safe_limit if not (query and query.strip()) else min(safe_limit * 6, 150)
        sql += " LIMIT ?"
        params.append(fetch_limit)

        with self._backend.lock:
            rows = self._backend.connection.execute(sql, params).fetchall()

        events = [
            EpisodicEvent(
                event_id=str(row["event_id"]),
                applicant_id=str(row["applicant_id"]),
                thread_id=str(row["thread_id"]),
                timestamp=str(row["timestamp"]),
                summary=str(row["summary"]),
                outcome=str(row["outcome"]),
                metadata=json.loads(str(row["metadata_json"])),
            )
            for row in rows
        ]

        if query and query.strip() and events:
            query_tokens = _tokenize(query)
            scored = sorted(
                events,
                key=lambda ev: (
                    _relevance_score(
                        query_tokens,
                        f"{ev.summary} {ev.outcome} {json.dumps(ev.metadata)}",
                    ),
                    ev.timestamp,
                ),
                reverse=True,
            )
            return scored[:safe_limit]

        return events[:safe_limit]

    def close(self) -> None:
        """Close SQLite connection if owned by this instance."""
        if self._owns_backend:
            self._backend.close()


class SemanticMemory:
    """Applicant-isolated stable factual memory with clean overwrite semantics."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        backend: _SQLiteBackend | None = None,
    ) -> None:
        self._backend: _SQLiteBackend = backend or _SQLiteBackend(db_path)
        self._owns_backend: bool = backend is None
        self._init_schema()

    def _init_schema(self) -> None:
        with self._backend.lock:
            conn = self._backend.connection
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS semantic_facts (
                        applicant_id TEXT NOT NULL,
                        fact_key TEXT NOT NULL,
                        fact_value TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (applicant_id, fact_key)
                    );

                    CREATE INDEX IF NOT EXISTS idx_semantic_applicant_updated
                        ON semantic_facts (applicant_id, updated_at DESC);
                    """
                )

    def save_fact(
        self,
        applicant_id: str,
        fact_key: str,
        fact_value: str | int | float | bool,
    ) -> SemanticFact:
        """Insert or cleanly overwrite a stable fact for applicant_id."""
        clean_applicant = _validate_identifier(applicant_id, "applicant_id")
        clean_key = _validate_identifier(fact_key, "fact_key").lower()
        clean_value = str(fact_value).strip()
        if not clean_value:
            raise ValueError("fact_value must not be empty")
        updated_at = _utc_now_iso()

        with self._backend.lock:
            conn = self._backend.connection
            with conn:
                conn.execute(
                    """
                    INSERT INTO semantic_facts (
                        applicant_id, fact_key, fact_value, updated_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(applicant_id, fact_key) DO UPDATE SET
                        fact_value = excluded.fact_value,
                        updated_at = excluded.updated_at
                    """,
                    (clean_applicant, clean_key, clean_value, updated_at),
                )

        logger.debug(
            "Saved semantic fact key=%s for applicant_id=%s",
            clean_key,
            clean_applicant,
        )
        return SemanticFact(
            applicant_id=clean_applicant,
            fact_key=clean_key,
            fact_value=clean_value,
            updated_at=updated_at,
        )

    def save_facts(
        self,
        applicant_id: str,
        facts: Mapping[str, str | int | float | bool],
    ) -> list[SemanticFact]:
        """Atomically upsert multiple stable facts for applicant_id."""
        clean_applicant = _validate_identifier(applicant_id, "applicant_id")
        if not facts:
            return []

        updated_at = _utc_now_iso()
        records: list[SemanticFact] = []
        rows: list[tuple[str, str, str, str]] = []
        for raw_key, raw_value in facts.items():
            clean_key = _validate_identifier(raw_key, "fact_key").lower()
            clean_value = str(raw_value).strip()
            if not clean_value:
                raise ValueError(f"fact_value for key {clean_key!r} must not be empty")
            records.append(
                SemanticFact(
                    applicant_id=clean_applicant,
                    fact_key=clean_key,
                    fact_value=clean_value,
                    updated_at=updated_at,
                )
            )
            rows.append((clean_applicant, clean_key, clean_value, updated_at))

        with self._backend.lock:
            conn = self._backend.connection
            with conn:
                conn.executemany(
                    """
                    INSERT INTO semantic_facts (
                        applicant_id, fact_key, fact_value, updated_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(applicant_id, fact_key) DO UPDATE SET
                        fact_value = excluded.fact_value,
                        updated_at = excluded.updated_at
                    """,
                    rows,
                )

        logger.debug(
            "Upserted %d semantic facts for applicant_id=%s",
            len(records),
            clean_applicant,
        )
        return records

    def retrieve_facts(
        self,
        applicant_id: str,
        *,
        query: str | None = None,
        keys: Sequence[str] | None = None,
        limit: int = 15,
    ) -> list[SemanticFact]:
        """Retrieve bounded stable facts strictly for applicant_id."""
        clean_applicant = _validate_identifier(applicant_id, "applicant_id")
        safe_limit = _validate_limit(limit, default_limit=15)

        sql = """
            SELECT applicant_id, fact_key, fact_value, updated_at
            FROM semantic_facts
            WHERE applicant_id = ?
        """
        params: list[Any] = [clean_applicant]

        if keys:
            clean_keys = [_validate_identifier(k, "fact_key").lower() for k in keys]
            placeholders = ", ".join("?" for _ in clean_keys)
            sql += f" AND fact_key IN ({placeholders})"
            params.extend(clean_keys)

        sql += " ORDER BY updated_at DESC, fact_key ASC"

        with self._backend.lock:
            rows = self._backend.connection.execute(sql, params).fetchall()

        facts = [
            SemanticFact(
                applicant_id=str(row["applicant_id"]),
                fact_key=str(row["fact_key"]),
                fact_value=str(row["fact_value"]),
                updated_at=str(row["updated_at"]),
            )
            for row in rows
        ]

        if query and query.strip() and facts:
            query_tokens = _tokenize(query)
            scored = sorted(
                facts,
                key=lambda f: (
                    _relevance_score(query_tokens, f"{f.fact_key} {f.fact_value}"),
                    f.updated_at,
                ),
                reverse=True,
            )
            return scored[:safe_limit]

        return facts[:safe_limit]

    def retrieve_fact_map(
        self,
        applicant_id: str,
        *,
        query: str | None = None,
        keys: Sequence[str] | None = None,
        limit: int = 15,
    ) -> dict[str, str]:
        """Convenience wrapper returning retrieved semantic facts as a dict."""
        return {
            fact.fact_key: fact.fact_value
            for fact in self.retrieve_facts(
                applicant_id, query=query, keys=keys, limit=limit
            )
        }

    def delete_fact(self, applicant_id: str, fact_key: str) -> bool:
        """Delete a single fact strictly for applicant_id."""
        clean_applicant = _validate_identifier(applicant_id, "applicant_id")
        clean_key = _validate_identifier(fact_key, "fact_key").lower()
        with self._backend.lock:
            conn = self._backend.connection
            with conn:
                cur = conn.execute(
                    "DELETE FROM semantic_facts WHERE applicant_id = ? AND fact_key = ?",
                    (clean_applicant, clean_key),
                )
        return cur.rowcount > 0

    def close(self) -> None:
        """Close SQLite connection if owned by this instance."""
        if self._owns_backend:
            self._backend.close()


DEFAULT_WEALTHPILOT_SOPS: tuple[tuple[str, int, str, str, int], ...] = (
    (
        "WP-SOP-001",
        1,
        "intake_sanitization",
        (
            "Strip all demographic proxies (age, gender, ethnicity, nationality, "
            "marital status, and demographic postal-code signals) before risk scoring "
            "or underwriting synthesis."
        ),
        1,
    ),
    (
        "WP-SOP-002",
        2,
        "financial_verification",
        (
            "Verify SME revenue, EBITDA, existing debt service, and cash reserves "
            "against bank statements and tax filings, persisting reconciled figures "
            "to Semantic Memory."
        ),
        1,
    ),
    (
        "WP-SOP-003",
        3,
        "dscr_compliance",
        (
            "Always verify the applicable sector DSCR floor before scoring or "
            "recommending approval; any application below the sector DSCR minimum "
            "must be flagged for manual credit review or declined."
        ),
        1,
    ),
    (
        "WP-SOP-004",
        4,
        "credit_history_check",
        (
            "Query Episodic Memory strictly scoped to the target applicant_id for "
            "prior underwriting outcomes, defaults, or covenant breaches; never "
            "reference records belonging to another applicant_id."
        ),
        1,
    ),
    (
        "WP-SOP-005",
        5,
        "underwriting_decision",
        (
            "Issue a structured underwriting outcome (APPROVE, REFER, or DECLINE) "
            "grounded in verified financial ratios, DSCR floor compliance, and "
            "collateral coverage, then log the timestamped event to Episodic Memory."
        ),
        1,
    ),
)


class ProceduralMemory:
    """Versioned WealthPilot SME underwriting SOP rule repository."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        backend: _SQLiteBackend | None = None,
    ) -> None:
        self._backend: _SQLiteBackend = backend or _SQLiteBackend(db_path)
        self._owns_backend: bool = backend is None
        self._init_schema()
        self._seed_default_rules()

    def _init_schema(self) -> None:
        with self._backend.lock:
            conn = self._backend.connection
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS procedural_rules (
                        rule_id TEXT NOT NULL,
                        version INTEGER NOT NULL,
                        step_order INTEGER NOT NULL,
                        task_type TEXT NOT NULL,
                        instruction TEXT NOT NULL,
                        is_active INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (rule_id, version)
                    );

                    CREATE INDEX IF NOT EXISTS idx_procedural_active_order
                        ON procedural_rules (is_active, task_type, step_order);
                    """
                )

    def _seed_default_rules(self) -> None:
        with self._backend.lock:
            conn = self._backend.connection
            row = conn.execute("SELECT COUNT(*) AS cnt FROM procedural_rules").fetchone()
            if row is not None and int(row["cnt"]) > 0:
                return

            now_iso = _utc_now_iso()
            with conn:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO procedural_rules (
                        rule_id, version, step_order, task_type,
                        instruction, is_active, created_at
                    ) VALUES (?, ?, ?, ?, ?, 1, ?)
                    """,
                    [
                        (rule_id, version, step_order, task_type, instruction, now_iso)
                        for rule_id, step_order, task_type, instruction, version in DEFAULT_WEALTHPILOT_SOPS
                    ],
                )
            logger.info(
                "Initialized %d default WealthPilot SOP rules in ProceduralMemory",
                len(DEFAULT_WEALTHPILOT_SOPS),
            )

    def save_rule(
        self,
        rule_id: str,
        step_order: int,
        task_type: str,
        instruction: str,
        *,
        version: int | None = None,
        is_active: bool = True,
    ) -> ProceduralRule:
        """Save or create a new version of an underwriting SOP rule."""
        clean_rule_id = _validate_identifier(rule_id, "rule_id")
        clean_task_type = _validate_identifier(task_type, "task_type").lower()
        clean_instruction = _validate_identifier(instruction, "instruction")
        if step_order < 1:
            raise ValueError(f"step_order must be >= 1, got {step_order}")
        if version is not None and version < 1:
            raise ValueError(f"version must be >= 1, got {version}")

        created_at = _utc_now_iso()
        with self._backend.lock:
            conn = self._backend.connection
            with conn:
                if version is None:
                    ver_row = conn.execute(
                        "SELECT COALESCE(MAX(version), 0) AS max_ver FROM procedural_rules WHERE rule_id = ?",
                        (clean_rule_id,),
                    ).fetchone()
                    target_version = int(ver_row["max_ver"]) + 1 if ver_row else 1
                else:
                    target_version = version

                if is_active:
                    conn.execute(
                        "UPDATE procedural_rules SET is_active = 0 WHERE rule_id = ?",
                        (clean_rule_id,),
                    )

                conn.execute(
                    """
                    INSERT INTO procedural_rules (
                        rule_id, version, step_order, task_type,
                        instruction, is_active, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(rule_id, version) DO UPDATE SET
                        step_order = excluded.step_order,
                        task_type = excluded.task_type,
                        instruction = excluded.instruction,
                        is_active = excluded.is_active,
                        created_at = excluded.created_at
                    """,
                    (
                        clean_rule_id,
                        target_version,
                        step_order,
                        clean_task_type,
                        clean_instruction,
                        1 if is_active else 0,
                        created_at,
                    ),
                )

        logger.info(
            "Saved procedural rule rule_id=%s version=%d task_type=%s active=%s",
            clean_rule_id,
            target_version,
            clean_task_type,
            is_active,
        )
        return ProceduralRule(
            rule_id=clean_rule_id,
            step_order=step_order,
            task_type=clean_task_type,
            instruction=clean_instruction,
            version=target_version,
            is_active=is_active,
            created_at=created_at,
        )

    def retrieve_rules(
        self,
        *,
        task_type: str | None = None,
        query: str | None = None,
        latest_only: bool = True,
        limit: int = 10,
    ) -> list[ProceduralRule]:
        """Retrieve ordered, relevance-filtered underwriting SOP rules."""
        safe_limit = _validate_limit(limit, default_limit=10)
        clauses: list[str] = []
        params: list[Any] = []

        if latest_only:
            clauses.append("is_active = 1")
        if task_type and task_type.strip():
            clauses.append("task_type = ?")
            params.append(task_type.strip().lower())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT rule_id, version, step_order, task_type, instruction, is_active, created_at
            FROM procedural_rules
            {where_sql}
            ORDER BY step_order ASC, version DESC
        """

        with self._backend.lock:
            rows = self._backend.connection.execute(sql, params).fetchall()

        rules = [
            ProceduralRule(
                rule_id=str(row["rule_id"]),
                step_order=int(row["step_order"]),
                task_type=str(row["task_type"]),
                instruction=str(row["instruction"]),
                version=int(row["version"]),
                is_active=bool(row["is_active"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

        if query and query.strip() and rules:
            query_tokens = _tokenize(query)
            scored = [
                (
                    _relevance_score(
                        query_tokens,
                        f"{rule.rule_id} {rule.task_type} {rule.instruction}",
                    ),
                    rule,
                )
                for rule in rules
            ]
            relevant = [
                rule
                for score, rule in sorted(
                    scored,
                    key=lambda item: (-item[0], item[1].step_order, -item[1].version),
                )
                if score > 0.0
            ]
            if relevant:
                return relevant[:safe_limit]

        return rules[:safe_limit]

    def retrieve_rule_versions(self, rule_id: str) -> list[ProceduralRule]:
        """Retrieve full version history for a specific SOP rule_id."""
        clean_rule_id = _validate_identifier(rule_id, "rule_id")
        with self._backend.lock:
            rows = self._backend.connection.execute(
                """
                SELECT rule_id, version, step_order, task_type, instruction, is_active, created_at
                FROM procedural_rules
                WHERE rule_id = ?
                ORDER BY version DESC
                """,
                (clean_rule_id,),
            ).fetchall()

        return [
            ProceduralRule(
                rule_id=str(row["rule_id"]),
                step_order=int(row["step_order"]),
                task_type=str(row["task_type"]),
                instruction=str(row["instruction"]),
                version=int(row["version"]),
                is_active=bool(row["is_active"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def close(self) -> None:
        """Close SQLite connection if owned by this instance."""
        if self._owns_backend:
            self._backend.close()


class WealthPilotMemoryManager:
    """Unified coordinator sharing one SQLite backend across all four memory types."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        seed_csv_path: str | Path = DEFAULT_PAST_DECISIONS_CSV,
        window_size: int = DEFAULT_WORKING_WINDOW_SIZE,
        summarizer: SummarizerFn | None = None,
    ) -> None:
        self._backend: _SQLiteBackend = _SQLiteBackend(db_path)
        self.working: WorkingMemory = WorkingMemory(
            window_size=window_size,
            summarizer=summarizer,
            backend=self._backend,
        )
        self.episodic: EpisodicMemory = EpisodicMemory(
            seed_csv_path=seed_csv_path,
            backend=self._backend,
        )
        self.semantic: SemanticMemory = SemanticMemory(backend=self._backend)
        self.procedural: ProceduralMemory = ProceduralMemory(backend=self._backend)

    def retrieve_applicant_bundle(
        self,
        applicant_id: str,
        thread_id: str,
        *,
        query: str | None = None,
        task_type: str | None = None,
        working_limit: int = DEFAULT_WORKING_WINDOW_SIZE,
        episodic_limit: int = 5,
        semantic_limit: int = 15,
        procedural_limit: int = 5,
    ) -> dict[str, Any]:
        """Assemble bounded, strictly applicant_id-isolated context across all 4 layers."""
        clean_applicant = _validate_identifier(applicant_id, "applicant_id")
        clean_thread = _validate_identifier(thread_id, "thread_id")

        return {
            "applicant_id": clean_applicant,
            "thread_id": clean_thread,
            "working": self.working.retrieve_context(
                clean_applicant,
                clean_thread,
                limit=working_limit,
                query=query,
            ),
            "episodic": self.episodic.retrieve_events(
                clean_applicant,
                query=query,
                limit=episodic_limit,
            ),
            "semantic": self.semantic.retrieve_facts(
                clean_applicant,
                query=query,
                limit=semantic_limit,
            ),
            "procedural": self.procedural.retrieve_rules(
                task_type=task_type,
                query=query,
                limit=procedural_limit,
            ),
        }

    def close(self) -> None:
        """Close the shared SQLite connection."""
        self._backend.close()


__all__ = [
    "DEFAULT_WEALTHPILOT_SOPS",
    "DEFAULT_WORKING_WINDOW_SIZE",
    "EpisodicEvent",
    "EpisodicMemory",
    "ProceduralMemory",
    "ProceduralRule",
    "SemanticFact",
    "SemanticMemory",
    "SummarizerFn",
    "WealthPilotMemoryManager",
    "WorkingContext",
    "WorkingMemory",
    "WorkingMessage",
]

