"""
detecto.models.record
=====================

Defines the two contracts every other backend module depends on:

1. **API schemas** (Pydantic) — the exact JSON shapes returned by the
   endpoints. FastAPI uses these to validate and to generate docs.
2. **Storage schema + repository** (stdlib sqlite3) — the ``detections``
   table and the small set of functions that read/write it.

Why keep them in one module?
    The API response and the stored row describe the same fact. Keeping
    them adjacent makes any drift between the two obvious at review time.

Why stdlib sqlite3 instead of an ORM?
    Our access pattern is a handful of parameterised filters. An ORM would
    add a dependency and hide the SQL we specifically want to own and
    explain. sqlite3 ships with Python, so the install stays small.

Concurrency note:
    SQLite is opened per-operation (short-lived connections). WAL mode is
    enabled so readers do not block the writer, which matters when the
    frontend polls /history while /detect is inserting.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """A single person detection, in ORIGINAL image pixel coordinates.

    The backend is responsible for mapping model-space boxes back to the
    original image (see utils/preprocessing.py). The frontend only scales
    these values to its display size.
    """

    x1: float = Field(..., description="Left edge, original-image pixels")
    y1: float = Field(..., description="Top edge, original-image pixels")
    x2: float = Field(..., description="Right edge, original-image pixels")
    y2: float = Field(..., description="Bottom edge, original-image pixels")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection score")


class DetectionResponse(BaseModel):
    """Body of a successful POST /detect."""

    person_count: int = Field(..., ge=0, description="Number of persons detected")
    boxes: list[BoundingBox] = Field(default_factory=list)
    # None (not 0.0) when person_count == 0 — there is nothing to average.
    avg_confidence: Optional[float] = Field(
        default=None, description="Mean confidence of the returned boxes, or null"
    )
    inference_time_ms: float = Field(
        ..., description="Time spent inside the model only, milliseconds"
    )
    total_time_ms: float = Field(
        ..., description="Total server-side processing time, milliseconds"
    )
    image_width: int = Field(..., gt=0, description="Original image width, pixels")
    image_height: int = Field(..., gt=0, description="Original image height, pixels")
    annotated_image_base64: Optional[str] = Field(
        default=None,
        description="PNG with boxes drawn, base64-encoded. Present only if requested.",
    )


class DetectionRecord(BaseModel):
    """A stored detection as returned by GET /history."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
    person_count: int = Field(..., ge=0)
    avg_confidence: Optional[float] = None
    inference_time_ms: float


class HistoryResponse(BaseModel):
    """Body of a successful GET /history."""

    records: list[DetectionRecord]
    total: int = Field(..., description="Total rows matching the filters")
    returned: int = Field(..., description="Rows in this page")
    limit: int
    offset: int


class ResetResponse(BaseModel):
    """Body of a successful DELETE /reset."""

    deleted: int = Field(..., description="Number of rows removed")
    message: str


class ErrorDetail(BaseModel):
    """Machine-readable error code plus a human-readable message."""

    code: str = Field(..., description="Stable identifier, e.g. 'unsupported_format'")
    message: str = Field(..., description="Human-readable explanation")
    detail: Optional[str] = Field(default=None, description="Optional extra context")


class ErrorResponse(BaseModel):
    """Consistent error envelope for every non-2xx response."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Body of GET /health — cheap liveness probe that does not run inference."""

    status: str
    model_loaded: bool
    model_path: str


# ---------------------------------------------------------------------------
# Storage schema
# ---------------------------------------------------------------------------

# One row per successful detection. Timestamps are ISO-8601 UTC so they sort
# lexicographically and round-trip through datetime.fromisoformat().
_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,          -- ISO-8601 UTC
    time_of_day       TEXT    NOT NULL,          -- 'HH:MM', precomputed for filtering
    person_count      INTEGER NOT NULL CHECK (person_count >= 0),
    avg_confidence    REAL,                      -- NULL when person_count = 0
    inference_time_ms REAL    NOT NULL CHECK (inference_time_ms >= 0)
);

CREATE INDEX IF NOT EXISTS idx_detections_timestamp   ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_time_of_day ON detections(time_of_day);
CREATE INDEX IF NOT EXISTS idx_detections_avg_conf    ON detections(avg_confidence);
"""


def init_db(db_path: str) -> None:
    """Create the database file, its parent directory, and the schema.

    Idempotent: safe to call on every startup. WAL mode is enabled here so
    concurrent readers (history polling) do not block the writer (detect).
    """
    parent = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(parent, exist_ok=True)

    with _connect(db_path) as conn:
        # WAL is a persistent property of the database file; setting it once
        # is enough, but it is harmless to reassert on each connection.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(_SCHEMA)


@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    """Open a short-lived connection and commit on clean exit.

    Connections are per-operation because sqlite3 connections are not safe
    to share across the threads FastAPI may dispatch to.
    """
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row  # access columns by name
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Repository functions
# ---------------------------------------------------------------------------


def insert_detection(
    db_path: str,
    *,
    person_count: int,
    avg_confidence: Optional[float],
    inference_time_ms: float,
    timestamp: Optional[datetime] = None,
) -> int:
    """Insert one detection row and return its new id.

    Called only after inference has fully succeeded, so a failed detection
    never leaves a partial row (brief requirement).
    """
    ts = timestamp or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)

    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO detections
                (timestamp, time_of_day, person_count, avg_confidence, inference_time_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                ts.isoformat(),
                ts.strftime("%H:%M"),  # precomputed: avoids SQLite tz parsing
                int(person_count),
                avg_confidence,
                float(inference_time_ms),
            ),
        )
        return int(cursor.lastrowid)


def _build_filters(
    *,
    start: Optional[str],
    end: Optional[str],
    time_start: Optional[str],
    time_end: Optional[str],
    min_confidence: Optional[float],
) -> tuple[str, list[object]]:
    """Build a parameterised WHERE clause from the optional filters.

    Returns (where_sql, params). Kept separate from query_detections so the
    same predicate can be reused for the COUNT(*) query without duplication.
    """
    clauses: list[str] = []
    params: list[object] = []

    if start is not None:
        clauses.append("timestamp >= ?")
        params.append(start)
    if end is not None:
        clauses.append("timestamp <= ?")
        params.append(end)
    if time_start is not None:
        clauses.append("time_of_day >= ?")
        params.append(time_start)
    if time_end is not None:
        clauses.append("time_of_day <= ?")
        params.append(time_end)
    if min_confidence is not None:
        # NULL avg_confidence (zero-person rows) intentionally fails this test.
        clauses.append("avg_confidence >= ?")
        params.append(min_confidence)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def query_detections(
    db_path: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[DetectionRecord], int]:
    """Return (page of records, total matching the filters).

    Newest first. Filtering happens in SQL (decision D5) so the payload and
    the client stay small.
    """
    where, params = _build_filters(
        start=start,
        end=end,
        time_start=time_start,
        time_end=time_end,
        min_confidence=min_confidence,
    )

    with _connect(db_path) as conn:
        total = int(
            conn.execute(
                f"SELECT COUNT(*) AS n FROM detections {where}", params
            ).fetchone()["n"]
        )
        rows = conn.execute(
            f"""
            SELECT id, timestamp, person_count, avg_confidence, inference_time_ms
            FROM detections
            {where}
            ORDER BY timestamp DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            [*params, int(limit), int(offset)],
        ).fetchall()

    records = [
        DetectionRecord(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            person_count=row["person_count"],
            avg_confidence=row["avg_confidence"],
            inference_time_ms=row["inference_time_ms"],
        )
        for row in rows
    ]
    return records, total


def delete_all(db_path: str) -> int:
    """Delete every detection row and return how many were removed."""
    with _connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM detections")
        return int(cursor.rowcount if cursor.rowcount is not None else 0)
