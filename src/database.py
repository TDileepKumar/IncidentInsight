"""SQLite persistence for IncidentInsight incident explanations.

A small, standalone module that saves the explanation dictionaries produced by
`src/incident_explainer.py` and reads them back across program runs.

Design rules:
  * Built-in `sqlite3` only - no new dependencies, no ORM, no migrations.
  * Importing this module has NO side effects: it never creates or opens a
    database until a function is called with an explicit `db_path`.
  * Structured fields (evidence, suggested_checks, limitations) are stored as
    JSON text - never pickled.
  * Every inserted/queried value uses a parameterized statement (no string
    interpolation), so SQL-looking log text is stored as inert data.
  * Writes are transactional: a batch either fully commits or fully rolls back.
  * Reads are read-only: `get_incidents` never creates a database or table; a
    missing file or table simply yields an empty list.
  * Persistence is always an explicit, caller-initiated operation. This module
    does not hook into `run_pipeline()` and does not re-run any analysis.

The `incidents` table has no uniqueness constraint on `cluster_id`: separate
runs (or repeated saves) may reuse a cluster ID, and each save is a distinct
row so analyses never overwrite one another.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

# --- Schema -------------------------------------------------------------------
TABLE_NAME = "incidents"

# Fields an incident mapping (src.incident_explainer.explain_incident output)
# must carry.
REQUIRED_INCIDENT_KEYS = (
    "cluster_id",
    "rank",
    "summary",
    "evidence",
    "suggested_checks",
    "limitations",
)

# `cluster_id` is declared without a type affinity so an integer ID stays an
# integer and a string ID stays a string on round-trip. `rank` is nullable
# because src.incident_explainer uses None when no ranking position was supplied.
# `rank` is a SQLite keyword (window functions), so it is always quoted.
_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id             NOT NULL,
    "rank"                 INTEGER,
    summary                TEXT    NOT NULL,
    evidence_json          TEXT    NOT NULL,
    suggested_checks_json  TEXT    NOT NULL,
    limitations_json       TEXT    NOT NULL,
    created_at             TEXT    NOT NULL
)
"""

_INSERT_SQL = f"""
INSERT INTO {TABLE_NAME}
    (cluster_id, "rank", summary, evidence_json,
     suggested_checks_json, limitations_json, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

# Newest first: timestamp desc, then id desc as the deterministic tie-breaker
# for records stored within the same batch (which share one created_at).
_SELECT_SQL = f"""
SELECT id, cluster_id, "rank", summary,
       evidence_json, suggested_checks_json, limitations_json, created_at
FROM {TABLE_NAME}
ORDER BY created_at DESC, id DESC
LIMIT ? OFFSET ?
"""

_TABLE_EXISTS_SQL = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?"


# --- Small internal helpers ---------------------------------------------------
def _resolve_path(db_path) -> Path:
    """Coerce `db_path` to a Path, rejecting non-path types and blank paths."""
    if not isinstance(db_path, (str, os.PathLike)):
        raise TypeError(
            f"db_path must be a string or path-like object, got {type(db_path).__name__}"
        )
    path = Path(os.fspath(db_path))
    if str(path).strip() == "":
        raise ValueError("db_path must be a non-empty path")
    return path


def _prepare_parent(path: Path) -> None:
    """Create the parent directory when there is a real one to create.

    A bare filename (parent == '.') is NOT treated as a missing directory, so we
    never attempt to create the current working directory.
    """
    parent = path.parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    """A timezone-aware UTC timestamp in ISO-8601 form (stable, sortable)."""
    return datetime.now(timezone.utc).isoformat()


def _is_null(value) -> bool:
    """True for None or a NaN float - both mean 'no value supplied'."""
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _table_exists(conn: sqlite3.Connection, name: str = TABLE_NAME) -> bool:
    """Whether the given table is present in the connected database."""
    return conn.execute(_TABLE_EXISTS_SQL, (name,)).fetchone() is not None


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the incidents table if absent (idempotent)."""
    conn.execute(_CREATE_TABLE_SQL)


def _validate_limit(limit) -> int | None:
    """Return a positive int limit, or None for 'no limit'; reject bools."""
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, numbers.Integral):
        raise ValueError(f"limit must be a positive integer or None, got {type(limit).__name__}")
    value = int(limit)
    if value <= 0:
        raise ValueError(f"limit must be > 0, got {value}")
    return value


def _validate_offset(offset) -> int:
    """Return a non-negative int offset; reject bools and negatives."""
    if isinstance(offset, bool) or not isinstance(offset, numbers.Integral):
        raise ValueError(f"offset must be a non-negative integer, got {type(offset).__name__}")
    value = int(offset)
    if value < 0:
        raise ValueError(f"offset must be >= 0, got {value}")
    return value


def _coerce_id(value, field: str, index: int):
    """Normalize an identifier/rank value to a SQLite-bindable scalar."""
    if isinstance(value, bool):
        raise TypeError(f"incident at index {index}: '{field}' must not be a bool")
    if isinstance(value, numbers.Integral):
        return int(value)
    return value


def _build_row(incident, index: int, created_at: str) -> tuple:
    """Validate one incident mapping and return its INSERT parameter tuple.

    Raises TypeError for wrong types / non-serializable JSON content and
    ValueError for missing keys or a null cluster_id. No value is invented.
    """
    if not isinstance(incident, Mapping):
        raise TypeError(
            f"incident at index {index} must be a mapping, got {type(incident).__name__}"
        )

    missing = [k for k in REQUIRED_INCIDENT_KEYS if k not in incident]
    if missing:
        raise ValueError(f"incident at index {index} is missing required field(s): {missing}")

    cluster_id = incident["cluster_id"]
    if _is_null(cluster_id):
        raise ValueError(f"incident at index {index} has a null or missing 'cluster_id'")
    cluster_id = _coerce_id(cluster_id, "cluster_id", index)

    rank = incident["rank"]
    if _is_null(rank):
        rank = None
    else:
        rank = _coerce_id(rank, "rank", index)

    summary = incident["summary"]
    if not isinstance(summary, str):
        raise TypeError(f"incident at index {index}: 'summary' must be a string")

    evidence = incident["evidence"]
    if not isinstance(evidence, Mapping):
        raise TypeError(f"incident at index {index}: 'evidence' must be a mapping")

    suggested_checks = incident["suggested_checks"]
    if not isinstance(suggested_checks, (list, tuple)):
        raise TypeError(f"incident at index {index}: 'suggested_checks' must be a list")

    limitations = incident["limitations"]
    if not isinstance(limitations, (list, tuple)):
        raise TypeError(f"incident at index {index}: 'limitations' must be a list")

    # Serialize BEFORE any database write so a non-serializable value fails the
    # whole batch without touching the database (no partial commit).
    evidence_json = json.dumps(dict(evidence), ensure_ascii=False)
    checks_json = json.dumps(list(suggested_checks), ensure_ascii=False)
    limitations_json = json.dumps(list(limitations), ensure_ascii=False)

    return (
        cluster_id,
        rank,
        summary,
        evidence_json,
        checks_json,
        limitations_json,
        created_at,
    )


def _row_to_record(row: tuple) -> dict:
    """Decode a SELECT row back into a Python dictionary."""
    return {
        "id": row[0],
        "cluster_id": row[1],
        "rank": row[2],
        "summary": row[3],
        "evidence": json.loads(row[4]),
        "suggested_checks": json.loads(row[5]),
        "limitations": json.loads(row[6]),
        "created_at": row[7],
    }


# --- Public API ---------------------------------------------------------------
def initialize_database(db_path) -> None:
    """Create the database file (if needed) and the incidents table.

    Safe to call repeatedly: the table is created only if it does not exist, so
    previously stored records are never destroyed. The parent directory is
    created when the path has a real one. The connection is always closed.

    Raises:
        TypeError: `db_path` is not a string / path-like object.
        ValueError: `db_path` is blank.
        sqlite3.Error: the path is unusable (e.g. points at a directory).
    """
    path = _resolve_path(db_path)
    _prepare_parent(path)
    conn = sqlite3.connect(str(path))
    try:
        with conn:  # commit on success, roll back on error
            _ensure_schema(conn)
    finally:
        conn.close()


def save_incidents(incidents, db_path) -> int:
    """Persist a collection of src.incident_explainer mappings; return the count.

    Args:
        incidents: an iterable of mappings, each shaped like the dict returned by
            `explain_incident()` (cluster_id, rank, summary, evidence,
            suggested_checks, limitations). A single mapping, a string, or a
            non-iterable is rejected.
        db_path: target SQLite file. The table is created if absent, so calling
            `initialize_database` first is optional.

    Returns:
        The number of records stored. An empty collection stores nothing and
        returns 0 (no fabricated records).

    All rows are inserted in ONE transaction: if any row is invalid or a write
    fails, nothing from that batch is committed. Incidents are never
    deduplicated by `cluster_id` - repeated IDs create separate rows. No
    clustering, ranking, or explanation is regenerated here.

    Raises:
        TypeError: `incidents` is not a collection of mappings, a field has a
            wrong type, or a value is not JSON-serializable.
        ValueError: `db_path` is blank, an incident is missing required fields,
            or a cluster_id is null.
    """
    path = _resolve_path(db_path)

    if isinstance(incidents, (str, bytes, Mapping)):
        raise TypeError(
            "incidents must be a collection of mappings (e.g. a list of "
            f"explain_incident() dicts), got {type(incidents).__name__}"
        )
    try:
        items = list(incidents)
    except TypeError:
        raise TypeError(
            f"incidents must be an iterable collection, got {type(incidents).__name__}"
        ) from None

    if not items:
        return 0

    # Validate + serialize everything first; this raises before any DB write.
    created_at = _utc_now_iso()
    rows = [_build_row(item, i, created_at) for i, item in enumerate(items)]

    _prepare_parent(path)
    conn = sqlite3.connect(str(path))
    try:
        with conn:  # single transaction for the whole batch
            _ensure_schema(conn)
            conn.executemany(_INSERT_SQL, rows)
    finally:
        conn.close()

    return len(rows)


def get_incidents(db_path, limit: int | None = None, offset: int = 0) -> list[dict]:
    """Retrieve saved incidents, newest first; JSON fields decoded to Python.

    Args:
        db_path: the SQLite file to read.
        limit: maximum number of records (positive int) or None for all.
        offset: number of newest records to skip (non-negative int).

    Returns:
        A list of dicts, each with keys: id, cluster_id, rank, summary,
        evidence (dict), suggested_checks (list), limitations (list),
        created_at. Ordered by created_at DESC then id DESC. An empty list is
        returned when there are no records.

    Read-only: if the database file or the incidents table does not exist, an
    empty list is returned and NOTHING is created. Booleans are rejected for
    limit/offset. The connection is always closed and never exposed.

    Raises:
        TypeError: `db_path` is not a string / path-like object.
        ValueError: `db_path` is blank, or limit/offset is invalid.
    """
    path = _resolve_path(db_path)
    limit_value = _validate_limit(limit)
    offset_value = _validate_offset(offset)

    if not path.exists():
        return []  # documented: reads never create a database

    conn = sqlite3.connect(str(path))
    try:
        if not _table_exists(conn):
            return []
        # SQLite treats a negative LIMIT as "no limit"; bind both as parameters.
        sql_limit = -1 if limit_value is None else limit_value
        cursor = conn.execute(_SELECT_SQL, (sql_limit, offset_value))
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [_row_to_record(row) for row in rows]
