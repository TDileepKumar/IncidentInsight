"""Day 3 - Log parser for IncidentInsight.

Stage 1 of the pipeline. Turns raw, unstructured log text into a structured
Pandas DataFrame. This is a PURE function: text in, table out. No database,
no UI, no network, no normalization (that is Day 4).

Expected raw line formats:
    <TIMESTAMP> <LEVEL> [<request_id>] [<service>] <message>
    <TIMESTAMP> <LEVEL> [<service>] <message>          (no request id in the log)
examples:
    2026-10-04T06:09:02Z FATAL [req-ba53cce8] [DatabaseService] Database connection pool exhausted...
    2026-10-26 10:07:30 ERROR [cache-service] Redis connection refused...
TIMESTAMP accepts ISO-8601 with either a 'T' or a space before the time, optional
fractional seconds, and an optional UTC suffix ('Z' or a numeric offset). Fields
the line does not carry are left as None - never invented.

Design decisions (approved Day 3):
  * Columns: line_id, timestamp, level, service, request_id, message, parsed
  * timestamps stay ISO-format STRINGS (nothing needs datetime yet)
  * malformed / unrecognized lines are KEPT with parsed=False and the original
    text preserved in `message` - we never silently drop data
  * the line regex and the valid severity set live here, not in config
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Column order for the output DataFrame.
COLUMNS = ["line_id", "timestamp", "level", "service", "request_id", "message", "parsed"]

# Severity levels we recognize. A line that is otherwise well-formed but uses an
# unknown level is treated as unparsed (parsed=False) rather than guessed at.
VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "FATAL"}

# One regex with NAMED capture groups -> each (?P<name>...) becomes a field.
#   timestamp : ISO-8601, with 'T' or a space before the time, optional
#               fractional seconds, optional 'Z'/numeric UTC offset
#   level     : a run of letters (validated against VALID_LEVELS afterwards)
#   first     : inside the first [ ... ] after the level, if any
#   second    : inside the second [ ... ], if any
#   message   : everything after the bracket(s) (".*" so brackets/colons inside
#               the message do not break the match)
# Both brackets are OPTIONAL because real logs vary: the bundled dataset always
# has [request_id] [service], while other formats carry only [service].
_LOG_TIME = (
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"(?:[.,]\d{1,9})?"
    r"(?:Z|[+-]\d{2}:?\d{2})?"
)

LOG_LINE_PATTERN = re.compile(
    rf"^(?P<timestamp>{_LOG_TIME})\s+"
    r"(?P<level>[A-Za-z]+)\s+"
    r"(?:\[(?P<first>[^\]]+)\](?:\s+|$))?"
    r"(?:\[(?P<second>[^\]]+)\](?:\s+|$))?"
    r"(?P<message>.*)$"
)

# A lone bracket is a request id when it announces itself as one; otherwise it
# is the service name (the far more common single-bracket layout).
_REQUEST_ID_PREFIXES = (
    "req-",
    "req_",
    "req:",
    "request-",
    "request_",
    "request:",
    "trace-",
    "trace_",
    "trace:",
)
_REQUEST_ID_KEYS = ("request_id=", "request-id=", "request id=", "trace_id=", "trace-id=", "trace id=")
_UUID_PATTERN = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_LONG_HEX_PATTERN = re.compile(r"[0-9a-fA-F]{8,}")


def _looks_like_request_id(value: str) -> bool:
    """Whether a lone bracket's contents identify a request rather than a service."""
    token = value.strip()
    lowered = token.lower()
    if lowered.startswith(_REQUEST_ID_PREFIXES):
        return True
    if any(key in lowered for key in _REQUEST_ID_KEYS):
        return True
    return bool(_UUID_PATTERN.fullmatch(token) or _LONG_HEX_PATTERN.fullmatch(token))


def _unparsed(line: str) -> dict:
    """Build a record for a line we could not parse, preserving the raw text."""
    return {
        "timestamp": None,
        "level": None,
        "service": None,
        "request_id": None,
        "message": line,
        "parsed": False,
    }


def parse_line(line: str) -> dict:
    """Parse a single log line into a field dict.

    Returns parsed=True with all fields when the line matches the expected
    format AND uses a known severity level; otherwise parsed=False with the
    original text kept in `message`.
    """
    match = LOG_LINE_PATTERN.match(line.rstrip())
    if not match:
        return _unparsed(line)

    level = match.group("level").upper()
    if level not in VALID_LEVELS:
        return _unparsed(line)

    first = match.group("first")
    second = match.group("second")
    if second is not None:
        # Documented two-bracket layout: [request_id] [service].
        request_id, service = first, second
    elif first is not None:
        if _looks_like_request_id(first):
            request_id, service = first, None
        else:
            request_id, service = None, first
    else:
        request_id, service = None, None

    return {
        "timestamp": match.group("timestamp"),
        "level": level,
        "service": service,
        "request_id": request_id,
        "message": match.group("message").strip(),
        "parsed": True,
    }


def parse_logs(text: str) -> pd.DataFrame:
    """Parse a multi-line log string into a DataFrame.

    Blank lines are skipped. `line_id` is the 0-based position among non-blank
    lines, so for a generated file with no blanks it matches ground_truth.csv.
    """
    records = []
    line_id = 0
    for raw in text.splitlines():
        if not raw.strip():
            continue
        record = parse_line(raw)
        record["line_id"] = line_id
        records.append(record)
        line_id += 1

    return pd.DataFrame(records, columns=COLUMNS)


def parse_file(path: "str | Path") -> pd.DataFrame:
    """Read a log file (UTF-8) and parse it into a DataFrame."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_logs(text)
