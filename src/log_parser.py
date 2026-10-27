"""Day 3 - Log parser for IncidentInsight.

Stage 1 of the pipeline. Turns raw, unstructured log text into a structured
Pandas DataFrame. This is a PURE function: text in, table out. No database,
no UI, no network, no normalization (that is Day 4).

Expected raw line format (produced by data_generator.py):
    <TIMESTAMP> <LEVEL> [<request_id>] [<service>] <message>
example:
    2026-10-04T06:09:02Z FATAL [req-ba53cce8] [DatabaseService] Database connection pool exhausted...

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
#   timestamp : ISO-8601 UTC, e.g. 2026-10-04T06:09:02Z
#   level     : a run of letters (validated against VALID_LEVELS afterwards)
#   request_id: inside the first [ ... ]
#   service   : inside the second [ ... ]
#   message   : everything after the service bracket (".*" so brackets/colons
#               inside the message do not break the match)
LOG_LINE_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+"
    r"(?P<level>[A-Za-z]+)\s+"
    r"\[(?P<request_id>[^\]]+)\]\s+"
    r"\[(?P<service>[^\]]+)\]\s+"
    r"(?P<message>.*)$"
)


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

    return {
        "timestamp": match.group("timestamp"),
        "level": level,
        "service": match.group("service"),
        "request_id": match.group("request_id"),
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
