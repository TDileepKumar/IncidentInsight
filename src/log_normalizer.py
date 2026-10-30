"""Log normalizer for IncidentInsight.

Stage 2 of the pipeline. Takes the parsed `message` text and replaces the
VARIABLE tokens (ids, numbers, hosts, paths, quoted strings, ...) with fixed
placeholders, so that the same underlying error produces the SAME string.
This is what makes TF-IDF + clustering viable (src/feature_extractor.py,
src/clusterer.py).

Design decisions:
  * add a NEW column `message_normalized`; the original `message` is preserved
  * only `message_normalized` is fed to later TF-IDF/clustering
  * host+port collapses to a single `<HOST>`
  * regex rules live here (not in config)
  * angle-bracket placeholders: <NUM>, <HOST>, <PATH>, <ID>, ...

This module is PURE: it does not read files, touch the DB, or call the network.
"""

from __future__ import annotations

import re

import pandas as pd

# Name of the column this stage adds.
NORMALIZED_COLUMN = "message_normalized"

# --- Ordered normalization rules ---------------------------------------------
# ORDER IS CRITICAL: specific/structured patterns run BEFORE the generic numeric
# pattern, otherwise the numeric rule would shred tokens we want to keep whole.
#
# Example of why order matters: "PaymentController.java:57" also matches the
# host:port pattern. Because the <LOC> rule runs first, it is replaced before
# the host rule ever sees it. (Verified before writing this file.)
#
# We deliberately DO NOT normalize meaningful words (class names like
# "PaymentController", service names, "NullPointerException", "timed out").
# Over-normalizing would merge genuinely different error types. Residual word
# differences are left for TF-IDF + cosine similarity to handle.
_RULES: list[tuple[re.Pattern, str]] = [
    # 1) Source location  file.ext:line  -> <LOC>   (before host:port!)
    (re.compile(r"\b[\w$.]+\.(?:java|py|js|ts|go|rb|cs|c|cpp|h|kt|php|rs):\d+\b"), "<LOC>"),
    # 2) host:port  -> <HOST>
    (re.compile(r"\b[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)+:\d{2,5}\b"), "<HOST>"),
    # 3) bare hostname with a known suffix  -> <HOST>
    (re.compile(r"\b(?:[a-zA-Z0-9\-]+\.)+(?:internal|local|localdomain|com|net|org|io)\b"), "<HOST>"),
    # 4) IPv4 address  -> <IP>
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<IP>"),
    # 5) unix-like path  -> <PATH>   (before numbers: paths contain digits like /v1/)
    (re.compile(r"(?:/[\w.\-]+)+/?"), "<PATH>"),
    # 6) quoted strings  -> <STR>
    (re.compile(r"'[^']*'"), "<STR>"),
    (re.compile(r'"[^"]*"'), "<STR>"),
    # 7) long hex ids (query_id, transaction, order)  -> <ID>   (before numbers)
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<ID>"),
    # 8) username following "for user "  -> <USER>   (before numbers)
    (re.compile(r"(?<=for user )[A-Za-z][\w.\-]*"), "<USER>"),
    # 9) GENERIC numbers (optional decimal), left to last  -> <NUM>
    (re.compile(r"\d+(?:\.\d+)?"), "<NUM>"),
]

_MULTI_SPACE = re.compile(r"\s{2,}")


def normalize_message(text: "str | None") -> str:
    """Normalize one message string. Safe on None / empty input.

    Idempotent: placeholders contain no digits, quotes, dots or slashes, so
    running the rules a second time changes nothing.
    """
    if text is None:
        return ""
    s = str(text)
    if not s.strip():
        return ""
    for pattern, replacement in _RULES:
        s = pattern.sub(replacement, s)
    return _MULTI_SPACE.sub(" ", s).strip()


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a COPY of `df` with a new `message_normalized` column.

    * The original `message` and every metadata column are left unchanged.
    * Rows the parser could not parse (parsed=False) pass their raw text
      through untouched - we never mangle data we failed to understand.
    * The new column is placed immediately after `message`.
    """
    out = df.copy()

    def _normalize_row(row: pd.Series) -> str:
        message = row.get("message", "")
        if not bool(row.get("parsed", True)):
            return "" if message is None else str(message)
        return normalize_message(message)

    out[NORMALIZED_COLUMN] = out.apply(_normalize_row, axis=1)

    # Reorder so message_normalized sits right after message.
    columns = [c for c in out.columns if c != NORMALIZED_COLUMN]
    insert_at = columns.index("message") + 1 if "message" in columns else len(columns)
    columns.insert(insert_at, NORMALIZED_COLUMN)
    return out[columns]
