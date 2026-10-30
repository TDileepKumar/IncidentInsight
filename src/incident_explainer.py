"""Deterministic, template-based incident explanations.

Turns an ALREADY-RANKED incident record (a row of ranker.rank_clusters() output)
into a structured, developer-friendly explanation. This module is PURE and
OFFLINE: no LLM, no network, no clustering, no ranking, no persistence. It only
reformats supplied data and applies fixed keyword rules.

Core honesty rules:
  * Evidence before interpretation. Observations (message, count, scores) are
    kept separate from the pattern label and from investigation suggestions.
  * Nothing is invented. A missing rank stays None; a missing count is simply
    not reported; absent severity is never fabricated.
  * Cluster IDs are opaque identifiers - no severity/importance is inferred
    from them.
  * Ranking scores are HEURISTIC algorithm outputs, never confirmed business
    impact, and the suggested checks are investigation prompts, NOT verified
    root causes.
  * Deterministic: identical input always yields identical output.

Input representation: one incident is a Mapping (e.g. a dict from
DataFrame.to_dict("records")) or a pandas Series (a DataFrame row). Field names
match the real ranking schema in src/ranker.py: cluster_id, size, rank,
representative_message, severity_score, frequency_score, risk_score.

Validation policy (explicit and consistent):
  * `cluster_id` is REQUIRED; a missing/null identifier raises ValueError.
  * A field that is absent or null is treated as UNAVAILABLE and never invented.
  * A field that is PRESENT but malformed (e.g. a negative or non-numeric
    count, a non-integer rank, a non-string message) raises ValueError rather
    than being silently coerced or dropped.
  * Optional score fields are included only when present and finite; an invalid
    score is omitted (scores are supplementary enrichment, not structural).
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping

import pandas as pd

from src.clusterer import CLUSTER_COLUMN

# --- Field names (match src/ranker.py OUTPUT_COLUMNS) -------------------------
CLUSTER_ID_FIELD = CLUSTER_COLUMN          # required identifier
RANK_FIELD = "rank"                        # optional ranking position
COUNT_FIELD = "size"                       # optional occurrence count
MESSAGE_FIELD = "representative_message"   # optional evidence text
SCORE_FIELDS = ("severity_score", "frequency_score", "risk_score")  # optional

# --- Investigation-suggestion templates (fixed, deterministic) ----------------
_DB_CONNECTION_CHECKS = [
    "Verify database server availability and network reachability.",
    "Check database connection settings (host, port, credentials).",
    "Inspect connection-pool capacity and saturation.",
]
_DB_TIMEOUT_CHECKS = [
    "Review slow-query logs and query execution plans.",
    "Check database response times and current load.",
    "Verify query and connection timeout configuration.",
]
_PAYMENT_CHECKS = [
    "Check payment gateway provider availability and status.",
    "Verify network connectivity to the payment gateway.",
    "Review gateway timeout and retry configuration.",
]
_AUTH_CHECKS = [
    "Inspect authentication configuration and stored credentials.",
    "Review authentication and authorization logs for the affected users.",
    "Check for expired tokens or a misconfigured identity provider.",
]
_FILE_CHECKS = [
    "Verify the file path and that the file exists.",
    "Check filesystem mounts and working-directory configuration.",
    "Review file access permissions.",
]
_REQUEST_TIMEOUT_CHECKS = [
    "Inspect request latency and downstream dependency response times.",
    "Check timeout and retry settings along the request path.",
    "Review load on the upstream and downstream services.",
]
_NULL_POINTER_CHECKS = [
    "Inspect the code path for unchecked null/None references.",
    "Review recent changes to the affected component.",
    "Add or verify input guards before the failing dereference.",
]
_VALIDATION_CHECKS = [
    "Review the input validation rules and the rejected payload.",
    "Check upstream data and schema conformance.",
    "Inspect validation error handling and logging.",
]
# Documented fallback policy: suggested_checks is ALWAYS non-empty. An unknown or
# missing message yields these generic prompts and asserts no specific category.
GENERIC_CHECKS = [
    "Review the representative log message and surrounding log context to "
    "identify the failing component.",
    "Correlate the incident with recent deployments or configuration changes.",
]


def _has_any(text: str, *needles: str) -> bool:
    """True if any needle is a substring of `text` (already lowercased)."""
    return any(n in text for n in needles)


# Ordered rule table; the FIRST matching rule wins, so more specific patterns are
# listed before broader ones (e.g. database timeout before database connection).
# Each rule is (label, matcher, checks).
_PATTERN_RULES = [
    ("payment gateway error", lambda m: "payment" in m, _PAYMENT_CHECKS),
    (
        "authentication failure",
        lambda m: _has_any(m, "authentication", "unauthorized", "invalid credentials", "login"),
        _AUTH_CHECKS,
    ),
    (
        "file-not-found error",
        lambda m: _has_any(m, "file not found", "no such file", "missing file", "filenotfound")
        or ("file" in m and "not found" in m),
        _FILE_CHECKS,
    ),
    (
        "null pointer",
        lambda m: _has_any(m, "null pointer", "nullpointer", "null reference", "nonetype"),
        _NULL_POINTER_CHECKS,
    ),
    (
        "database timeout / slow query",
        lambda m: "slow query" in m
        or (_has_any(m, "timeout", "timed out") and _has_any(m, "database", "query", " db")),
        _DB_TIMEOUT_CHECKS,
    ),
    (
        "database connection failure",
        lambda m: _has_any(m, "database", " db", "connection pool")
        and _has_any(m, "connect", "connection"),
        _DB_CONNECTION_CHECKS,
    ),
    (
        "validation error",
        lambda m: _has_any(m, "validation", "invalid input", "invalid payload", "malformed", "schema"),
        _VALIDATION_CHECKS,
    ),
    (
        "request timeout",
        lambda m: _has_any(m, "timeout", "timed out")
        and _has_any(m, "request", "upstream", "client", "gateway"),
        _REQUEST_TIMEOUT_CHECKS,
    ),
]


def _is_missing(value) -> bool:
    """True when a supplied value should be treated as unavailable (None/NaN)."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _coerce_count(value, field: str) -> int:
    """Validate a present occurrence count: a finite, non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"incident '{field}' must be a non-negative integer, got {value!r}")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"incident '{field}' must be finite, got {value!r}")
    if num < 0:
        raise ValueError(f"incident '{field}' must be non-negative, got {num}")
    if not num.is_integer():
        raise ValueError(f"incident '{field}' must be a whole number, got {value!r}")
    return int(num)


def _coerce_rank(value, field: str) -> int:
    """Validate a present rank: a finite integer (never invented if absent)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"incident '{field}' must be an integer, got {value!r}")
    num = float(value)
    if not math.isfinite(num) or not num.is_integer():
        raise ValueError(f"incident '{field}' must be a finite integer, got {value!r}")
    return int(num)


def _as_mapping(incident) -> dict:
    """Accept a Mapping or pandas Series; return a plain dict copy."""
    if isinstance(incident, pd.Series):
        return incident.to_dict()
    if isinstance(incident, Mapping):
        return dict(incident)
    raise TypeError(
        "incident must be a mapping or pandas Series (a ranking row), "
        f"got {type(incident).__name__}"
    )


def _select_pattern(message: str | None):
    """Return (label, checks) for a message; (None, GENERIC_CHECKS) if unknown."""
    if message is None:
        return None, list(GENERIC_CHECKS)
    lowered = message.lower()
    for label, matcher, checks in _PATTERN_RULES:
        if matcher(lowered):
            return label, list(checks)
    return None, list(GENERIC_CHECKS)


def explain_incident(incident) -> dict:
    """Build a deterministic explanation dict for ONE ranked incident.

    Args:
        incident: a Mapping or pandas Series holding one ranking row. Recognized
            fields (all optional except cluster_id): cluster_id, rank, size,
            representative_message, severity_score, frequency_score, risk_score.

    Returns:
        A dict with keys: cluster_id, rank, summary, evidence, suggested_checks,
        limitations. `rank` is None when not supplied. `evidence` contains only
        the fields that were genuinely available. `suggested_checks` and
        `limitations` are always non-empty lists.

    Raises:
        TypeError: `incident` is not a Mapping or Series.
        ValueError: cluster_id is missing/null, or a present field is malformed
            (bad count, rank, or message type).
    """
    data = _as_mapping(incident)

    # --- cluster_id: required, opaque identifier ------------------------------
    if _is_missing(data.get(CLUSTER_ID_FIELD)):
        raise ValueError(f"incident must include a valid '{CLUSTER_ID_FIELD}'")
    cluster_id = data.get(CLUSTER_ID_FIELD)

    # --- rank: include only when genuinely available --------------------------
    rank_raw = data.get(RANK_FIELD)
    rank = None if _is_missing(rank_raw) else _coerce_rank(rank_raw, RANK_FIELD)

    # --- occurrence count: include only when available and valid --------------
    count_raw = data.get(COUNT_FIELD)
    count = None if _is_missing(count_raw) else _coerce_count(count_raw, COUNT_FIELD)

    # --- representative message ----------------------------------------------
    msg_raw = data.get(MESSAGE_FIELD)
    if _is_missing(msg_raw):
        message = None
    elif not isinstance(msg_raw, str):
        raise ValueError(f"incident '{MESSAGE_FIELD}' must be a string, got {type(msg_raw).__name__}")
    else:
        message = msg_raw if msg_raw.strip() else None

    label, suggested_checks = _select_pattern(message)

    # --- evidence: observed facts only, present fields only -------------------
    evidence: dict = {}
    if message is not None:
        evidence[MESSAGE_FIELD] = message
    if count is not None:
        evidence["occurrence_count"] = count
    for field in SCORE_FIELDS:
        value = data.get(field)
        if _is_missing(value) or isinstance(value, bool):
            continue
        if isinstance(value, numbers.Real) and math.isfinite(float(value)):
            evidence[field] = float(value)

    # --- summary: factual, no impact claims ----------------------------------
    head = f"Cluster {cluster_id}"
    if rank is not None:
        head += f" (rank {rank})"
    head += (
        f" matched a '{label}' pattern" if label
        else " matched an unrecognized incident pattern"
    )
    if count is not None:
        head += f"; observed {count} time(s)"
    summary = head + "."

    # --- limitations: explicit about what is NOT known ------------------------
    limitations = [
        "Root cause is not confirmed; suggested_checks are investigation prompts "
        "derived from log message text, not verified causes.",
    ]
    if label is None:
        limitations.append(
            "The message did not match a known incident pattern, so no specific "
            "category is asserted."
        )
    if message is None:
        limitations.append(
            "No representative message was supplied, so pattern-based checks are generic."
        )
    if count is None:
        limitations.append("Occurrence count was not supplied and is not inferred.")
    if rank is None:
        limitations.append("Ranking position was not supplied and is not inferred.")
    if "severity_score" not in evidence:
        limitations.append("No severity information was supplied; none is inferred.")
    if any(f in evidence for f in SCORE_FIELDS):
        limitations.append(
            "Ranking scores (severity/frequency/risk) are heuristic algorithm "
            "outputs, not confirmed real-world or business impact."
        )

    return {
        "cluster_id": cluster_id,
        "rank": rank,
        "summary": summary,
        "evidence": evidence,
        "suggested_checks": suggested_checks,
        "limitations": limitations,
    }


def explain_ranked_incidents(ranking_df: pd.DataFrame) -> list[dict]:
    """Explain every incident in a ranked DataFrame, preserving input order.

    Args:
        ranking_df: the ranking output of ranker.rank_clusters() (or
            PipelineResult.ranking). Must contain a `cluster_id` column; all
            other recognized fields are optional.

    Returns:
        A list of explanation dicts (see explain_incident), one per input row,
        in the SAME order as the input. An empty DataFrame yields an empty list.

    Raises:
        TypeError: `ranking_df` is not a pandas DataFrame.
        ValueError: the required `cluster_id` column is missing, or any row is
            malformed (propagated from explain_incident so no row is silently
            dropped).

    The input DataFrame is never mutated, reordered, or re-scored, and no
    parsing/normalization/vectorization/clustering/ranking is performed.
    """
    if not isinstance(ranking_df, pd.DataFrame):
        raise TypeError(
            f"ranking_df must be a pandas DataFrame, got {type(ranking_df).__name__}"
        )
    if CLUSTER_ID_FIELD not in ranking_df.columns:
        raise ValueError(f"ranking_df is missing required column '{CLUSTER_ID_FIELD}'")
    if len(ranking_df) == 0:
        return []

    # to_dict("records") copies values and preserves row order; the source is
    # never modified. Each record is explained with the same rules as a single
    # incident, so a malformed row raises instead of being silently dropped.
    return [explain_incident(record) for record in ranking_df.to_dict("records")]
