"""UI support helpers for the Streamlit application (``app.py``).

Three concerns live here instead of in ``app.py``:

  1. Turning the user's log input (an uploaded file, pasted text, or a path) into
     something ``run_workflow()`` can consume, with honest validation.
  2. Turning the real ranking output into the small tables the dashboard charts.
  3. Resolving the cluster-detail view (``app.py``) against an analysis result that
     is already in memory: which clusters are selectable, which log rows belong to
     the selected cluster, and which explanation goes with it.

This module deliberately imports NO Streamlit. ``app.py`` only renders what is
returned here, which keeps the validation and chart-data logic unit-testable
without a browser, a server, or Streamlit installed.

Nothing here runs the ML pipeline, and nothing here is written to the project
tree: pasted/uploaded logs go to a private temporary file that the caller
deletes as soon as the analysis result is in memory.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd

from src import config
from src.ranker import DEFAULT_SEVERITY_WEIGHTS, MAX_SEVERITY_WEIGHT

# --- Input handling -----------------------------------------------------------
SUPPORTED_LOG_EXTENSIONS = (".log", ".txt")
MAX_INPUT_BYTES = 20 * 1024 * 1024   # 20 MB is far beyond any demo log file
_TEMP_FILE_PREFIX = "incidentinsight-"

# Worst-first severity ladder, used as the chart's fixed category axis.
SEVERITY_ORDER = tuple(
    label for label, _ in sorted(DEFAULT_SEVERITY_WEIGHTS.items(), key=lambda item: (item[1], item[0]))
)
_LABEL_BY_WEIGHT = {float(weight): label for label, weight in DEFAULT_SEVERITY_WEIGHTS.items()}

DEFAULT_CHART_TOP_N = 10
_COUNT_COLUMN = "log_count"
_SEVERITY_COLUMN = "incident_clusters"

# Cluster detail view (app.py): the log-row columns shown for a selected cluster,
# in display order. Only the ones that actually exist are used, so an older or
# trimmed DataFrame degrades instead of raising.
CLUSTER_DETAIL_COLUMNS = (
    "line_id",
    "timestamp",
    "level",
    "service",
    "request_id",
    "message",
    "message_normalized",
)
DETAIL_PREVIEW_ROWS = 25


class LogInputError(ValueError):
    """A problem with the log input that the UI can show the user verbatim."""


def _reject_oversize(size: int, what: str) -> None:
    if size > MAX_INPUT_BYTES:
        limit_mb = MAX_INPUT_BYTES // (1024 * 1024)
        raise LogInputError(f"{what} is larger than the {limit_mb} MB limit. Upload a smaller log file.")


def decode_log_bytes(data) -> tuple[str, "str | None"]:
    """Decode uploaded bytes into log text.

    Returns ``(text, warning)``. The warning is not None when the file was not
    valid UTF-8: rather than refusing a log the user can plainly read, undecodable
    bytes are replaced and the UI says so. Raises ``LogInputError`` for empty or
    oversized input.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise LogInputError("The uploaded file could not be read. Try again, or paste the log text instead.")
    raw = bytes(data)
    if not raw.strip():
        raise LogInputError("The uploaded file is empty - it contains no log lines.")
    _reject_oversize(len(raw), "The uploaded file")
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), (
            "The file was not valid UTF-8. Undecodable bytes were replaced, so a few "
            "characters in the log text may look wrong."
        )


def validate_uploaded_log(name, data) -> tuple[str, "str | None"]:
    """Validate an uploaded file's type and content; return ``(text, warning)``."""
    extension = os.path.splitext(str(name or ""))[1].lower()
    if extension not in SUPPORTED_LOG_EXTENSIONS:
        shown = extension or "unknown"
        allowed = ", ".join(SUPPORTED_LOG_EXTENSIONS)
        raise LogInputError(f"Unsupported file type '{shown}'. Upload a plain-text log file ({allowed}).")
    return decode_log_bytes(data)


def validate_pasted_log(text) -> tuple[str, "str | None"]:
    """Validate pasted log text; return ``(text, warning)``."""
    if not isinstance(text, str) or not text.strip():
        raise LogInputError("Nothing was pasted. Add at least one log line, or choose another input method.")
    _reject_oversize(len(text.encode("utf-8")), "The pasted log text")
    return text.strip(), None


def resolve_log_path(log_path) -> str:
    """Validate the path input and return it trimmed (existence is the pipeline's job)."""
    if not isinstance(log_path, str) or not log_path.strip():
        raise LogInputError("Enter a log file path, or choose another input method.")
    return log_path.strip()


def write_temp_log(text: str) -> str:
    """Write log text to a private temporary file and return its path.

    The caller MUST pass the result to ``remove_temp_log`` once the analysis
    result is in memory. Uploads are never written into the project tree or any
    other permanent location.
    """
    handle, path = tempfile.mkstemp(prefix=_TEMP_FILE_PREFIX, suffix=".log", text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    except BaseException:
        remove_temp_log(path)
        raise
    return path


def remove_temp_log(path) -> None:
    """Delete a temporary log file. Missing/undeletable files are not an error."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


# --- Chart data ---------------------------------------------------------------
def _as_frame(value, name: str) -> pd.DataFrame:
    """Accept a DataFrame (or None) and return it, or an empty frame."""
    if value is None:
        return pd.DataFrame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(value).__name__}")
    return value


def _ranking_frame(ranking) -> pd.DataFrame:
    """Accept a ranking DataFrame (or None) and return it, or an empty frame."""
    return _as_frame(ranking, "ranking")


def incident_count_data(ranking, top_n: int = DEFAULT_CHART_TOP_N) -> pd.DataFrame:
    """Log entries per incident cluster, best-ranked first.

    Returns a single-column DataFrame (``log_count``) indexed by a readable
    incident label such as ``#1 (cluster 15)``. The values are the ranker's own
    ``size`` column - nothing is estimated or invented. An empty ranking gives an
    empty DataFrame so the UI can skip the chart instead of crashing.
    """
    frame = _ranking_frame(ranking)
    empty = pd.DataFrame({_COUNT_COLUMN: pd.Series(dtype="int64")})
    empty.index.name = "incident"
    if frame.empty or "size" not in frame.columns:
        return empty
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}")

    ordered = frame.sort_values("rank") if "rank" in frame.columns else frame
    top = ordered.head(top_n)
    counts = pd.to_numeric(top["size"], errors="coerce").fillna(0).astype("int64")

    if "rank" in top.columns and "cluster_id" in top.columns:
        labels = [f"#{rank} (cluster {cluster})" for rank, cluster in zip(top["rank"], top["cluster_id"])]
    elif "cluster_id" in top.columns:
        labels = [f"cluster {cluster}" for cluster in top["cluster_id"]]
    else:
        labels = [f"row {position}" for position in range(len(top))]

    result = pd.DataFrame({_COUNT_COLUMN: counts.to_numpy()}, index=pd.Index(labels, name="incident"))
    return result


def severity_label(severity_score) -> str:
    """Map a ranker severity_score back to its severity level name.

    ``severity_score`` is the cluster's maximum severity weight divided by
    ``MAX_SEVERITY_WEIGHT``, so the mapping is exact for values the ranker
    produced. Anything unrecognizable (missing, non-numeric, out of range) is
    reported as ``UNKNOWN`` rather than guessed.
    """
    try:
        weight = round(float(severity_score) * MAX_SEVERITY_WEIGHT)
    except (TypeError, ValueError):
        # Includes NaN, which round() rejects with a ValueError.
        return "UNKNOWN"
    return _LABEL_BY_WEIGHT.get(float(weight), "UNKNOWN")


def severity_distribution_data(ranking) -> pd.DataFrame:
    """How many incident clusters have each maximum severity level.

    The category axis is the full severity ladder (DEBUG ... FATAL) in ascending
    weight order, including levels with zero clusters, so the reader can see the
    whole scale at a glance. Counts come from the ranker's ``severity_score``.
    """
    frame = _ranking_frame(ranking)
    empty = pd.DataFrame({_SEVERITY_COLUMN: pd.Series(dtype="int64")})
    empty.index.name = "max_severity"
    if frame.empty or "severity_score" not in frame.columns:
        return empty

    labels = frame["severity_score"].map(severity_label)
    counts = labels.value_counts()
    rows = [(level, int(counts.get(level, 0))) for level in SEVERITY_ORDER]
    for level in sorted(set(labels) - set(SEVERITY_ORDER)):
        rows.append((level, int(counts.get(level, 0))))
    return pd.DataFrame(rows, columns=["max_severity", _SEVERITY_COLUMN]).set_index("max_severity")


# --- Cluster detail view ------------------------------------------------------
# Everything below resolves against an analysis result that is ALREADY in memory.
# No function here runs the pipeline, re-clusters, re-reads the log file, or calls
# the LLM, so switching the selected cluster is a pure lookup.
def _cluster_label(row: pd.Series, cluster_id: int) -> str:
    """A readable selector label built only from ranker columns that exist."""
    parts = [f"Cluster {cluster_id}"]
    if "rank" in row.index and pd.notna(row["rank"]):
        parts.append(f"rank {int(row['rank'])}")
    if "size" in row.index and pd.notna(row["size"]):
        size = int(row["size"])
        parts.append(f"{size} log {'entry' if size == 1 else 'entries'}")
    if "risk_score" in row.index and pd.notna(row["risk_score"]):
        parts.append(f"risk {float(row['risk_score']):.2f}")
    return " - ".join(parts)


def cluster_options(ranking) -> dict:
    """Selectable incident clusters in rank order: ``{cluster_id: label}``.

    The ids and every number in the label come from the ranker's own output -
    nothing is estimated. An empty or unusable ranking gives an empty dict so the
    UI can hide the selector instead of crashing.
    """
    frame = _ranking_frame(ranking)
    cluster_column = config.CLUSTER_COLUMN
    if frame.empty or cluster_column not in frame.columns:
        return {}

    ordered = frame.sort_values("rank") if "rank" in frame.columns else frame
    options = {}
    for _, row in ordered.iterrows():
        try:
            cluster_id = int(row[cluster_column])
        except (TypeError, ValueError):
            continue
        options[cluster_id] = _cluster_label(row, cluster_id)
    return options


def cluster_log_entries(clustered, cluster_id) -> pd.DataFrame:
    """The original log rows belonging to ``cluster_id``, in original row order.

    This only FILTERS the clustered DataFrame the pipeline already produced: the
    ``message`` column is returned exactly as stored, and ``message_normalized``
    is carried along as supplementary information (never as a replacement). Only
    columns that actually exist are kept, so a trimmed frame degrades instead of
    raising. Returns an empty DataFrame for a missing cluster id, an unknown id,
    or a cluster with no matching rows.
    """
    frame = _as_frame(clustered, "clustered")
    cluster_column = config.CLUSTER_COLUMN
    if frame.empty or cluster_column not in frame.columns or cluster_id is None:
        return pd.DataFrame()
    try:
        target = int(cluster_id)
    except (TypeError, ValueError):
        return pd.DataFrame()

    ids = pd.to_numeric(frame[cluster_column], errors="coerce")
    matched = frame[ids == target]
    columns = [name for name in CLUSTER_DETAIL_COLUMNS if name in matched.columns]
    if not columns:
        return pd.DataFrame()
    return matched[columns].reset_index(drop=True)


def find_explanation(explanations, cluster_id):
    """The explanation already produced for ``cluster_id`` in this run, or None.

    A lookup only - it never generates, regenerates, or re-ranks anything. Ids are
    compared as ints when possible so an id that arrived as a numpy integer, a
    float, or a string still matches; malformed entries are skipped.
    """
    if cluster_id is None or not isinstance(explanations, (list, tuple)):
        return None

    def as_key(value):
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)

    target = as_key(cluster_id)
    for explanation in explanations:
        if not isinstance(explanation, dict):
            continue
        found = explanation.get(config.CLUSTER_COLUMN)
        if found is not None and as_key(found) == target:
            return explanation
    return None
