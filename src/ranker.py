"""Deterministic incident-cluster ranking for IncidentInsight.

Stage 6 of the pipeline: turn the src/clusterer.py clusters into a prioritized
incident list so the loudest AND most severe problems surface first. This module
is PURE and READ-ONLY - it does not cluster, evaluate, persist, or call anything
external, and it never mutates its inputs.

Design notes:
  * Ranking blends FREQUENCY (how often a cluster occurs) and MAX SEVERITY (the
    worst log level in the cluster). Both are normalized to [0, 1]-ish scores and
    combined with caller weights (default 60% frequency / 40% severity).
  * Cluster size and representative_message are REUSED from
    clusterer.summarize_clusters() - never recomputed here.
  * Severity ordering lives in THIS module (DEFAULT_SEVERITY_WEIGHTS), not in
    config.py. Unknown or missing levels map to weight 0 and never crash.
  * Output ordering is fully deterministic: risk_score desc, then size desc, then
    cluster_id asc; `rank` is a consecutive 1-based integer after that sort.
"""

from __future__ import annotations

import math
import numbers

import pandas as pd

from src.clusterer import CLUSTER_COLUMN, summarize_clusters

# Severity ordering. FATAL is worst. Levels not listed here (or missing/None)
# receive weight 0 rather than raising - real logs contain unknown levels.
DEFAULT_SEVERITY_WEIGHTS = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
    "CRITICAL": 4,
    "FATAL": 5,
}

# severity_score is the cluster's max weight divided by this (the FATAL weight),
# so a worst-case cluster scores 1.0.
MAX_SEVERITY_WEIGHT = 5

# Exact, documented output schema (column order matters).
OUTPUT_COLUMNS = [
    "cluster_id",
    "size",
    "representative_message",
    "severity_score",
    "frequency_score",
    "risk_score",
    "rank",
]

# Columns the ranker itself needs on the clustered DataFrame.
REQUIRED_COLUMNS = [CLUSTER_COLUMN, "level"]

# Columns a caller-supplied `summary` must carry (from summarize_clusters()).
SUMMARY_COLUMNS = [CLUSTER_COLUMN, "size", "representative_message"]


def _validate_rank_weights(w_freq, w_sev) -> tuple[float, float]:
    """Validate the frequency/severity weights and return them as floats.

    Both must be finite, non-negative numbers (bools rejected) summing to 1.0
    within floating-point tolerance. Invalid values raise, never get defaulted.
    """
    values = []
    for name, val in (("w_freq", w_freq), ("w_sev", w_sev)):
        if isinstance(val, bool) or not isinstance(val, numbers.Real):
            raise ValueError(f"{name} must be a non-negative number, got {type(val).__name__}")
        num = float(val)
        if not math.isfinite(num):
            raise ValueError(f"{name} must be finite, got {num!r}")
        if num < 0.0:
            raise ValueError(f"{name} must be non-negative, got {num}")
        values.append(num)
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError(f"w_freq + w_sev must equal 1.0, got {values[0]} + {values[1]}")
    return values[0], values[1]


def _normalize_severity_weights(severity_weights) -> dict[str, float]:
    """Return a {UPPER_LEVEL: weight} mapping from the default or a custom one.

    Keys are normalized (strip + upper) so lookups are case/whitespace tolerant.
    Values must be finite, non-negative numbers; bools are rejected.
    """
    source = DEFAULT_SEVERITY_WEIGHTS if severity_weights is None else severity_weights
    if not hasattr(source, "items"):
        raise TypeError("severity_weights must be a mapping of level name -> numeric weight")

    normalized: dict[str, float] = {}
    for key, val in source.items():
        if isinstance(val, bool) or not isinstance(val, numbers.Real):
            raise ValueError(
                f"severity weight for {key!r} must be a number, got {type(val).__name__}"
            )
        num = float(val)
        if not math.isfinite(num):
            raise ValueError(f"severity weight for {key!r} must be finite, got {num!r}")
        if num < 0.0:
            raise ValueError(f"severity weight for {key!r} must be non-negative, got {num}")
        normalized[str(key).strip().upper()] = num
    return normalized


def _validate_summary(summary: pd.DataFrame, clustered_df: pd.DataFrame) -> None:
    """Ensure a caller-supplied summary matches the clusters being ranked."""
    if not isinstance(summary, pd.DataFrame):
        raise TypeError(f"summary must be a pandas DataFrame, got {type(summary).__name__}")

    missing = [c for c in SUMMARY_COLUMNS if c not in summary.columns]
    if missing:
        raise ValueError(f"summary is missing required column(s): {missing}")

    if not summary[CLUSTER_COLUMN].is_unique:
        raise ValueError("summary contains duplicate cluster_id rows")

    df_ids = set(clustered_df[CLUSTER_COLUMN].tolist())
    sum_ids = set(summary[CLUSTER_COLUMN].tolist())
    if df_ids != sum_ids:
        raise ValueError(
            "summary does not correspond to clustered_df "
            f"(only in clustered_df: {sorted(df_ids - sum_ids)[:5]}; "
            f"only in summary: {sorted(sum_ids - df_ids)[:5]})"
        )

    actual_sizes = clustered_df[CLUSTER_COLUMN].value_counts()
    provided_sizes = summary.set_index(CLUSTER_COLUMN)["size"]
    for cid in sum_ids:
        if int(provided_sizes.loc[cid]) != int(actual_sizes.get(cid, 0)):
            raise ValueError(
                f"summary size for cluster {cid!r} does not match clustered_df"
            )


def _empty_result() -> pd.DataFrame:
    """An empty ranking result that still carries the documented schema."""
    return pd.DataFrame(
        {
            "cluster_id": pd.Series(dtype="int64"),
            "size": pd.Series(dtype="int64"),
            "representative_message": pd.Series(dtype="object"),
            "severity_score": pd.Series(dtype="float64"),
            "frequency_score": pd.Series(dtype="float64"),
            "risk_score": pd.Series(dtype="float64"),
            "rank": pd.Series(dtype="int64"),
        }
    )[OUTPUT_COLUMNS]


def rank_clusters(
    clustered_df: pd.DataFrame,
    summary: pd.DataFrame | None = None,
    severity_weights: dict | None = None,
    w_freq: float = 0.6,
    w_sev: float = 0.4,
) -> pd.DataFrame:
    """Rank incident clusters by blended frequency + max severity.

    Returns a NEW DataFrame with columns OUTPUT_COLUMNS, sorted by risk_score
    desc, size desc, cluster_id asc, with a consecutive 1-based `rank`. Neither
    `clustered_df` nor a supplied `summary` is modified.

    Scores:
      severity_score  = (max severity weight in the cluster) / 5
      frequency_score = cluster size / largest cluster size   (0 if none)
      risk_score      = w_freq * frequency_score + w_sev * severity_score
    """
    if not isinstance(clustered_df, pd.DataFrame):
        raise TypeError(f"clustered_df must be a pandas DataFrame, got {type(clustered_df).__name__}")

    missing = [c for c in REQUIRED_COLUMNS if c not in clustered_df.columns]
    if missing:
        raise ValueError(f"clustered_df is missing required column(s): {missing}")

    w_freq, w_sev = _validate_rank_weights(w_freq, w_sev)
    weights = _normalize_severity_weights(severity_weights)

    if summary is not None:
        _validate_summary(summary, clustered_df)

    if len(clustered_df) == 0:
        return _empty_result()

    # Reuse the existing summary (size + representative_message); never recompute.
    base = summarize_clusters(clustered_df) if summary is None else summary
    out = base[SUMMARY_COLUMNS].copy()

    # Per-cluster MAX severity weight (unknown/missing levels -> 0, no crash).
    row_weights = clustered_df["level"].map(lambda v: weights.get(str(v).strip().upper(), 0.0))
    max_sev = row_weights.groupby(clustered_df[CLUSTER_COLUMN]).max()

    out["severity_score"] = out[CLUSTER_COLUMN].map(max_sev).fillna(0.0).astype(float) / MAX_SEVERITY_WEIGHT

    sizes = out["size"].astype(float)
    max_size = float(out["size"].max())
    out["frequency_score"] = (sizes / max_size) if max_size > 0 else 0.0

    out["risk_score"] = w_freq * out["frequency_score"] + w_sev * out["severity_score"]

    out = out.sort_values(
        ["risk_score", "size", CLUSTER_COLUMN],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)

    return out[OUTPUT_COLUMNS]
