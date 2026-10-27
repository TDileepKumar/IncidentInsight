"""Day 9 - End-to-end pipeline orchestration for IncidentInsight.

Composes the existing, already-tested stages into ONE runnable workflow:

    parse_file() -> normalize_dataframe() -> cluster_dataframe() -> rank_clusters()

Design notes (approved Day 9):
  * This module only ORCHESTRATES. Every algorithm lives in its own stage module
    and is reused as-is - nothing is reimplemented here.
  * Clustering happens EXACTLY ONCE: cluster_dataframe() already vectorizes
    internally (it calls vectorize_dataframe then cluster_matrix), so we pass its
    clustered DataFrame straight into rank_clusters() and never re-cluster.
  * Evaluation / ground-truth comparison is deliberately NOT part of the
    production pipeline - it stays in src/evaluation.py, and the pipeline never
    requires ground-truth labels to run.
  * Failures are surfaced, not swallowed: a missing file, an empty file, or
    unusable (feature-less) messages raise clear errors from the relevant stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import config
from src.clusterer import CLUSTER_COLUMN, cluster_dataframe
from src.log_normalizer import NORMALIZED_COLUMN, normalize_dataframe
from src.log_parser import parse_file
from src.ranker import rank_clusters


@dataclass
class PipelineResult:
    """Structured outcome of a single pipeline run.

    Attributes:
        clustered: every parsed log row with `message_normalized` and `cluster_id`
            appended, in original row order (row i <-> line_id i).
        ranking: the ranked incident table from rank_clusters() with columns
            cluster_id, size, representative_message, severity_score,
            frequency_score, risk_score, rank.
        n_logs: number of log rows processed.
        n_clusters: number of distinct clusters discovered.
    """

    clustered: pd.DataFrame
    ranking: pd.DataFrame
    n_logs: int
    n_clusters: int


def run_pipeline(
    log_path: "str | Path" = config.RAW_LOG_PATH,
    distance_threshold: float = config.CLUSTER_DISTANCE_THRESHOLD,
    column: str = NORMALIZED_COLUMN,
) -> PipelineResult:
    """Run parse -> normalize -> cluster -> rank on a log file.

    Args:
        log_path: path to the raw log file (defaults to the generated dataset).
        distance_threshold: agglomerative clustering distance threshold; passed
            straight through to the existing clusterer (validated there).
        column: text column to cluster on (defaults to the normalized message).

    Returns:
        A PipelineResult holding the clustered DataFrame and the ranked table.

    Raises:
        FileNotFoundError: the log path does not exist or is not a file.
        ValueError: the file has no parseable rows, the messages carry no usable
            TF-IDF features (e.g. all blank/punctuation), or the threshold is
            invalid - all propagated from the underlying stage with clear text.
    """
    path = Path(log_path)
    if not path.is_file():
        raise FileNotFoundError(f"log file not found: {path}")

    parsed = parse_file(path)
    if len(parsed) == 0:
        raise ValueError(f"no log entries found in {path}")

    normalized = normalize_dataframe(parsed)
    # cluster_dataframe() vectorizes internally - this is the ONLY clustering call.
    clustered = cluster_dataframe(normalized, column=column, distance_threshold=distance_threshold)
    ranking = rank_clusters(clustered)

    return PipelineResult(
        clustered=clustered,
        ranking=ranking,
        n_logs=int(len(clustered)),
        n_clusters=int(clustered[CLUSTER_COLUMN].nunique()),
    )
