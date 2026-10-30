"""Cluster purity and ranking evaluation for IncidentInsight.

An OFFLINE, READ-ONLY evaluation module. It measures two things about a
clustering + ranking that ALREADY happened elsewhere:

  1. cluster_purity()  - how "pure" the discovered clusters are with respect to
     the ground-truth categories, and (crucially) how many clusters produced
     that purity. Purity alone is DEGENERATE: splitting every log into its own
     cluster gives purity 1.0, so we ALWAYS report it next to `n_clusters`.

  2. evaluate_ranking() - the ground-truth category COMPOSITION and dataset
     COVERAGE of the top-K ranked clusters. This describes what the ranking
     surfaces; it is NOT a claim that the ranking is "correct" or predictive.

Design rules:
  * Both functions REUSE evaluation.build_cluster_crosstab() to align ground
    truth to predictions by `line_id` (never by row position) - no alignment or
    contingency logic is duplicated here.
  * NOTHING is re-clustered and NO risk score is recomputed or changed. The
    ranked table is consumed exactly as the ranker produced it.
  * Categories are plain ground-truth labels. This module never calls a
    category "high-severity" (the ground-truth schema has no severity field),
    and never claims that category representation proves ranking effectiveness.
  * Inputs are never mutated.
"""

from __future__ import annotations

import numbers

import pandas as pd

from src.clusterer import CLUSTER_COLUMN
from src.evaluation import build_cluster_crosstab

# Columns this module requires on the ranked table produced by rank_clusters().
RANKING_REQUIRED_COLUMNS = [CLUSTER_COLUMN, "rank"]

# Deterministic schema of per_cluster_purity().
PER_CLUSTER_COLUMNS = [
    CLUSTER_COLUMN,
    "size",
    "majority_category",
    "majority_count",
    "purity",
]


# --- Small validation helpers -------------------------------------------------
def _require_dataframe(df, name: str) -> None:
    """Reject anything that is not a pandas DataFrame with a clear TypeError."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")


def _require_nonempty(df: pd.DataFrame, name: str) -> None:
    """Reject an empty DataFrame - purity/ranking are undefined without records."""
    if len(df) == 0:
        raise ValueError(f"{name} must contain at least one row, got an empty DataFrame")


def _validate_top_k(top_k) -> int:
    """Validate `top_k` is a positive integer; booleans are rejected.

    Rejects bool (True/False are ints in Python but meaningless here), any
    non-integer type, and non-positive values, all with a clear ValueError.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral):
        raise ValueError(
            f"top_k must be a positive integer, got {type(top_k).__name__}"
        )
    value = int(top_k)
    if value <= 0:
        raise ValueError(f"top_k must be > 0, got {value}")
    return value


# --- A. Cluster purity --------------------------------------------------------
def _purity_from_crosstab(table: pd.DataFrame) -> dict[str, float]:
    """Compute overall purity from a category x cluster contingency table.

    purity = (sum over clusters of the largest single-category count) / N, where
    N is the total number of aligned records. `table` has categories as rows and
    cluster IDs as columns (exactly what build_cluster_crosstab returns).
    """
    total = int(table.values.sum())
    if table.values.size == 0 or total == 0:
        raise ValueError(
            "cluster_purity requires at least one aligned record; "
            "the contingency table is empty"
        )

    # Per cluster (column): the count of its majority category.
    majority_counts = table.max(axis=0)
    purity = float(majority_counts.sum() / total)
    return {
        "purity": purity,
        "n_clusters": int(table.shape[1]),
        "n_logs": total,
    }


def cluster_purity(
    ground_truth: pd.DataFrame,
    clustered_data: pd.DataFrame,
    category_column: str = "category",
    cluster_column: str = CLUSTER_COLUMN,
    id_column: str = "line_id",
) -> dict[str, float]:
    """Return overall cluster purity WITH the cluster count that produced it.

    Result keys:
      * "purity"    - float in [0, 1]; 1.0 means every cluster is single-category.
      * "n_clusters" - number of clusters in the aligned data (int).
      * "n_logs"     - number of aligned records counted (int).

    Purity is reported ALONGSIDE n_clusters on purpose: purity rises as clusters
    are split finer, so a high purity with a huge cluster count is not "good"
    clustering - the pair must be read together.

    Ground truth is aligned to predictions by `id_column` via
    build_cluster_crosstab(), so all of that function's validation applies
    (required columns, unique non-null identifiers, identical identifier sets).
    An empty contingency table (no aligned records) raises ValueError rather than
    reporting a misleading 0/0 score. Inputs are not modified.
    """
    _require_dataframe(ground_truth, "ground_truth")
    _require_dataframe(clustered_data, "clustered_data")
    _require_nonempty(ground_truth, "ground_truth")
    _require_nonempty(clustered_data, "clustered_data")

    table = build_cluster_crosstab(
        ground_truth,
        clustered_data,
        category_column=category_column,
        cluster_column=cluster_column,
        id_column=id_column,
    )
    return _purity_from_crosstab(table)


def per_cluster_purity(
    ground_truth: pd.DataFrame,
    clustered_data: pd.DataFrame,
    category_column: str = "category",
    cluster_column: str = CLUSTER_COLUMN,
    id_column: str = "line_id",
) -> pd.DataFrame:
    """Per-cluster purity detail: cluster_id, size, majority_category, count, purity.

    One row per cluster. `majority_category` is the ground-truth category with
    the highest count inside that cluster (ties broken by the crosstab's sorted
    category index, so the result is deterministic). `purity` is that majority
    count divided by the cluster size. Rows are ordered by size descending then
    cluster_id ascending - the same deterministic ordering the project uses
    elsewhere. Inputs are not modified.
    """
    _require_dataframe(ground_truth, "ground_truth")
    _require_dataframe(clustered_data, "clustered_data")
    _require_nonempty(ground_truth, "ground_truth")
    _require_nonempty(clustered_data, "clustered_data")

    table = build_cluster_crosstab(
        ground_truth,
        clustered_data,
        category_column=category_column,
        cluster_column=cluster_column,
        id_column=id_column,
    )
    if table.values.size == 0:
        return pd.DataFrame(columns=PER_CLUSTER_COLUMNS)

    rows = []
    for cid in table.columns:
        column = table[cid]
        size = int(column.sum())
        majority_category = column.idxmax()
        majority_count = int(column.max())
        rows.append(
            {
                CLUSTER_COLUMN: cid,
                "size": size,
                "majority_category": majority_category,
                "majority_count": majority_count,
                "purity": (majority_count / size) if size > 0 else 0.0,
            }
        )

    detail = pd.DataFrame(rows, columns=PER_CLUSTER_COLUMNS)
    return detail.sort_values(
        ["size", CLUSTER_COLUMN], ascending=[False, True]
    ).reset_index(drop=True)


# --- B. Ranking evaluation ----------------------------------------------------
def _select_top_cluster_ids(ranking: pd.DataFrame, top_k: int) -> list:
    """Return the cluster IDs of the top-K ranked rows, deterministically.

    Uses the ranker's own `rank` column, tie-broken by cluster_id ascending, so
    the selection never depends on incidental row order. If fewer than `top_k`
    clusters exist, all of them are returned.
    """
    ordered = ranking.sort_values(
        ["rank", CLUSTER_COLUMN], ascending=[True, True]
    )
    return ordered[CLUSTER_COLUMN].head(top_k).tolist()


def evaluate_ranking(
    ranking: pd.DataFrame,
    ground_truth: pd.DataFrame,
    clustered_data: pd.DataFrame,
    top_k: int = 5,
    category_column: str = "category",
    id_column: str = "line_id",
) -> dict:
    """Describe the ground-truth composition of the top-K ranked clusters.

    This is a COMPOSITION / COVERAGE report, not a correctness verdict. It says
    what categories the highest-ranked clusters contain and how much of the
    dataset they cover - it does NOT measure whether the ranking is predictive
    or "right", and it never re-ranks, re-clusters, or changes any risk score.

    Result keys:
      * "n_clusters_available" - distinct clusters in the ranked table.
      * "n_clusters_evaluated" - clusters actually selected (min(top_k, available)).
      * "n_logs_evaluated"     - LOG ENTRIES represented by the selected clusters.
      * "n_logs_total"         - LOG ENTRIES in the full aligned dataset.
      * "coverage"             - n_logs_evaluated / n_logs_total, in [0, 1].
      * "category_counts"      - {category: count} within the selected clusters.
      * "category_proportions" - {category: fraction} of n_logs_evaluated.

    The cluster-level counts (n_clusters_*) and the log-entry-level counts
    (n_logs_*) are deliberately kept separate: K clusters can represent very
    different numbers of log entries.

    Validation: `top_k` must be a positive integer (booleans rejected); all three
    inputs must be non-empty DataFrames; `ranking` must carry `cluster_id` and
    `rank`; every ranked cluster must exist in the aligned contingency table.
    Ground truth is aligned by `id_column` via build_cluster_crosstab(). Inputs
    are not modified.
    """
    k = _validate_top_k(top_k)

    _require_dataframe(ranking, "ranking")
    _require_dataframe(ground_truth, "ground_truth")
    _require_dataframe(clustered_data, "clustered_data")
    _require_nonempty(ranking, "ranking")
    _require_nonempty(ground_truth, "ground_truth")
    _require_nonempty(clustered_data, "clustered_data")

    missing = [c for c in RANKING_REQUIRED_COLUMNS if c not in ranking.columns]
    if missing:
        raise ValueError(f"ranking is missing required column(s): {missing}")

    # Align ground truth to ALL clusters once (reuses the existing crosstab).
    table = build_cluster_crosstab(
        ground_truth,
        clustered_data,
        category_column=category_column,
        cluster_column=CLUSTER_COLUMN,
        id_column=id_column,
    )

    available_ids = list(table.columns)
    ranked_ids = ranking[CLUSTER_COLUMN].tolist()
    unknown = [cid for cid in ranked_ids if cid not in set(available_ids)]
    if unknown:
        raise ValueError(
            "ranking references cluster_id(s) absent from the aligned data: "
            f"{unknown[:5]}"
        )

    selected_ids = _select_top_cluster_ids(ranking, k)
    selected = table[selected_ids]

    n_logs_total = int(table.values.sum())
    category_counts = selected.sum(axis=1)
    n_logs_evaluated = int(category_counts.sum())

    counts = {str(cat): int(n) for cat, n in category_counts.items()}
    proportions = {
        str(cat): (float(n) / n_logs_evaluated if n_logs_evaluated > 0 else 0.0)
        for cat, n in category_counts.items()
    }

    return {
        "n_clusters_available": int(len(available_ids)),
        "n_clusters_evaluated": int(len(selected_ids)),
        "n_logs_evaluated": n_logs_evaluated,
        "n_logs_total": n_logs_total,
        "coverage": (n_logs_evaluated / n_logs_total) if n_logs_total > 0 else 0.0,
        "category_counts": counts,
        "category_proportions": proportions,
    }
