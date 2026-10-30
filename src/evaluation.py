"""Clustering evaluation for IncidentInsight.

Stage 5 of the pipeline: MEASURE how good the src/clusterer.py clusters are, using
the ground-truth `category` labels from data/processed/ground_truth.csv. This
module is PURE evaluation - it does not cluster, tune, persist, or visualize
anything.

Design notes:
  * Quality is measured with two chance-adjusted, label-permutation-invariant
    metrics from scikit-learn: Adjusted Rand Index (ARI) and Adjusted Mutual
    Information (AMI). Cluster IDs are ARBITRARY identifiers - they are never
    assumed to equal category IDs, and ARI/AMI are NOT accuracy percentages
    (0.0 ~= chance, 1.0 = perfect agreement; they can be negative).
  * Ground truth is aligned to predictions by the shared `line_id` key, NOT by
    row position, so ordering differences can never silently corrupt a score.
  * The production threshold in config.py is never modified; compare_thresholds
    is a read-only evaluation utility.

Empty-input contract: evaluate_clustering RAISES ValueError on empty input
rather than echoing scikit-learn's misleading 1.0 for zero-length labels.
"""

from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from src.clusterer import CLUSTER_COLUMN, cluster_dataframe, summarize_clusters
from src.log_normalizer import NORMALIZED_COLUMN

# Column order of the compare_thresholds() result.
THRESHOLD_REPORT_COLUMNS = [
    "distance_threshold",
    "n_clusters",
    "n_singletons",
    "ari",
    "ami",
]


# --- Small validation helpers -------------------------------------------------
def _as_label_list(labels, name: str) -> list:
    """Coerce a label sequence (list / ndarray / Series) to a flat Python list."""
    if labels is None:
        raise ValueError(f"{name} must be a non-empty sequence of labels, got None")
    if isinstance(labels, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of labels, not a string")
    if isinstance(labels, pd.Series):
        return labels.tolist()
    if isinstance(labels, np.ndarray):
        if labels.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional, got ndim={labels.ndim}")
        return labels.tolist()
    try:
        return list(labels)
    except TypeError:
        raise ValueError(f"{name} must be an iterable of labels") from None


def _require_columns(df: pd.DataFrame, columns, name: str) -> None:
    """Raise KeyError listing any required columns missing from `df`."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required column(s): {missing}")


def _validate_ids(df: pd.DataFrame, id_column: str, name: str) -> None:
    """Require the identifier column to be present, non-null, and unique."""
    ids = df[id_column]
    if ids.isna().any():
        raise ValueError(f"{name} contains missing (null) values in '{id_column}'")
    if not ids.is_unique:
        dupes = ids[ids.duplicated()].unique().tolist()[:5]
        raise ValueError(f"{name} contains duplicate '{id_column}' values: {dupes}")


def _aligned_merge(
    ground_truth: pd.DataFrame,
    clustered_data: pd.DataFrame,
    category_column: str,
    cluster_column: str,
    id_column: str,
) -> pd.DataFrame:
    """Join ground truth to predictions on `id_column` (one-to-one, validated).

    Verifies required columns, unique non-null identifiers on BOTH sides, and
    identical identifier sets. A one_to_one merge validate guarantees no
    many-to-many row multiplication. Neither input is modified.
    """
    _require_columns(ground_truth, [id_column, category_column], "ground_truth")
    _require_columns(clustered_data, [id_column, cluster_column], "clustered_data")
    _validate_ids(ground_truth, id_column, "ground_truth")
    _validate_ids(clustered_data, id_column, "clustered_data")

    gt_ids = set(ground_truth[id_column].tolist())
    cl_ids = set(clustered_data[id_column].tolist())
    if gt_ids != cl_ids:
        only_gt = sorted(gt_ids - cl_ids)[:5]
        only_cl = sorted(cl_ids - gt_ids)[:5]
        raise ValueError(
            f"'{id_column}' sets differ between ground_truth and clustered_data "
            f"(only in ground_truth: {only_gt}; only in clustered_data: {only_cl})"
        )

    return ground_truth[[id_column, category_column]].merge(
        clustered_data[[id_column, cluster_column]],
        on=id_column,
        how="inner",
        validate="one_to_one",
    )


# --- A. Metric evaluation -----------------------------------------------------
def evaluate_clustering(y_true, y_pred) -> dict[str, float]:
    """Return {"ari": float, "ami": float} for true categories vs cluster labels.

    Both metrics are chance-adjusted and invariant to how clusters are numbered,
    so cluster IDs need not correspond to category IDs. They are NOT accuracy
    percentages: 1.0 is perfect agreement, ~0.0 is chance-level, and values can
    be negative when agreement is worse than chance.

    Raises ValueError on empty input (we refuse to report sklearn's misleading
    1.0 for zero-length labels), on mismatched lengths, and on invalid inputs.
    """
    true_labels = _as_label_list(y_true, "y_true")
    pred_labels = _as_label_list(y_pred, "y_pred")

    if len(true_labels) == 0 or len(pred_labels) == 0:
        raise ValueError(
            "evaluate_clustering requires non-empty label sequences "
            f"(got len(y_true)={len(true_labels)}, len(y_pred)={len(pred_labels)})"
        )
    if len(true_labels) != len(pred_labels):
        raise ValueError(
            f"label length mismatch: y_true has {len(true_labels)} labels, "
            f"y_pred has {len(pred_labels)}"
        )

    return {
        "ari": float(adjusted_rand_score(true_labels, pred_labels)),
        "ami": float(adjusted_mutual_info_score(true_labels, pred_labels)),
    }


# --- B. Ground-truth cross-tabulation -----------------------------------------
def build_cluster_crosstab(
    ground_truth: pd.DataFrame,
    clustered_data: pd.DataFrame,
    category_column: str = "category",
    cluster_column: str = CLUSTER_COLUMN,
    id_column: str = "line_id",
) -> pd.DataFrame:
    """Return a contingency table: actual categories (rows) x cluster IDs (cols).

    Rows are aligned strictly by `id_column` (default `line_id`), never by
    positional order. Values are log counts. Raises a clear error on missing
    columns, null/duplicate identifiers, or differing identifier sets - records
    are never silently dropped or multiplied. Inputs are not modified.
    """
    merged = _aligned_merge(
        ground_truth, clustered_data, category_column, cluster_column, id_column
    )
    return pd.crosstab(merged[category_column], merged[cluster_column])


# --- C. Cluster composition (reuses src.clusterer summary) --------------------
def cluster_composition(
    clustered_df: pd.DataFrame,
    message_column: str = NORMALIZED_COLUMN,
) -> pd.DataFrame:
    """Readable per-cluster summary: cluster_id, size, representative_message.

    A thin reuse of clusterer.summarize_clusters() - no clustering or summary
    logic is duplicated, and its deterministic ordering is preserved. The input
    DataFrame is not modified.
    """
    return summarize_clusters(clustered_df, message_column=message_column)


# --- D. Threshold comparison --------------------------------------------------
def _validate_thresholds(thresholds) -> list[float]:
    """Validate a collection of thresholds; return them as a list of floats."""
    if thresholds is None:
        raise ValueError("thresholds must be a non-empty collection of positive numbers")
    if isinstance(thresholds, (str, bytes)):
        raise ValueError("thresholds must be a collection of numbers, not a string")
    try:
        items = list(thresholds)
    except TypeError:
        raise ValueError("thresholds must be an iterable of positive numbers") from None
    if len(items) == 0:
        raise ValueError("thresholds must be a non-empty collection")

    validated: list[float] = []
    for t in items:
        if isinstance(t, bool):
            raise ValueError(f"threshold values must be numbers, not bool (got {t!r})")
        if not isinstance(t, numbers.Real):
            raise ValueError(
                f"threshold values must be numbers, got {type(t).__name__}"
            )
        value = float(t)
        if not math.isfinite(value):
            raise ValueError(f"threshold values must be finite, got {value!r}")
        if value <= 0.0:
            raise ValueError(f"threshold values must be > 0, got {value}")
        validated.append(value)
    return validated


def compare_thresholds(
    df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    thresholds=(0.3, 0.5, 0.7),
    category_column: str = "category",
    id_column: str = "line_id",
) -> pd.DataFrame:
    """Evaluate several distance thresholds on the SAME data and ground truth.

    For each threshold, re-runs cluster_dataframe() (the production config in
    config.py is never touched), then reports distance_threshold, n_clusters,
    n_singletons (clusters of exactly one log), ari and ami. Predictions are
    aligned to ground truth by `id_column`. Errors propagate - a failing
    threshold is never silently skipped. Input DataFrames and their row order
    are preserved.
    """
    validated_thresholds = _validate_thresholds(thresholds)

    _require_columns(df, [id_column], "df")
    _require_columns(ground_truth, [id_column, category_column], "ground_truth")
    _validate_ids(df, id_column, "df")
    _validate_ids(ground_truth, id_column, "ground_truth")

    rows = []
    for threshold in validated_thresholds:
        clustered = cluster_dataframe(df, distance_threshold=threshold)

        merged = _aligned_merge(
            ground_truth, clustered, category_column, CLUSTER_COLUMN, id_column
        )
        scores = evaluate_clustering(merged[category_column], merged[CLUSTER_COLUMN])

        sizes = clustered[CLUSTER_COLUMN].value_counts()
        rows.append(
            {
                "distance_threshold": threshold,
                "n_clusters": int(clustered[CLUSTER_COLUMN].nunique()),
                "n_singletons": int((sizes == 1).sum()),
                "ari": scores["ari"],
                "ami": scores["ami"],
            }
        )

    return pd.DataFrame(rows, columns=THRESHOLD_REPORT_COLUMNS)
