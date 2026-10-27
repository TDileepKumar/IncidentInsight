"""Day 6 - Agglomerative clustering for IncidentInsight.

Stage 4 of the pipeline. Groups the TF-IDF vectors of normalized log messages
into incident clusters so that "the same underlying error" lands in the same
group. This module is PURE: no DB, no UI, no network, no evaluation metrics.

Approved Day 6 decisions:
  * AgglomerativeClustering with cosine distance + average linkage
  * a DISTANCE THRESHOLD (n_clusters=None) so the cluster count is discovered
    from the data instead of hard-coded
  * accept the sparse TF-IDF matrix but DENSIFY it before fitting: the installed
    scikit-learn (1.8.0) rejects sparse input for this configuration. At 1,000 x
    ~120 features the dense matrix is under 1 MB, so this is cheap and safe.
  * one integer label per input row, in the ORIGINAL row order (row i <-> line_id i)
  * never fabricate labels for invalid input - raise an actionable error instead

Degenerate-input contract (verified against scikit-learn 1.8.0):
  * 0 rows              -> empty integer label array
  * 1 row, >=1 feature  -> [0] (short-circuited; sklearn needs >=2 samples)
  * >=2 rows, 0 features-> ValueError (all messages blank/unusable)
  * invalid threshold / non-numeric / non-finite / wrong ndim -> ValueError
"""

from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.cluster import AgglomerativeClustering

from src import config
from src.feature_extractor import vectorize_dataframe
from src.log_normalizer import NORMALIZED_COLUMN

# Column appended by cluster_dataframe / expected by summarize_clusters.
CLUSTER_COLUMN = config.CLUSTER_COLUMN

# Numeric dtypes we accept: signed/unsigned int, float. Bool ('b'), object ('O')
# and unicode ('U') are rejected as non-numeric.
_NUMERIC_KINDS = {"i", "u", "f"}


def _validate_threshold(distance_threshold) -> float:
    """Return distance_threshold as a float, or raise a clear ValueError.

    Rejects booleans, strings, None, NaN, +/-inf, zero and negatives. We never
    silently substitute a default - an invalid threshold is a caller bug.
    """
    # bool is a subclass of int, so exclude it explicitly before the Real check.
    if isinstance(distance_threshold, bool):
        raise ValueError("distance_threshold must be a positive number, not a bool")
    if not isinstance(distance_threshold, numbers.Real):
        raise ValueError(
            "distance_threshold must be a positive number, got "
            f"{type(distance_threshold).__name__}"
        )
    value = float(distance_threshold)
    if not math.isfinite(value):
        raise ValueError(f"distance_threshold must be finite, got {value!r}")
    if value <= 0.0:
        raise ValueError(f"distance_threshold must be > 0, got {value}")
    return value


def build_clusterer(
    distance_threshold: float = config.CLUSTER_DISTANCE_THRESHOLD,
    metric: str = config.CLUSTER_METRIC,
    linkage: str = config.CLUSTER_LINKAGE,
) -> AgglomerativeClustering:
    """Build an AgglomerativeClustering estimator in distance-threshold mode.

    n_clusters is forced to None (sklearn requires exactly one of n_clusters /
    distance_threshold). Incompatible metric/linkage combos are NOT corrected
    here - they raise a clear error from scikit-learn at fit time.
    """
    threshold = _validate_threshold(distance_threshold)
    return AgglomerativeClustering(
        n_clusters=None,
        metric=metric,
        linkage=linkage,
        distance_threshold=threshold,
    )


def cluster_matrix(
    X,
    distance_threshold: float = config.CLUSTER_DISTANCE_THRESHOLD,
) -> np.ndarray:
    """Cluster a (sparse or dense) numeric matrix; return one int label per row.

    Row order is preserved exactly. Sparse input is densified with .toarray()
    only AFTER rows/features are validated. Degenerate inputs follow the module
    contract; invalid inputs raise ValueError rather than returning fake labels.
    """
    threshold = _validate_threshold(distance_threshold)

    sparse = issparse(X)
    if sparse:
        n_rows, n_features = X.shape
    else:
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"X must be a 2-dimensional numeric matrix, got ndim={arr.ndim}")
        n_rows, n_features = arr.shape

    # Edge case: no rows -> nothing to cluster.
    if n_rows == 0:
        return np.empty(0, dtype=np.int64)

    # Edge case: no features -> messages carry no usable signal (all blank /
    # punctuation-only after TF-IDF). Clustering would be meaningless.
    if n_features == 0:
        raise ValueError(
            "cannot cluster: the matrix has 0 features "
            "(all messages are blank, empty, or contain no usable tokens)"
        )

    # Validate dtype and finiteness before densifying, then densify.
    if sparse:
        if X.dtype.kind not in _NUMERIC_KINDS:
            raise ValueError(f"X must contain numeric values, got dtype kind '{X.dtype.kind}'")
        if X.nnz and not np.isfinite(X.data).all():
            raise ValueError("X contains non-finite values (NaN or inf)")
        dense = X.toarray()
    else:
        if arr.dtype.kind not in _NUMERIC_KINDS:
            raise ValueError(f"X must contain numeric values, got dtype kind '{arr.dtype.kind}'")
        if arr.size and not np.isfinite(arr).all():
            raise ValueError("X contains non-finite values (NaN or inf)")
        dense = arr

    # Edge case: a single usable sample forms exactly one cluster. Short-circuit
    # because sklearn requires >= 2 samples.
    if n_rows == 1:
        return np.array([0], dtype=np.int64)

    estimator = build_clusterer(distance_threshold=threshold)
    labels = estimator.fit_predict(np.asarray(dense, dtype=float))
    return np.asarray(labels, dtype=np.int64)


def cluster_dataframe(
    df: pd.DataFrame,
    column: str = NORMALIZED_COLUMN,
    distance_threshold: float = config.CLUSTER_DISTANCE_THRESHOLD,
) -> pd.DataFrame:
    """Return a COPY of `df` with an appended `cluster_id` column.

    Uses the existing vectorize_dataframe() (no duplicated TF-IDF logic), so the
    alignment contract holds: cluster_id at row i belongs to df row i / line_id i.
    Original columns, values, index and row order are preserved and the input is
    never mutated. Empty DataFrame -> empty integer cluster_id column.
    """
    out = df.copy()

    if len(df) == 0:
        out[CLUSTER_COLUMN] = np.empty(0, dtype=np.int64)
        return out

    matrix, _ = vectorize_dataframe(df, column=column)
    labels = cluster_matrix(matrix, distance_threshold=distance_threshold)

    # labels is a plain ndarray -> assigned positionally, preserving df's index.
    out[CLUSTER_COLUMN] = labels
    return out


def summarize_clusters(
    clustered_df: pd.DataFrame,
    message_column: str = NORMALIZED_COLUMN,
) -> pd.DataFrame:
    """Return a one-row-per-cluster summary: cluster_id, size, representative.

    The representative message is the MOST FREQUENT non-null, non-empty value of
    `message_column` within the cluster; ties are broken lexicographically so the
    result is deterministic. A cluster with no usable message yields "" as a
    documented fallback. Cluster IDs are opaque labels - no category names are
    inferred. Ordering is deterministic: size descending, then cluster_id
    ascending. The supplied DataFrame is not modified.
    """
    summary_columns = [CLUSTER_COLUMN, "size", "representative_message"]

    if CLUSTER_COLUMN not in clustered_df.columns:
        raise KeyError(f"column '{CLUSTER_COLUMN}' not found in DataFrame")

    if len(clustered_df) == 0:
        return pd.DataFrame(columns=summary_columns)

    rows = []
    for cid, group in clustered_df.groupby(CLUSTER_COLUMN, sort=True):
        messages = group[message_column].dropna()
        messages = messages[messages.astype(str).str.strip() != ""]
        if len(messages) > 0:
            counts = messages.value_counts()
            top = sorted(counts[counts == counts.max()].index.tolist())
            representative = top[0]
        else:
            representative = ""  # documented fallback: no usable message
        rows.append({CLUSTER_COLUMN: cid, "size": int(len(group)),
                     "representative_message": representative})

    summary = pd.DataFrame(rows, columns=summary_columns)
    return summary.sort_values(
        ["size", CLUSTER_COLUMN], ascending=[False, True]
    ).reset_index(drop=True)
