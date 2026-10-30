"""Focused tests for agglomerative clustering."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src import config
from src.clusterer import (
    CLUSTER_COLUMN,
    build_clusterer,
    cluster_dataframe,
    cluster_matrix,
    summarize_clusters,
)
from src.feature_extractor import vectorize_messages
from src.log_normalizer import NORMALIZED_COLUMN, normalize_dataframe
from src.log_parser import parse_file
from tests.conftest import needs_generated_data, needs_ground_truth

# --- Small deterministic fixtures ---------------------------------------------
# Two messages that share most vocabulary (should group) and one clearly
# different template (should separate) at a sensible threshold.
SIM_A = "connection to database failed after <NUM>s on <HOST>"
SIM_B = "connection to database timed out after <NUM>s on <HOST>"
DISTINCT = "Payment declined by provider PayPal for order <ID>, code=<NUM>"


def _csr(docs):
    matrix, _ = vectorize_messages(docs)
    return matrix


# --- build_clusterer configuration --------------------------------------------
def test_build_clusterer_uses_threshold_mode_and_config():
    est = build_clusterer()
    assert est.n_clusters is None                       # threshold mode requires this
    assert est.metric == config.CLUSTER_METRIC == "cosine"
    assert est.linkage == config.CLUSTER_LINKAGE == "average"
    assert est.distance_threshold == config.CLUSTER_DISTANCE_THRESHOLD


def test_build_clusterer_accepts_explicit_threshold():
    est = build_clusterer(distance_threshold=0.25)
    assert est.distance_threshold == 0.25


# --- Basic clustering ---------------------------------------------------------
def test_similar_grouped_and_distinct_separated():
    matrix = _csr([SIM_A, SIM_B, DISTINCT])
    labels = cluster_matrix(matrix, distance_threshold=0.5)
    # The two database-connection templates belong together...
    assert labels[0] == labels[1]
    # ...and the payment template is its own cluster.
    assert labels[2] != labels[0]


def test_labels_are_integer_and_one_per_row():
    matrix = _csr([SIM_A, SIM_B, DISTINCT])
    labels = cluster_matrix(matrix)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype.kind == "i"                     # integer dtype
    assert len(labels) == 3                             # one per input row


def test_sparse_and_dense_inputs_agree():
    matrix = _csr([SIM_A, SIM_B, DISTINCT, SIM_A, SIM_B])
    sparse_labels = cluster_matrix(matrix, distance_threshold=0.5)
    dense_labels = cluster_matrix(matrix.toarray(), distance_threshold=0.5)
    assert np.array_equal(sparse_labels, dense_labels)


def test_deterministic_across_runs():
    matrix = _csr([SIM_A, SIM_B, DISTINCT, SIM_A])
    a = cluster_matrix(matrix, distance_threshold=0.5)
    b = cluster_matrix(matrix, distance_threshold=0.5)
    assert np.array_equal(a, b)


def test_row_order_preserved():
    # Identical duplicate rows must receive the identical label regardless of
    # position, proving labels follow input order.
    matrix = _csr([SIM_A, DISTINCT, SIM_A, DISTINCT])
    labels = cluster_matrix(matrix, distance_threshold=0.5)
    assert labels[0] == labels[2]
    assert labels[1] == labels[3]
    assert labels[0] != labels[1]


# --- Edge cases ---------------------------------------------------------------
def test_empty_dense_matrix_returns_empty_integer_labels():
    labels = cluster_matrix(np.zeros((0, 5)))
    assert isinstance(labels, np.ndarray)
    assert len(labels) == 0
    assert labels.dtype.kind == "i"


def test_empty_sparse_matrix_returns_empty_integer_labels():
    labels = cluster_matrix(csr_matrix((0, 5)))
    assert len(labels) == 0
    assert labels.dtype.kind == "i"


def test_single_sample_returns_zero_label():
    labels = cluster_matrix(np.array([[0.5, 0.1, 0.9]]))
    assert np.array_equal(labels, np.array([0]))


def test_single_sparse_sample_returns_zero_label():
    labels = cluster_matrix(_csr([SIM_A]))
    assert np.array_equal(labels, np.array([0]))


def test_multi_row_zero_feature_matrix_raises():
    with pytest.raises(ValueError, match="0 features"):
        cluster_matrix(np.zeros((3, 0)))


def test_all_blank_messages_raise_via_dataframe():
    # TF-IDF of all-blank text -> (n x 0) matrix -> actionable error, not labels.
    df = pd.DataFrame({"line_id": [0, 1, 2], NORMALIZED_COLUMN: ["", "   ", ""]})
    with pytest.raises(ValueError):
        cluster_dataframe(df)


# --- Threshold validation -----------------------------------------------------
@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5, float("nan"), float("inf"), float("-inf")])
def test_non_positive_or_nonfinite_threshold_rejected(bad):
    with pytest.raises(ValueError):
        cluster_matrix(_csr([SIM_A, SIM_B, DISTINCT]), distance_threshold=bad)


@pytest.mark.parametrize("bad", [True, False, "0.5", None, [0.5]])
def test_non_numeric_threshold_rejected(bad):
    with pytest.raises(ValueError):
        build_clusterer(distance_threshold=bad)


# --- Matrix validation --------------------------------------------------------
def test_one_dimensional_input_rejected():
    with pytest.raises(ValueError, match="2-dimensional"):
        cluster_matrix(np.array([0.1, 0.2, 0.3]))


def test_three_dimensional_input_rejected():
    with pytest.raises(ValueError, match="2-dimensional"):
        cluster_matrix(np.zeros((2, 2, 2)))


def test_nonfinite_values_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        cluster_matrix(np.array([[0.1, np.nan], [0.2, 0.3]]))


def test_infinite_values_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        cluster_matrix(np.array([[0.1, np.inf], [0.2, 0.3]]))


def test_non_numeric_matrix_rejected():
    with pytest.raises(ValueError, match="numeric"):
        cluster_matrix(np.array([["a", "b"], ["c", "d"]]))


def test_incompatible_metric_linkage_not_silently_corrected():
    # ward requires euclidean; pairing it with cosine must raise, not be fixed.
    est = build_clusterer(metric="cosine", linkage="ward")
    with pytest.raises(ValueError):
        est.fit_predict(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.1]]))


# --- cluster_dataframe --------------------------------------------------------
def test_dataframe_appends_column_preserves_original_and_no_mutation():
    matrix_docs = [SIM_A, SIM_B, DISTINCT, SIM_A]
    df = pd.DataFrame(
        {"line_id": [0, 1, 2, 3], "message": matrix_docs, NORMALIZED_COLUMN: matrix_docs}
    )
    before = df.copy(deep=True)

    out = cluster_dataframe(df)

    assert CLUSTER_COLUMN in out.columns
    # Appended last; every original column still present, unchanged, in order.
    assert list(out.columns) == list(before.columns) + [CLUSTER_COLUMN]
    assert out[list(before.columns)].equals(before)
    assert len(out) == len(df)
    # Input frame was not mutated in place.
    assert CLUSTER_COLUMN not in df.columns
    assert df.equals(before)


def test_dataframe_preserves_nondefault_index_and_order():
    docs = [SIM_A, SIM_B, DISTINCT]
    idx = [10, 20, 30]
    df = pd.DataFrame({NORMALIZED_COLUMN: docs}, index=idx)
    out = cluster_dataframe(df)
    assert list(out.index) == idx                       # index preserved
    assert len(out) == 3


def test_empty_dataframe_returns_empty_integer_cluster_column():
    df = pd.DataFrame({"line_id": [], "message": [], NORMALIZED_COLUMN: []})
    out = cluster_dataframe(df)
    assert CLUSTER_COLUMN in out.columns
    assert len(out) == 0
    assert out[CLUSTER_COLUMN].dtype.kind == "i"


def test_one_row_dataframe_label_zero():
    df = pd.DataFrame({"line_id": [0], NORMALIZED_COLUMN: [SIM_A]})
    out = cluster_dataframe(df)
    assert list(out[CLUSTER_COLUMN]) == [0]


def test_dataframe_labels_align_positionally_with_line_id():
    docs = [SIM_A, SIM_B, DISTINCT, SIM_A, DISTINCT]
    df = pd.DataFrame({"line_id": list(range(5)), NORMALIZED_COLUMN: docs})
    out = cluster_dataframe(df, distance_threshold=0.5)
    # Equal messages -> equal labels at the same positions.
    assert out.loc[0, CLUSTER_COLUMN] == out.loc[3, CLUSTER_COLUMN]
    assert out.loc[2, CLUSTER_COLUMN] == out.loc[4, CLUSTER_COLUMN]
    assert out.loc[0, CLUSTER_COLUMN] != out.loc[2, CLUSTER_COLUMN]


# --- summarize_clusters -------------------------------------------------------
def test_summarize_columns_sizes_and_order():
    docs = [SIM_A, SIM_A, SIM_A, SIM_B, DISTINCT]
    df = pd.DataFrame({"line_id": list(range(5)), NORMALIZED_COLUMN: docs})
    clustered = cluster_dataframe(df, distance_threshold=0.5)
    summary = summarize_clusters(clustered)

    assert list(summary.columns) == [CLUSTER_COLUMN, "size", "representative_message"]
    # Sizes must account for every row exactly once.
    assert summary["size"].sum() == len(df)
    # Deterministic ordering: size descending, then cluster_id ascending.
    sizes = summary["size"].tolist()
    assert sizes == sorted(sizes, reverse=True)
    # The largest cluster's representative is one of its actual messages.
    assert summary.iloc[0]["representative_message"] in set(docs)


def test_summarize_representative_is_most_frequent():
    df = pd.DataFrame({NORMALIZED_COLUMN: [SIM_A, SIM_A, DISTINCT, DISTINCT, DISTINCT]})
    clustered = cluster_dataframe(df, distance_threshold=0.5)
    summary = summarize_clusters(clustered)
    # The DISTINCT message appears 3x, SIM_A 2x; each cluster's representative is
    # its own most frequent message.
    reps = dict(zip(summary[CLUSTER_COLUMN], summary["representative_message"]))
    for cid, rep in reps.items():
        members = clustered.loc[clustered[CLUSTER_COLUMN] == cid, NORMALIZED_COLUMN]
        assert rep == members.value_counts().idxmax()


def test_summarize_empty_returns_documented_columns():
    empty = pd.DataFrame({CLUSTER_COLUMN: [], "size": [], "representative_message": []})
    clustered_empty = pd.DataFrame(
        {NORMALIZED_COLUMN: [], CLUSTER_COLUMN: pd.Series([], dtype="int64")}
    )
    summary = summarize_clusters(clustered_empty)
    assert list(summary.columns) == [CLUSTER_COLUMN, "size", "representative_message"]
    assert len(summary) == 0


def test_summarize_blank_messages_use_fallback():
    # One cluster whose messages are all blank -> representative falls back to "".
    clustered = pd.DataFrame(
        {NORMALIZED_COLUMN: ["", "   "], CLUSTER_COLUMN: [0, 0]}
    )
    summary = summarize_clusters(clustered)
    assert len(summary) == 1
    assert summary.iloc[0]["representative_message"] == ""


def test_summarize_does_not_mutate_input():
    docs = [SIM_A, SIM_B, DISTINCT]
    df = pd.DataFrame({NORMALIZED_COLUMN: docs})
    clustered = cluster_dataframe(df)
    snapshot = clustered.copy(deep=True)
    summarize_clusters(clustered)
    assert clustered.equals(snapshot)


def test_summarize_missing_cluster_column_raises():
    with pytest.raises(KeyError):
        summarize_clusters(pd.DataFrame({NORMALIZED_COLUMN: [SIM_A]}))


# --- Integration with the existing 1,000-row dataset --------------------------
@needs_generated_data
def test_full_dataset_clustering_alignment_and_counts():
    df = normalize_dataframe(parse_file(config.RAW_LOG_PATH))
    clustered = cluster_dataframe(df)

    # Exactly 1,000 rows and labels; nothing dropped or reordered.
    assert len(clustered) == 1000
    assert len(clustered[CLUSTER_COLUMN]) == 1000
    assert clustered["line_id"].tolist() == list(range(1000))
    # Every row received a label.
    assert clustered[CLUSTER_COLUMN].notna().all()
    assert clustered[CLUSTER_COLUMN].dtype.kind == "i"

    # Original messages survived untouched.
    assert clustered["message"].equals(df["message"])
    assert clustered[NORMALIZED_COLUMN].equals(df[NORMALIZED_COLUMN])

    # Cluster count is meaningful: more than one, at most one per row.
    n_clusters = clustered[CLUSTER_COLUMN].nunique()
    assert 1 < n_clusters <= len(clustered)


@needs_ground_truth
def test_full_dataset_ground_truth_alignment_preserved():
    df = normalize_dataframe(parse_file(config.RAW_LOG_PATH))
    clustered = cluster_dataframe(df)
    truth = pd.read_csv(config.GROUND_TRUTH_PATH)

    merged = clustered.merge(truth, on="line_id", suffixes=("_p", "_t"))
    assert len(merged) == len(truth) == 1000
    assert (merged["message_p"] == merged["message_t"]).all()
