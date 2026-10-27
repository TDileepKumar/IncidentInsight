"""Day 7 focused tests for clustering evaluation."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.clusterer import CLUSTER_COLUMN, cluster_dataframe, summarize_clusters
from src.evaluation import (
    THRESHOLD_REPORT_COLUMNS,
    build_cluster_crosstab,
    cluster_composition,
    compare_thresholds,
    evaluate_clustering,
)
from src.log_normalizer import NORMALIZED_COLUMN, normalize_dataframe
from src.log_parser import parse_file

needs_generated_data = pytest.mark.skipif(
    not config.RAW_LOG_PATH.exists(), reason="run `python data_generator.py` first"
)

# A controlled fixture with DISJOINT vocabulary groups: identical strings within
# a group (cosine distance 0) and no shared tokens across groups (cosine
# distance 1.0). This makes the clustering stable at every threshold in
# (0.3, 0.5, 0.7): {0,1}, {2,3}, {4} -> 3 clusters, 1 singleton, perfect
# agreement with the categories below.
FIXTURE_MESSAGES = [
    "alpha beta gamma",
    "alpha beta gamma",
    "delta epsilon zeta",
    "delta epsilon zeta",
    "eta theta iota",
]
FIXTURE_CATEGORIES = ["cat_a", "cat_a", "cat_b", "cat_b", "cat_c"]
FIXTURE_IDS = [0, 1, 2, 3, 4]


def _fixture_df():
    return pd.DataFrame({"line_id": FIXTURE_IDS, NORMALIZED_COLUMN: FIXTURE_MESSAGES})


def _fixture_ground_truth():
    return pd.DataFrame({"line_id": FIXTURE_IDS, "category": FIXTURE_CATEGORIES})


# --- A. evaluate_clustering ---------------------------------------------------
def test_perfect_agreement_scores_one():
    scores = evaluate_clustering([0, 0, 1, 1], [0, 0, 1, 1])
    assert set(scores.keys()) == {"ari", "ami"}
    assert scores["ari"] == pytest.approx(1.0)
    assert scores["ami"] == pytest.approx(1.0)
    assert isinstance(scores["ari"], float) and isinstance(scores["ami"], float)


def test_cluster_renaming_does_not_change_perfect_agreement():
    # Same partition, arbitrary different cluster IDs -> still perfect.
    scores = evaluate_clustering([0, 0, 1, 1], [7, 7, 3, 3])
    assert scores["ari"] == pytest.approx(1.0)
    assert scores["ami"] == pytest.approx(1.0)


def test_known_non_perfect_grouping_is_valid():
    scores = evaluate_clustering([0, 0, 1, 1], [0, 0, 0, 1])
    assert -1.0 <= scores["ari"] <= 1.0
    assert scores["ari"] < 1.0            # not a perfect partition
    assert 0.0 <= scores["ami"] <= 1.0


def test_accepts_series_and_array_inputs():
    s_true = pd.Series([0, 0, 1, 1])
    a_pred = np.array([5, 5, 9, 9])
    scores = evaluate_clustering(s_true, a_pred)
    assert scores["ari"] == pytest.approx(1.0)


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        evaluate_clustering([], [])
    with pytest.raises(ValueError):
        evaluate_clustering([], [1, 2])
    with pytest.raises(ValueError):
        evaluate_clustering([1, 2], [])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate_clustering([0, 0, 1], [0, 1])


@pytest.mark.parametrize("bad", [None, "abc", np.zeros((2, 2))])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        evaluate_clustering(bad, [0, 1])


# --- B. build_cluster_crosstab ------------------------------------------------
def test_crosstab_counts_are_correct():
    gt = pd.DataFrame({"line_id": [0, 1, 2, 3], "category": ["a", "a", "b", "c"]})
    clustered = pd.DataFrame({"line_id": [0, 1, 2, 3], CLUSTER_COLUMN: [0, 0, 1, 2]})
    table = build_cluster_crosstab(gt, clustered)

    assert list(table.index) == ["a", "b", "c"]
    assert table.loc["a", 0] == 2
    assert table.loc["b", 1] == 1
    assert table.loc["c", 2] == 1
    assert table.values.sum() == 4


def test_crosstab_aligns_by_id_not_position():
    gt = pd.DataFrame({"line_id": [0, 1, 2, 3], "category": ["a", "a", "b", "c"]})
    ordered = pd.DataFrame({"line_id": [0, 1, 2, 3], CLUSTER_COLUMN: [0, 0, 1, 2]})
    shuffled = pd.DataFrame({"line_id": [2, 0, 3, 1], CLUSTER_COLUMN: [1, 0, 2, 0]})

    t_ordered = build_cluster_crosstab(gt, ordered)
    t_shuffled = build_cluster_crosstab(gt, shuffled)
    # Identical id->cluster mapping, different row order -> identical table.
    assert t_ordered.equals(t_shuffled)


def test_crosstab_duplicate_identifiers_raise():
    gt = pd.DataFrame({"line_id": [0, 1, 2], "category": ["a", "b", "c"]})
    clustered = pd.DataFrame({"line_id": [0, 0, 1], CLUSTER_COLUMN: [0, 1, 2]})
    with pytest.raises(ValueError, match="duplicate"):
        build_cluster_crosstab(gt, clustered)


def test_crosstab_missing_identifiers_raise():
    gt = pd.DataFrame({"line_id": [0, 1, 2], "category": ["a", "b", "c"]})
    clustered = pd.DataFrame({"line_id": [0, None, 2], CLUSTER_COLUMN: [0, 1, 2]})
    with pytest.raises(ValueError, match="missing"):
        build_cluster_crosstab(gt, clustered)


def test_crosstab_different_id_sets_raise():
    gt = pd.DataFrame({"line_id": [0, 1, 2], "category": ["a", "b", "c"]})
    clustered = pd.DataFrame({"line_id": [0, 1, 3], CLUSTER_COLUMN: [0, 1, 2]})
    with pytest.raises(ValueError, match="differ"):
        build_cluster_crosstab(gt, clustered)


def test_crosstab_missing_columns_raise():
    gt = pd.DataFrame({"line_id": [0, 1], "category": ["a", "b"]})
    clustered = pd.DataFrame({"line_id": [0, 1]})  # no cluster_id column
    with pytest.raises(KeyError):
        build_cluster_crosstab(gt, clustered)


def test_crosstab_does_not_mutate_inputs():
    gt = pd.DataFrame({"line_id": [0, 1, 2], "category": ["a", "a", "b"]})
    clustered = pd.DataFrame({"line_id": [0, 1, 2], CLUSTER_COLUMN: [0, 0, 1]})
    gt_snap, cl_snap = gt.copy(deep=True), clustered.copy(deep=True)
    build_cluster_crosstab(gt, clustered)
    assert gt.equals(gt_snap)
    assert clustered.equals(cl_snap)


# --- C. cluster_composition ---------------------------------------------------
def test_cluster_composition_columns_and_sizes():
    clustered = cluster_dataframe(_fixture_df(), distance_threshold=0.5)
    summary = cluster_composition(clustered)

    assert list(summary.columns) == [CLUSTER_COLUMN, "size", "representative_message"]
    assert summary["size"].sum() == len(_fixture_df())
    assert sorted(summary["size"].tolist()) == [1, 2, 2]   # {0,1},{2,3},{4}


def test_cluster_composition_matches_summarize_clusters():
    clustered = cluster_dataframe(_fixture_df(), distance_threshold=0.5)
    assert cluster_composition(clustered).equals(summarize_clusters(clustered))


def test_cluster_composition_empty_returns_documented_columns():
    empty = pd.DataFrame(
        {NORMALIZED_COLUMN: [], CLUSTER_COLUMN: pd.Series([], dtype="int64")}
    )
    summary = cluster_composition(empty)
    assert list(summary.columns) == [CLUSTER_COLUMN, "size", "representative_message"]
    assert len(summary) == 0


def test_cluster_composition_missing_cluster_column_raises():
    with pytest.raises(KeyError):
        cluster_composition(pd.DataFrame({NORMALIZED_COLUMN: ["alpha beta"]}))


def test_cluster_composition_does_not_mutate_input():
    clustered = cluster_dataframe(_fixture_df(), distance_threshold=0.5)
    snap = clustered.copy(deep=True)
    cluster_composition(clustered)
    assert clustered.equals(snap)


# --- D. compare_thresholds ----------------------------------------------------
def test_compare_thresholds_one_row_per_threshold_and_columns():
    result = compare_thresholds(_fixture_df(), _fixture_ground_truth(), thresholds=(0.3, 0.5, 0.7))
    assert list(result.columns) == THRESHOLD_REPORT_COLUMNS
    assert len(result) == 3
    assert result["distance_threshold"].tolist() == [0.3, 0.5, 0.7]
    # ARI and AMI computed for every row.
    assert result["ari"].notna().all() and result["ami"].notna().all()


def test_compare_thresholds_controlled_counts():
    result = compare_thresholds(_fixture_df(), _fixture_ground_truth(), thresholds=(0.3, 0.5, 0.7))
    # Disjoint-vocabulary fixture -> 3 clusters, 1 singleton at every threshold.
    assert (result["n_clusters"] == 3).all()
    assert (result["n_singletons"] == 1).all()
    # Perfect agreement with the fixture categories.
    assert np.allclose(result["ari"], 1.0)
    assert np.allclose(result["ami"], 1.0)


def test_compare_thresholds_empty_collection_raises():
    with pytest.raises(ValueError):
        compare_thresholds(_fixture_df(), _fixture_ground_truth(), thresholds=())
    with pytest.raises(ValueError):
        compare_thresholds(_fixture_df(), _fixture_ground_truth(), thresholds=[])


@pytest.mark.parametrize("bad", [0, -0.5, float("nan"), float("inf"), True, "0.5"])
def test_compare_thresholds_invalid_threshold_raises(bad):
    with pytest.raises(ValueError):
        compare_thresholds(_fixture_df(), _fixture_ground_truth(), thresholds=(bad,))


def test_compare_thresholds_duplicate_identifiers_raise():
    df = _fixture_df()
    gt = pd.DataFrame({"line_id": [0, 0, 2, 3, 4], "category": FIXTURE_CATEGORIES})
    with pytest.raises(ValueError, match="duplicate"):
        compare_thresholds(df, gt, thresholds=(0.5,))


def test_compare_thresholds_mismatched_records_raise():
    df = _fixture_df()
    gt = pd.DataFrame({"line_id": [0, 1, 2, 3], "category": ["a", "a", "b", "c"]})  # missing id 4
    with pytest.raises(ValueError, match="differ"):
        compare_thresholds(df, gt, thresholds=(0.5,))


def test_compare_thresholds_does_not_mutate_inputs():
    df, gt = _fixture_df(), _fixture_ground_truth()
    df_snap, gt_snap = df.copy(deep=True), gt.copy(deep=True)
    compare_thresholds(df, gt, thresholds=(0.3, 0.5, 0.7))
    assert df.equals(df_snap)
    assert gt.equals(gt_snap)
    assert df[NORMALIZED_COLUMN].tolist() == FIXTURE_MESSAGES   # row order intact


# --- Integration with the 1,000-row dataset -----------------------------------
@needs_generated_data
def test_real_dataset_crosstab_and_scores_are_meaningful():
    df = normalize_dataframe(parse_file(config.RAW_LOG_PATH))
    clustered = cluster_dataframe(df)
    truth = pd.read_csv(config.GROUND_TRUTH_PATH)

    table = build_cluster_crosstab(truth, clustered)
    # Every one of the 1,000 logs is counted exactly once.
    assert table.values.sum() == 1000
    # All eight ground-truth categories appear as rows.
    assert set(table.index) == set(truth["category"].unique())

    # Align categories to the clustered rows by line_id, then score.
    categories = truth.set_index("line_id").loc[clustered["line_id"], "category"]
    scores = evaluate_clustering(categories, clustered[CLUSTER_COLUMN])
    assert -1.0 <= scores["ari"] <= 1.0
    assert 0.0 <= scores["ami"] <= 1.0


@needs_generated_data
def test_real_dataset_threshold_comparison_shape():
    df = normalize_dataframe(parse_file(config.RAW_LOG_PATH))
    truth = pd.read_csv(config.GROUND_TRUTH_PATH)

    result = compare_thresholds(df, truth, thresholds=(0.3, 0.5, 0.7))
    assert list(result.columns) == THRESHOLD_REPORT_COLUMNS
    assert len(result) == 3
    # Meaningful behavior, not fixed counts: >1 cluster, valid metric ranges.
    assert (result["n_clusters"] > 1).all()
    assert (result["n_clusters"] <= len(df)).all()
    assert result["ari"].between(-1.0, 1.0).all()
    assert result["ami"].between(0.0, 1.0).all()
