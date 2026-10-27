"""Day 10 focused tests for cluster purity and ranking evaluation."""

from unittest import mock

import pandas as pd
import pytest

from src import config
from src.clusterer import CLUSTER_COLUMN
from src.pipeline import run_pipeline
from src.ranking_evaluation import (
    PER_CLUSTER_COLUMNS,
    RANKING_REQUIRED_COLUMNS,
    cluster_purity,
    evaluate_ranking,
    per_cluster_purity,
)

needs_generated_data = pytest.mark.skipif(
    not config.RAW_LOG_PATH.exists(), reason="run `python data_generator.py` first"
)


# --- Fixtures -----------------------------------------------------------------
def _gt(line_ids, categories):
    return pd.DataFrame({"line_id": line_ids, "category": categories})


def _clustered(line_ids, cluster_ids):
    return pd.DataFrame({"line_id": line_ids, CLUSTER_COLUMN: cluster_ids})


def _ranking(cluster_ids, ranks):
    return pd.DataFrame({CLUSTER_COLUMN: cluster_ids, "rank": ranks})


# A 6-log fixture with three clusters of different purity:
#   cluster 0 -> {a, a, b}  (majority a, purity 2/3)
#   cluster 1 -> {c, c}     (pure)
#   cluster 2 -> {d}        (pure)
RK_IDS = [0, 1, 2, 3, 4, 5]
RK_CLUSTERS = [0, 0, 0, 1, 1, 2]
RK_CATEGORIES = ["a", "a", "b", "c", "c", "d"]


def _rk_clustered():
    return _clustered(RK_IDS, RK_CLUSTERS)


def _rk_ground_truth():
    return _gt(RK_IDS, RK_CATEGORIES)


def _rk_ranking():
    # cluster 0 is largest -> rank 1; then cluster 1 -> rank 2; cluster 2 -> rank 3.
    return _ranking([0, 1, 2], [1, 2, 3])


# --- A. cluster_purity --------------------------------------------------------
def test_purity_perfect_is_one():
    gt = _gt([0, 1, 2, 3], ["a", "a", "b", "b"])
    clustered = _clustered([0, 1, 2, 3], [0, 0, 1, 1])
    result = cluster_purity(gt, clustered)
    assert result["purity"] == pytest.approx(1.0)
    assert result["n_clusters"] == 2
    assert result["n_logs"] == 4


def test_purity_known_mixed_value():
    # One cluster holding {a, a, b, b}: majority count 2 of 4 -> purity 0.5.
    gt = _gt([0, 1, 2, 3], ["a", "a", "b", "b"])
    clustered = _clustered([0, 1, 2, 3], [0, 0, 0, 0])
    result = cluster_purity(gt, clustered)
    assert result["purity"] == pytest.approx(0.5)
    assert result["n_clusters"] == 1
    assert result["n_logs"] == 4


def test_purity_reports_cluster_count_alongside():
    gt = _gt([0, 1, 2, 3], ["a", "a", "b", "b"])
    clustered = _clustered([0, 1, 2, 3], [0, 0, 0, 0])
    result = cluster_purity(gt, clustered)
    assert set(result.keys()) == {"purity", "n_clusters", "n_logs"}
    assert isinstance(result["purity"], float)
    assert isinstance(result["n_clusters"], int)


def test_purity_invariant_to_cluster_id_renaming():
    gt = _gt([0, 1, 2, 3], ["a", "a", "b", "b"])
    original = cluster_purity(gt, _clustered([0, 1, 2, 3], [0, 0, 1, 1]))
    renamed = cluster_purity(gt, _clustered([0, 1, 2, 3], [7, 7, 3, 3]))
    assert original["purity"] == pytest.approx(renamed["purity"])
    assert original["n_clusters"] == renamed["n_clusters"]


def test_splitting_increases_purity_but_reports_more_clusters():
    # Purity is degenerate on its own: the finer split scores higher, which is
    # exactly why n_clusters must be reported next to it.
    gt = _gt([0, 1, 2, 3], ["a", "a", "b", "b"])
    coarse = cluster_purity(gt, _clustered([0, 1, 2, 3], [0, 0, 0, 0]))
    fine = cluster_purity(gt, _clustered([0, 1, 2, 3], [0, 0, 1, 1]))
    assert fine["purity"] > coarse["purity"]
    assert fine["n_clusters"] > coarse["n_clusters"]


def test_purity_empty_input_raises():
    empty_gt = _gt([], [])
    empty_clustered = _clustered([], [])
    with pytest.raises(ValueError):
        cluster_purity(empty_gt, empty_clustered)
    with pytest.raises(ValueError):
        cluster_purity(_rk_ground_truth(), empty_clustered)
    with pytest.raises(ValueError):
        cluster_purity(empty_gt, _rk_clustered())


def test_purity_duplicate_identifiers_raise():
    gt = _gt([0, 0, 1], ["a", "b", "c"])
    clustered = _clustered([0, 0, 1], [0, 1, 2])
    with pytest.raises(ValueError, match="duplicate"):
        cluster_purity(gt, clustered)


def test_purity_null_identifiers_raise():
    gt = _gt([0, None, 2], ["a", "b", "c"])
    clustered = _clustered([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match="missing"):
        cluster_purity(gt, clustered)


def test_purity_mismatched_id_sets_raise():
    gt = _gt([0, 1, 2], ["a", "b", "c"])
    clustered = _clustered([0, 1, 3], [0, 1, 2])
    with pytest.raises(ValueError, match="differ"):
        cluster_purity(gt, clustered)


def test_purity_missing_columns_raise():
    gt = _gt([0, 1], ["a", "b"])
    clustered = pd.DataFrame({"line_id": [0, 1]})  # no cluster_id
    with pytest.raises(KeyError):
        cluster_purity(gt, clustered)


def test_purity_non_dataframe_raises():
    with pytest.raises(TypeError):
        cluster_purity([("a", 0)], _rk_clustered())


def test_purity_does_not_mutate_inputs():
    gt, clustered = _rk_ground_truth(), _rk_clustered()
    gt_snap, cl_snap = gt.copy(deep=True), clustered.copy(deep=True)
    cluster_purity(gt, clustered)
    per_cluster_purity(gt, clustered)
    assert gt.equals(gt_snap)
    assert clustered.equals(cl_snap)


# --- per_cluster_purity -------------------------------------------------------
def test_per_cluster_purity_columns_and_values():
    detail = per_cluster_purity(_rk_ground_truth(), _rk_clustered())
    assert list(detail.columns) == PER_CLUSTER_COLUMNS
    # Ordered by size desc then cluster_id asc: cluster 0 (3), 1 (2), 2 (1).
    assert detail[CLUSTER_COLUMN].tolist() == [0, 1, 2]
    assert detail["size"].tolist() == [3, 2, 1]
    assert detail["majority_category"].tolist() == ["a", "c", "d"]
    assert detail["majority_count"].tolist() == [2, 2, 1]
    assert detail["purity"].iloc[0] == pytest.approx(2 / 3)
    assert detail["purity"].iloc[1] == pytest.approx(1.0)


def test_per_cluster_purity_sizes_match_overall_purity():
    detail = per_cluster_purity(_rk_ground_truth(), _rk_clustered())
    overall = cluster_purity(_rk_ground_truth(), _rk_clustered())
    # Sum of majority counts / total == overall purity.
    assert detail["majority_count"].sum() / detail["size"].sum() == pytest.approx(
        overall["purity"]
    )


def test_per_cluster_purity_deterministic_order():
    first = per_cluster_purity(_rk_ground_truth(), _rk_clustered())
    second = per_cluster_purity(_rk_ground_truth(), _rk_clustered())
    assert first.equals(second)


# --- B. evaluate_ranking ------------------------------------------------------
def test_evaluate_ranking_known_top_k_composition():
    result = evaluate_ranking(_rk_ranking(), _rk_ground_truth(), _rk_clustered(), top_k=2)
    # Top-2 by rank -> clusters 0 and 1 -> categories a:2, b:1, c:2, d:0.
    assert result["n_clusters_available"] == 3
    assert result["n_clusters_evaluated"] == 2
    assert result["n_logs_evaluated"] == 5
    assert result["n_logs_total"] == 6
    assert result["coverage"] == pytest.approx(5 / 6)
    assert result["category_counts"] == {"a": 2, "b": 1, "c": 2, "d": 0}
    assert result["category_proportions"]["a"] == pytest.approx(2 / 5)
    assert sum(result["category_proportions"].values()) == pytest.approx(1.0)


def test_evaluate_ranking_top_k_one():
    result = evaluate_ranking(_rk_ranking(), _rk_ground_truth(), _rk_clustered(), top_k=1)
    assert result["n_clusters_evaluated"] == 1
    assert result["n_logs_evaluated"] == 3          # cluster 0 has 3 logs
    assert result["coverage"] == pytest.approx(0.5)
    assert result["category_counts"] == {"a": 2, "b": 1, "c": 0, "d": 0}


def test_evaluate_ranking_top_k_exceeds_available_evaluates_all():
    result = evaluate_ranking(_rk_ranking(), _rk_ground_truth(), _rk_clustered(), top_k=99)
    assert result["n_clusters_evaluated"] == 3
    assert result["n_logs_evaluated"] == 6
    assert result["coverage"] == pytest.approx(1.0)


def test_evaluate_ranking_ties_broken_by_cluster_id():
    # Two clusters share rank 1; the lower cluster_id must be selected first.
    ranking = _ranking([1, 0], [1, 1])
    result = evaluate_ranking(ranking, _rk_ground_truth(), _rk_clustered(), top_k=1)
    # cluster 0 (size 3, {a,a,b}) is chosen, not cluster 1 (size 2, {c,c}).
    assert result["category_counts"] == {"a": 2, "b": 1, "c": 0, "d": 0}


def test_evaluate_ranking_invalid_top_k_raises():
    for bad in (0, -1, True, False, 2.5, "2", None):
        with pytest.raises(ValueError):
            evaluate_ranking(_rk_ranking(), _rk_ground_truth(), _rk_clustered(), top_k=bad)


def test_evaluate_ranking_missing_columns_raise():
    bad_ranking = pd.DataFrame({CLUSTER_COLUMN: [0, 1, 2]})  # no "rank"
    with pytest.raises(ValueError, match="missing"):
        evaluate_ranking(bad_ranking, _rk_ground_truth(), _rk_clustered(), top_k=1)


def test_evaluate_ranking_unknown_cluster_id_raises():
    ranking = _ranking([0, 99], [1, 2])  # cluster 99 does not exist
    with pytest.raises(ValueError, match="absent"):
        evaluate_ranking(ranking, _rk_ground_truth(), _rk_clustered(), top_k=2)


def test_evaluate_ranking_empty_inputs_raise():
    empty_ranking = _ranking([], [])
    with pytest.raises(ValueError):
        evaluate_ranking(empty_ranking, _rk_ground_truth(), _rk_clustered(), top_k=1)
    with pytest.raises(ValueError):
        evaluate_ranking(_rk_ranking(), _gt([], []), _rk_clustered(), top_k=1)
    with pytest.raises(ValueError):
        evaluate_ranking(_rk_ranking(), _rk_ground_truth(), _clustered([], []), top_k=1)


def test_evaluate_ranking_non_dataframe_raises():
    with pytest.raises(TypeError):
        evaluate_ranking("nope", _rk_ground_truth(), _rk_clustered(), top_k=1)


def test_evaluate_ranking_mismatched_ids_raise():
    gt = _gt([0, 1, 2, 3, 4], RK_CATEGORIES[:5])  # missing line_id 5
    with pytest.raises(ValueError, match="differ"):
        evaluate_ranking(_rk_ranking(), gt, _rk_clustered(), top_k=2)


def test_evaluate_ranking_duplicate_ids_raise():
    clustered = _clustered([0, 0, 2, 3, 4, 5], RK_CLUSTERS)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_ranking(_rk_ranking(), _rk_ground_truth(), clustered, top_k=2)


def test_evaluate_ranking_is_deterministic():
    first = evaluate_ranking(_rk_ranking(), _rk_ground_truth(), _rk_clustered(), top_k=2)
    second = evaluate_ranking(_rk_ranking(), _rk_ground_truth(), _rk_clustered(), top_k=2)
    assert first == second


def test_evaluate_ranking_does_not_mutate_inputs():
    ranking, gt, clustered = _rk_ranking(), _rk_ground_truth(), _rk_clustered()
    r_snap, g_snap, c_snap = (
        ranking.copy(deep=True),
        gt.copy(deep=True),
        clustered.copy(deep=True),
    )
    evaluate_ranking(ranking, gt, clustered, top_k=2)
    assert ranking.equals(r_snap)
    assert gt.equals(g_snap)
    assert clustered.equals(c_snap)


def test_evaluate_ranking_calls_no_clustering():
    # Ranking evaluation must consume existing labels only - never re-cluster.
    with mock.patch("src.evaluation.cluster_dataframe") as eval_spy, mock.patch(
        "src.clusterer.cluster_dataframe"
    ) as clus_spy:
        evaluate_ranking(_rk_ranking(), _rk_ground_truth(), _rk_clustered(), top_k=2)
        cluster_purity(_rk_ground_truth(), _rk_clustered())
    assert eval_spy.call_count == 0
    assert clus_spy.call_count == 0


def test_ranking_required_columns_constant():
    assert RANKING_REQUIRED_COLUMNS == [CLUSTER_COLUMN, "rank"]


# --- Integration with the real 1,000-row generated dataset --------------------
@needs_generated_data
def test_real_dataset_purity_and_ranking_are_consistent():
    result = run_pipeline(config.RAW_LOG_PATH)
    truth = pd.read_csv(config.GROUND_TRUTH_PATH)

    purity = cluster_purity(truth, result.clustered)
    assert 0.0 <= purity["purity"] <= 1.0
    assert purity["n_logs"] == 1000
    assert purity["n_clusters"] == result.n_clusters

    detail = per_cluster_purity(truth, result.clustered)
    assert len(detail) == result.n_clusters
    assert detail["size"].sum() == 1000
    assert detail["purity"].between(0.0, 1.0).all()

    top = evaluate_ranking(result.ranking, truth, result.clustered, top_k=5)
    assert top["n_clusters_available"] == result.n_clusters
    assert top["n_clusters_evaluated"] == min(5, result.n_clusters)
    assert 0.0 <= top["coverage"] <= 1.0
    assert top["n_logs_evaluated"] <= top["n_logs_total"] == 1000
    assert sum(top["category_counts"].values()) == top["n_logs_evaluated"]
    assert sum(top["category_proportions"].values()) == pytest.approx(1.0)
