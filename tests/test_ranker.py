"""Focused tests for deterministic incident-cluster ranking."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.clusterer import CLUSTER_COLUMN, cluster_dataframe, summarize_clusters
from src.log_normalizer import NORMALIZED_COLUMN, normalize_dataframe
from src.log_parser import parse_file
from src.ranker import (
    DEFAULT_SEVERITY_WEIGHTS,
    OUTPUT_COLUMNS,
    rank_clusters,
)

needs_generated_data = pytest.mark.skipif(
    not config.RAW_LOG_PATH.exists(), reason="run `python data_generator.py` first"
)


def _make_df(rows):
    """Build a minimal clustered DataFrame from (cluster_id, level, message) rows."""
    return pd.DataFrame(
        {
            "line_id": range(len(rows)),
            CLUSTER_COLUMN: [r[0] for r in rows],
            "level": [r[1] for r in rows],
            NORMALIZED_COLUMN: [r[2] for r in rows],
        }
    )


# --- 1. Schema ----------------------------------------------------------------
def test_output_schema_and_column_order():
    df = _make_df([(0, "ERROR", "db timeout <NUM>"), (0, "ERROR", "db timeout <NUM>"),
                   (1, "WARNING", "cache miss")])
    out = rank_clusters(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == OUTPUT_COLUMNS


# --- 2. Sizes / representative reuse summarize_clusters -----------------------
def test_sizes_and_representatives_match_summarize_clusters():
    df = _make_df([(0, "ERROR", "alpha"), (0, "ERROR", "alpha"), (0, "ERROR", "beta"),
                   (1, "FATAL", "gamma"), (1, "FATAL", "gamma")])
    out = rank_clusters(df)
    summ = summarize_clusters(df).set_index(CLUSTER_COLUMN)
    ranked = out.set_index(CLUSTER_COLUMN)
    for cid in ranked.index:
        assert ranked.loc[cid, "size"] == summ.loc[cid, "size"]
        assert ranked.loc[cid, "representative_message"] == summ.loc[cid, "representative_message"]


def test_accepts_precomputed_summary():
    df = _make_df([(0, "ERROR", "alpha"), (1, "FATAL", "gamma")])
    summ = summarize_clusters(df)
    out = rank_clusters(df, summary=summ)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(out) == 2


# --- 3. Max severity aggregation ----------------------------------------------
def test_max_severity_within_cluster():
    # One cluster mixing WARNING, FATAL, INFO -> max is FATAL (5) -> 1.0
    df = _make_df([(0, "WARNING", "a"), (0, "FATAL", "a"), (0, "INFO", "a")])
    out = rank_clusters(df)
    assert out["severity_score"].iloc[0] == pytest.approx(1.0)


# --- 4. FATAL outranks WARNING at equal frequency -----------------------------
def test_fatal_outranks_warning_when_frequency_controlled():
    df = _make_df([(0, "WARNING", "w msg"), (1, "FATAL", "f msg")])  # both size 1
    out = rank_clusters(df)
    assert out.iloc[0][CLUSTER_COLUMN] == 1                  # FATAL cluster ranked first
    assert out.iloc[0]["severity_score"] > out.iloc[1]["severity_score"]


# --- 5. Both frequency and severity contribute --------------------------------
def test_frequency_and_severity_both_contribute_to_risk():
    df = _make_df([
        (0, "ERROR", "x"), (0, "ERROR", "x"), (0, "ERROR", "x"), (0, "ERROR", "x"),  # size 4, ERROR
        (1, "FATAL", "y"),                                                            # size 1, FATAL
    ])
    out = rank_clusters(df, w_freq=0.6, w_sev=0.4).set_index(CLUSTER_COLUMN)
    # max size = 4
    assert out.loc[0, "frequency_score"] == pytest.approx(1.0)      # 4/4
    assert out.loc[0, "severity_score"] == pytest.approx(3 / 5)     # ERROR
    assert out.loc[0, "risk_score"] == pytest.approx(0.6 * 1.0 + 0.4 * 0.6)
    assert out.loc[1, "frequency_score"] == pytest.approx(0.25)     # 1/4
    assert out.loc[1, "severity_score"] == pytest.approx(1.0)       # FATAL
    assert out.loc[1, "risk_score"] == pytest.approx(0.6 * 0.25 + 0.4 * 1.0)


# --- 6. Sorting + consecutive 1-based ranks -----------------------------------
def test_ranks_are_consecutive_and_sorted_by_risk():
    df = _make_df([(0, "ERROR", "a"), (0, "ERROR", "a"), (1, "FATAL", "b"),
                   (2, "WARNING", "c"), (2, "WARNING", "c"), (2, "WARNING", "c")])
    out = rank_clusters(df)
    assert out["rank"].tolist() == list(range(1, len(out) + 1))
    assert out["risk_score"].is_monotonic_decreasing


# --- 7. Deterministic tie-breaking (size, then cluster_id) ---------------------
def test_tie_break_by_size_desc_when_risk_equal():
    # w_freq=0 -> risk depends only on severity; equal severity -> size decides.
    df = _make_df([(0, "WARNING", "a"), (0, "WARNING", "a"), (0, "WARNING", "a"),  # size 3
                   (1, "WARNING", "b"),                                             # size 1
                   (2, "WARNING", "c"), (2, "WARNING", "c")])                       # size 2
    out = rank_clusters(df, w_freq=0.0, w_sev=1.0)
    assert out[CLUSTER_COLUMN].tolist() == [0, 2, 1]
    assert out["rank"].tolist() == [1, 2, 3]


def test_tie_break_by_cluster_id_asc_when_risk_and_size_equal():
    df = _make_df([(5, "WARNING", "a"), (3, "WARNING", "b")])  # both size 1
    out = rank_clusters(df, w_freq=0.0, w_sev=1.0)
    assert out[CLUSTER_COLUMN].tolist() == [3, 5]


# --- 8. Case-insensitive + whitespace-tolerant severity ------------------------
def test_severity_label_normalization():
    df = _make_df([(0, "  fatal  ", "a"), (1, "Warning", "b"), (2, "ERROR", "c")])
    out = rank_clusters(df).set_index(CLUSTER_COLUMN)
    assert out.loc[0, "severity_score"] == pytest.approx(1.0)    # FATAL 5/5
    assert out.loc[1, "severity_score"] == pytest.approx(0.4)    # WARNING 2/5
    assert out.loc[2, "severity_score"] == pytest.approx(0.6)    # ERROR 3/5


# --- 9. Unknown / missing severity -> weight 0, no crash ------------------------
def test_unknown_and_missing_severity_score_zero():
    df = _make_df([(0, "BANANA", "a"), (1, None, "b"), (2, "", "c")])
    out = rank_clusters(df).set_index(CLUSTER_COLUMN)
    assert out.loc[0, "severity_score"] == 0.0
    assert out.loc[1, "severity_score"] == 0.0
    assert out.loc[2, "severity_score"] == 0.0


# --- 10. Single-cluster and empty input ----------------------------------------
def test_single_cluster():
    df = _make_df([(0, "ERROR", "a"), (0, "ERROR", "b")])
    out = rank_clusters(df)
    assert len(out) == 1
    assert out.iloc[0]["frequency_score"] == pytest.approx(1.0)
    assert out.iloc[0]["rank"] == 1


def test_valid_empty_dataframe_returns_empty_result_with_schema():
    empty = pd.DataFrame(
        {
            "line_id": pd.Series(dtype="int64"),
            CLUSTER_COLUMN: pd.Series(dtype="int64"),
            "level": pd.Series(dtype="object"),
            NORMALIZED_COLUMN: pd.Series(dtype="object"),
        }
    )
    out = rank_clusters(empty)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(out) == 0


def test_equal_frequency_all_score_one():
    df = _make_df([(0, "ERROR", "a"), (1, "WARNING", "b"), (2, "FATAL", "c")])  # each size 1
    out = rank_clusters(df)
    assert (out["frequency_score"] == 1.0).all()


# --- 11. Validation errors -----------------------------------------------------
def test_missing_required_columns_raise_value_error():
    with pytest.raises(ValueError):
        rank_clusters(pd.DataFrame({"level": ["ERROR"], "x": [1]}))          # no cluster_id
    with pytest.raises(ValueError):
        rank_clusters(pd.DataFrame({CLUSTER_COLUMN: [0], NORMALIZED_COLUMN: ["a"]}))  # no level


def test_invalid_input_type_raises():
    with pytest.raises(TypeError):
        rank_clusters("not a dataframe")
    with pytest.raises(TypeError):
        rank_clusters([(0, "ERROR", "a")])


def test_malformed_summary_raises():
    df = _make_df([(0, "ERROR", "a"), (1, "WARNING", "b")])
    # missing a required summary column
    with pytest.raises(ValueError):
        rank_clusters(df, summary=pd.DataFrame({CLUSTER_COLUMN: [0, 1], "size": [1, 1]}))
    # cluster ids do not correspond to clustered_df
    with pytest.raises(ValueError):
        rank_clusters(df, summary=pd.DataFrame(
            {CLUSTER_COLUMN: [0, 99], "size": [1, 1], "representative_message": ["a", "b"]}))
    # sizes do not match the actual counts
    with pytest.raises(ValueError):
        rank_clusters(df, summary=pd.DataFrame(
            {CLUSTER_COLUMN: [0, 1], "size": [5, 5], "representative_message": ["a", "b"]}))
    # summary of the wrong type
    with pytest.raises(TypeError):
        rank_clusters(df, summary="nope")


@pytest.mark.parametrize("wf,ws", [(0.5, 0.4), (0.7, 0.6), (-0.1, 1.1), (1.5, -0.5)])
def test_weights_must_sum_to_one_and_be_nonnegative(wf, ws):
    df = _make_df([(0, "ERROR", "a"), (1, "WARNING", "b")])
    with pytest.raises(ValueError):
        rank_clusters(df, w_freq=wf, w_sev=ws)


def test_non_numeric_or_bool_weights_raise():
    df = _make_df([(0, "ERROR", "a"), (1, "WARNING", "b")])
    with pytest.raises(ValueError):
        rank_clusters(df, w_freq="0.6", w_sev=0.4)
    with pytest.raises(ValueError):
        rank_clusters(df, w_freq=True, w_sev=False)


# --- 12. Custom severity weights -----------------------------------------------
def test_custom_severity_weights():
    df = _make_df([(0, "ERROR", "a"), (1, "WARNING", "b")])
    out = rank_clusters(df, severity_weights={"ERROR": 10, "WARNING": 1}).set_index(CLUSTER_COLUMN)
    assert out.loc[0, "severity_score"] == pytest.approx(10 / 5)   # 2.0
    assert out.loc[1, "severity_score"] == pytest.approx(1 / 5)    # 0.2


def test_custom_weights_unknown_level_is_zero():
    df = _make_df([(0, "FATAL", "a"), (1, "BANANA", "b")])
    # Custom map omits BANANA entirely -> weight 0.
    out = rank_clusters(df, severity_weights={"FATAL": 5}).set_index(CLUSTER_COLUMN)
    assert out.loc[1, "severity_score"] == 0.0


def test_invalid_custom_severity_weights_raise():
    df = _make_df([(0, "ERROR", "a")])
    with pytest.raises(ValueError):
        rank_clusters(df, severity_weights={"ERROR": -1})
    with pytest.raises(TypeError):
        rank_clusters(df, severity_weights="not a mapping")


# --- 13. No mutation of inputs --------------------------------------------------
def test_inputs_are_not_mutated():
    df = _make_df([(0, "ERROR", "a"), (0, "ERROR", "a"), (1, "FATAL", "b")])
    summ = summarize_clusters(df)
    df_snap, summ_snap = df.copy(deep=True), summ.copy(deep=True)
    rank_clusters(df, summary=summ)
    assert df.equals(df_snap)
    assert summ.equals(summ_snap)


# --- 14. Every cluster appears exactly once -------------------------------------
def test_every_cluster_appears_exactly_once():
    df = _make_df([(0, "ERROR", "a"), (1, "WARNING", "b"), (2, "FATAL", "c"), (0, "ERROR", "a")])
    out = rank_clusters(df)
    assert out[CLUSTER_COLUMN].is_unique
    assert sorted(out[CLUSTER_COLUMN].tolist()) == [0, 1, 2]


def test_all_scores_are_finite_numeric():
    df = _make_df([(0, "ERROR", "a"), (1, None, "b"), (2, "FATAL", "c")])
    out = rank_clusters(df)
    score_cols = ["severity_score", "frequency_score", "risk_score"]
    assert np.isfinite(out[score_cols].to_numpy(dtype=float)).all()


def test_default_severity_mapping_is_as_documented():
    assert DEFAULT_SEVERITY_WEIGHTS == {
        "DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4, "FATAL": 5,
    }


# --- Integration with the 1,000-row dataset (no hardcoded scores) ---------------
@needs_generated_data
def test_real_dataset_ranks_all_clusters():
    df = normalize_dataframe(parse_file(config.RAW_LOG_PATH))
    clustered = cluster_dataframe(df)
    out = rank_clusters(clustered)

    n_clusters = clustered[CLUSTER_COLUMN].nunique()
    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(out) == n_clusters
    assert out[CLUSTER_COLUMN].is_unique
    assert out["rank"].tolist() == list(range(1, n_clusters + 1))
    assert out["size"].sum() == len(clustered)                     # every record counted once
    assert out["risk_score"].is_monotonic_decreasing
    assert np.isfinite(out[["severity_score", "frequency_score", "risk_score"]].to_numpy(dtype=float)).all()
