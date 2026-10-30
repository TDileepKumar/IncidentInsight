"""Focused tests for the deterministic incident explainer."""

import pandas as pd
import pytest

from src import config
from src.clusterer import CLUSTER_COLUMN
from src.incident_explainer import (
    COUNT_FIELD,
    GENERIC_CHECKS,
    MESSAGE_FIELD,
    RANK_FIELD,
    explain_incident,
    explain_ranked_incidents,
)
from src.pipeline import run_pipeline
from tests.conftest import needs_generated_data

RESULT_KEYS = {
    "cluster_id",
    "rank",
    "summary",
    "evidence",
    "suggested_checks",
    "limitations",
}


def _incident(**overrides):
    """A fully-populated, valid database-connection incident record."""
    base = {
        CLUSTER_COLUMN: 15,
        RANK_FIELD: 1,
        COUNT_FIELD: 76,
        MESSAGE_FIELD: "Could not connect to database server <HOST>: connection refused",
        "severity_score": 0.6,
        "frequency_score": 0.95,
        "risk_score": 0.81,
    }
    base.update(overrides)
    return base


# --- A. Single incident explanation -------------------------------------------
def test_valid_db_incident_returns_structured_dict():
    result = explain_incident(_incident())
    assert isinstance(result, dict)
    assert set(result.keys()) == RESULT_KEYS
    assert result["cluster_id"] == 15
    assert result["rank"] == 1


def test_cluster_id_is_preserved_verbatim():
    # A large / arbitrary id is copied as-is, never interpreted as importance.
    result = explain_incident(_incident(**{CLUSTER_COLUMN: 999}))
    assert result["cluster_id"] == 999


def test_occurrence_count_matches_supplied_record():
    result = explain_incident(_incident(**{COUNT_FIELD: 42}))
    assert result["evidence"]["occurrence_count"] == 42
    assert "42" in result["summary"]


def test_representative_message_is_preserved_as_evidence():
    msg = "Could not connect to database server <HOST>: connection refused"
    result = explain_incident(_incident(**{MESSAGE_FIELD: msg}))
    assert result["evidence"][MESSAGE_FIELD] == msg


def test_db_connection_suggested_checks_are_relevant_and_deterministic():
    result = explain_incident(_incident())
    checks = result["suggested_checks"]
    assert isinstance(checks, list) and checks
    joined = " ".join(checks).lower()
    assert "database" in joined or "connection" in joined
    # Deterministic: identical input -> identical checks.
    assert checks == explain_incident(_incident())["suggested_checks"]


def test_limitation_states_root_cause_unconfirmed():
    result = explain_incident(_incident())
    joined = " ".join(result["limitations"]).lower()
    assert "root cause is not confirmed" in joined


# --- B. Other message patterns ------------------------------------------------
@pytest.mark.parametrize(
    "message,expected_fragment",
    [
        ("Timeout waiting for database response from <HOST>", "database"),
        ("Slow query exceeded <NUM>s threshold on <HOST>", "query"),
        ("Payment gateway returned <NUM> for transaction <ID>", "payment gateway"),
        ("Authentication failed for user <USER>: invalid credentials", "authentication"),
        ("File not found: <PATH>", "file"),
        ("Upstream request timed out after <NUM>s", "request"),
    ],
)
def test_pattern_rules_match_expected_checks(message, expected_fragment):
    result = explain_incident(_incident(**{MESSAGE_FIELD: message}))
    assert expected_fragment in result["summary"].lower()
    assert result["suggested_checks"]


def test_unknown_message_gets_generic_checks_and_no_category():
    result = explain_incident(_incident(**{MESSAGE_FIELD: "zzz qqq unexpected thing"}))
    assert result["suggested_checks"] == GENERIC_CHECKS
    assert "unrecognized incident pattern" in result["summary"].lower()
    # No unsupported category is asserted.
    assert any("no specific category" in lim.lower() for lim in result["limitations"])


# --- C. Missing and invalid data ----------------------------------------------
def test_missing_message_yields_generic_checks_and_limitation():
    incident = _incident()
    del incident[MESSAGE_FIELD]
    result = explain_incident(incident)
    assert MESSAGE_FIELD not in result["evidence"]
    assert result["suggested_checks"] == GENERIC_CHECKS
    assert any("no representative message" in lim.lower() for lim in result["limitations"])


def test_empty_message_is_treated_as_unavailable():
    result = explain_incident(_incident(**{MESSAGE_FIELD: "   "}))
    assert MESSAGE_FIELD not in result["evidence"]
    assert result["suggested_checks"] == GENERIC_CHECKS


def test_missing_severity_is_not_fabricated():
    incident = _incident()
    del incident["severity_score"]
    result = explain_incident(incident)
    assert "severity_score" not in result["evidence"]
    assert any("no severity information" in lim.lower() for lim in result["limitations"])


def test_missing_count_is_not_invented():
    incident = _incident()
    del incident[COUNT_FIELD]
    result = explain_incident(incident)
    assert "occurrence_count" not in result["evidence"]
    assert any("occurrence count was not supplied" in lim.lower() for lim in result["limitations"])


def test_missing_rank_is_none_not_invented():
    incident = _incident()
    del incident[RANK_FIELD]
    result = explain_incident(incident)
    assert result["rank"] is None
    assert any("ranking position was not supplied" in lim.lower() for lim in result["limitations"])


@pytest.mark.parametrize("bad_count", [-1, -5, "abc", float("inf"), 2.5, True])
def test_invalid_count_raises(bad_count):
    with pytest.raises(ValueError):
        explain_incident(_incident(**{COUNT_FIELD: bad_count}))


def test_nan_count_is_treated_as_unavailable_not_invalid():
    # NaN means "missing" under the documented policy, so it is not invented and
    # does not raise - it simply produces an unavailable count.
    result = explain_incident(_incident(**{COUNT_FIELD: float("nan")}))
    assert "occurrence_count" not in result["evidence"]
    assert any("occurrence count was not supplied" in lim.lower() for lim in result["limitations"])


def test_missing_cluster_id_raises():
    incident = _incident()
    del incident[CLUSTER_COLUMN]
    with pytest.raises(ValueError, match="cluster_id"):
        explain_incident(incident)


def test_null_cluster_id_raises():
    with pytest.raises(ValueError, match="cluster_id"):
        explain_incident(_incident(**{CLUSTER_COLUMN: None}))


def test_non_string_message_raises():
    with pytest.raises(ValueError, match="representative_message"):
        explain_incident(_incident(**{MESSAGE_FIELD: 12345}))


@pytest.mark.parametrize("bad", [None, 42, "not-a-mapping", ["cluster_id"]])
def test_invalid_input_type_raises(bad):
    with pytest.raises(TypeError):
        explain_incident(bad)


def test_empty_mapping_raises():
    with pytest.raises(ValueError):
        explain_incident({})


def test_accepts_pandas_series_row():
    row = pd.Series(_incident())
    result = explain_incident(row)
    assert result["cluster_id"] == 15
    assert result["evidence"]["occurrence_count"] == 76


# --- D. Ranked incident explanations ------------------------------------------
def _ranking_df():
    return pd.DataFrame(
        [
            _incident(**{CLUSTER_COLUMN: 15, RANK_FIELD: 1, COUNT_FIELD: 76}),
            _incident(
                **{
                    CLUSTER_COLUMN: 9,
                    RANK_FIELD: 2,
                    COUNT_FIELD: 75,
                    MESSAGE_FIELD: "Failed to establish connection to primary database <HOST>",
                }
            ),
            _incident(
                **{
                    CLUSTER_COLUMN: 4,
                    RANK_FIELD: 3,
                    COUNT_FIELD: 80,
                    MESSAGE_FIELD: "Upstream request timed out after <NUM>s",
                }
            ),
        ]
    )


def test_multiple_records_one_explanation_each():
    df = _ranking_df()
    results = explain_ranked_incidents(df)
    assert isinstance(results, list)
    assert len(results) == len(df) == 3
    assert all(set(r.keys()) == RESULT_KEYS for r in results)


def test_empty_ranking_returns_empty_list():
    empty = pd.DataFrame(
        columns=[CLUSTER_COLUMN, RANK_FIELD, COUNT_FIELD, MESSAGE_FIELD]
    )
    assert explain_ranked_incidents(empty) == []


def test_input_order_is_preserved():
    df = _ranking_df()
    results = explain_ranked_incidents(df)
    assert [r["cluster_id"] for r in results] == df[CLUSTER_COLUMN].tolist()
    assert [r["rank"] for r in results] == df[RANK_FIELD].tolist()


def test_one_to_one_correspondence():
    df = _ranking_df()
    results = explain_ranked_incidents(df)
    for (_, row), result in zip(df.iterrows(), results):
        assert result["cluster_id"] == row[CLUSTER_COLUMN]


def test_missing_required_column_raises():
    df = pd.DataFrame([{RANK_FIELD: 1, COUNT_FIELD: 5}])  # no cluster_id column
    with pytest.raises(ValueError, match="cluster_id"):
        explain_ranked_incidents(df)


def test_malformed_row_is_not_silently_dropped():
    df = _ranking_df()
    df.loc[1, COUNT_FIELD] = -3  # invalid count in the middle row
    with pytest.raises(ValueError):
        explain_ranked_incidents(df)


def test_non_dataframe_raises():
    with pytest.raises(TypeError):
        explain_ranked_incidents([_incident()])


def test_input_dataframe_is_not_mutated():
    df = _ranking_df()
    snapshot = df.copy(deep=True)
    explain_ranked_incidents(df)
    assert df.equals(snapshot)
    assert list(df.columns) == list(snapshot.columns)
    assert df.index.equals(snapshot.index)


# --- E. Determinism and reliability -------------------------------------------
def test_repeated_calls_are_equal():
    incident = _incident()
    assert explain_incident(incident) == explain_incident(incident)


def test_suggested_checks_are_deterministic_across_calls():
    first = explain_incident(_incident())["suggested_checks"]
    second = explain_incident(_incident())["suggested_checks"]
    assert first == second


def test_cluster_id_does_not_affect_severity_or_checks():
    # Same message, wildly different ids -> identical checks and no severity claim.
    a = explain_incident(_incident(**{CLUSTER_COLUMN: 1}))
    b = explain_incident(_incident(**{CLUSTER_COLUMN: 100000}))
    assert a["suggested_checks"] == b["suggested_checks"]
    assert a["summary"].replace("Cluster 1", "Cluster X") == b["summary"].replace(
        "Cluster 100000", "Cluster X"
    )


def test_no_definitive_root_cause_claim():
    result = explain_incident(_incident())
    joined = " ".join(result["limitations"] + [result["summary"]]).lower()
    assert "root cause is not confirmed" in joined
    assert "confirmed root cause" not in joined


def test_missing_severity_not_fabricated_in_evidence():
    incident = _incident()
    del incident["severity_score"]
    result = explain_incident(incident)
    assert "severity_score" not in result["evidence"]
    assert not any("severity_score" in str(v) for v in result["evidence"].values())


def test_scores_are_labeled_heuristic_not_impact():
    result = explain_incident(_incident())
    joined = " ".join(result["limitations"]).lower()
    assert "heuristic" in joined
    assert "not confirmed real-world" in joined


# --- Smoke test against the real pipeline ranking ------------------------------
@needs_generated_data
def test_real_ranking_explanations_preserve_order_and_ids():
    result = run_pipeline(config.RAW_LOG_PATH)
    ranking = result.ranking

    explanations = explain_ranked_incidents(ranking)
    assert len(explanations) == len(ranking)
    assert [e["cluster_id"] for e in explanations] == ranking[CLUSTER_COLUMN].tolist()
    assert [e["rank"] for e in explanations] == ranking[RANK_FIELD].tolist()
    for e in explanations:
        assert set(e.keys()) == RESULT_KEYS
        assert e["suggested_checks"]            # never empty
        assert e["limitations"]                 # never empty
        # Every real record carries a count, so it must be reported.
        assert e["evidence"]["occurrence_count"] >= 1
