"""Tests for the Streamlit support helpers (src/ui_support.py).

These cover the pieces of logic the UI depends on but must not implement inline:
turning a user's log input into a path the existing workflow accepts, turning the
real ranking output into chart tables, and resolving the cluster detail view
against an analysis result that is already in memory.

No Streamlit, no network, no project data is touched: temporary files are created
under pytest's tmp_path or the system temp dir and removed again.
"""

import os

import pandas as pd
import pytest

from src import ui_support
from src.incident_explainer import explain_ranked_incidents
from src.pipeline import run_pipeline
from src.ranker import OUTPUT_COLUMNS
from src.ui_support import (
    DEFAULT_CHART_TOP_N,
    LogInputError,
    cluster_log_entries,
    cluster_options,
    decode_log_bytes,
    find_explanation,
    incident_count_data,
    remove_temp_log,
    resolve_log_path,
    severity_distribution_data,
    severity_label,
    validate_pasted_log,
    validate_uploaded_log,
    write_temp_log,
)

GOOD_LOG = """\
2026-10-01T00:00:00Z ERROR [req-1] [DatabaseService] Database connection failed after 3s
2026-10-01T00:00:01Z ERROR [req-2] [DatabaseService] Database connection failed after 7s
2026-10-01T00:00:02Z FATAL [req-3] [PaymentService] Payment declined by provider PayPal for order 12
2026-10-01T00:00:03Z WARNING [req-4] [ApiGateway] Upstream request timed out after 5s
2026-10-01T00:00:04Z INFO [req-5] [ApiGateway] Request completed in 12ms
"""


def _ranking(rows):
    """A ranking DataFrame with the ranker's documented columns."""
    frame = pd.DataFrame(rows)
    return frame[[c for c in OUTPUT_COLUMNS if c in frame.columns]]


def _two_clusters():
    return _ranking(
        [
            {
                "cluster_id": 15,
                "size": 76,
                "representative_message": "Database connection failed",
                "severity_score": 1.0,
                "frequency_score": 1.0,
                "risk_score": 1.0,
                "rank": 1,
            },
            {
                "cluster_id": 3,
                "size": 12,
                "representative_message": "Payment declined",
                "severity_score": 0.6,
                "frequency_score": 0.2,
                "risk_score": 0.3,
                "rank": 2,
            },
        ]
    )


# --- A. Uploaded files --------------------------------------------------------
def test_a1_valid_utf8_upload_decodes_without_warning():
    text, warning = decode_log_bytes(GOOD_LOG.encode("utf-8"))
    assert text == GOOD_LOG
    assert warning is None


def test_a2_non_utf8_upload_is_replaced_and_reported():
    raw = "ERROR caf\u00e9 failed".encode("utf-8")[:8] + b"\xff\xfe" + b" tail"
    text, warning = decode_log_bytes(raw)
    assert warning is not None
    assert "not valid UTF-8" in warning
    assert "\ufffd" in text                      # honest about the replacement
    assert "tail" in text                         # the readable part survives


def test_a3_empty_upload_rejected():
    with pytest.raises(LogInputError, match="empty"):
        decode_log_bytes(b"")
    with pytest.raises(LogInputError, match="empty"):
        decode_log_bytes(b"   \n\t ")


def test_a4_oversized_upload_rejected(monkeypatch):
    monkeypatch.setattr(ui_support, "MAX_INPUT_BYTES", 10)
    with pytest.raises(LogInputError, match="larger than the"):
        decode_log_bytes(b"x" * 11)


def test_a5_non_bytes_upload_rejected():
    for bad in (None, "text", 123, ["a"]):
        with pytest.raises(LogInputError):
            decode_log_bytes(bad)


@pytest.mark.parametrize("name", ["app.log", "app.txt", "nested/APP.LOG", "a.LOG", "a.TxT"])
def test_a6_supported_extensions_accepted(name):
    text, warning = validate_uploaded_log(name, GOOD_LOG.encode("utf-8"))
    assert text == GOOD_LOG
    assert warning is None


@pytest.mark.parametrize("name", ["app.csv", "app.json", "app.exe", "app", "", None])
def test_a7_unsupported_extensions_rejected(name):
    with pytest.raises(LogInputError, match="Unsupported file type"):
        validate_uploaded_log(name, GOOD_LOG.encode("utf-8"))


def test_a8_error_names_the_supported_types():
    with pytest.raises(LogInputError) as excinfo:
        validate_uploaded_log("logs.csv", b"x")
    assert ".log" in str(excinfo.value)
    assert ".txt" in str(excinfo.value)


def test_a9_upload_validation_is_a_value_error():
    # app.py catches ValueError, so a bad upload must be a ValueError subclass.
    assert issubclass(LogInputError, ValueError)


# --- B. Pasted text and paths -------------------------------------------------
def test_b1_pasted_text_is_trimmed():
    text, warning = validate_pasted_log("  \n" + GOOD_LOG + "\n  ")
    assert text == GOOD_LOG.strip()
    assert warning is None


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None, 42, ["line"]])
def test_b2_empty_or_invalid_paste_rejected(bad):
    with pytest.raises(LogInputError):
        validate_pasted_log(bad)


def test_b3_oversized_paste_rejected(monkeypatch):
    monkeypatch.setattr(ui_support, "MAX_INPUT_BYTES", 10)
    with pytest.raises(LogInputError, match="larger than the"):
        validate_pasted_log("x" * 11)


def test_b4_path_is_trimmed():
    assert resolve_log_path("  data/raw/app.log  ") == "data/raw/app.log"


@pytest.mark.parametrize("bad", ["", "   ", None, 42])
def test_b5_blank_path_rejected(bad):
    with pytest.raises(LogInputError, match="Enter a log file path"):
        resolve_log_path(bad)


def test_b6_nonexistent_path_is_left_to_the_pipeline():
    # Existence is the pipeline's job: it reports a real analysis failure there.
    assert resolve_log_path("/does/not/exist.log") == "/does/not/exist.log"


# --- C. Temporary files -------------------------------------------------------
def test_c1_temp_file_round_trips_and_is_deleted():
    path = write_temp_log(GOOD_LOG)
    try:
        assert os.path.isfile(path)
        assert path.endswith(".log")
        assert os.path.basename(path).startswith("incidentinsight-")
        with open(path, encoding="utf-8") as handle:
            assert handle.read() == GOOD_LOG
    finally:
        remove_temp_log(path)
    assert not os.path.exists(path)


def test_c2_temp_file_is_outside_the_project_tree():
    path = write_temp_log("line")
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(ui_support.__file__)))
        assert not os.path.abspath(path).startswith(project_root)
    finally:
        remove_temp_log(path)


def test_c3_unicode_log_survives_the_temp_file():
    text = "2026-10-01T00:00:00Z ERROR [req-1] [Svc] utilisateur \u00e9chou\u00e9 \u2013 cl\u00e9"
    path = write_temp_log(text)
    try:
        with open(path, encoding="utf-8") as handle:
            assert handle.read() == text
    finally:
        remove_temp_log(path)


def test_c4_removing_twice_or_nothing_is_safe():
    remove_temp_log(None)
    remove_temp_log("")
    remove_temp_log(os.path.join(os.path.dirname(__file__), "never-existed.log"))


def test_c5_two_temp_files_never_collide():
    first = write_temp_log("a")
    second = write_temp_log("a")
    try:
        assert first != second
    finally:
        remove_temp_log(first)
        remove_temp_log(second)


# --- D. Severity mapping ------------------------------------------------------
@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, "DEBUG"),
        (0.2, "INFO"),
        (0.4, "WARNING"),
        (0.6, "ERROR"),
        (0.8, "CRITICAL"),
        (1.0, "FATAL"),
    ],
)
def test_d1_severity_scores_map_back_to_their_level(score, expected):
    assert severity_label(score) == expected


@pytest.mark.parametrize("bad", [None, "", "abc", float("nan"), -1.0, 5.0, 1.5, object()])
def test_d2_unrecognizable_scores_are_unknown_never_guessed(bad):
    assert severity_label(bad) == "UNKNOWN"


def test_d3_ladder_order_is_worst_last_and_complete():
    assert ui_support.SEVERITY_ORDER == ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "FATAL")


# --- E. Count chart data ------------------------------------------------------
def test_e1_counts_use_the_ranker_sizes_in_rank_order():
    data = incident_count_data(_two_clusters())
    assert list(data.index) == ["#1 (cluster 15)", "#2 (cluster 3)"]
    assert list(data["log_count"]) == [76, 12]
    assert data.index.name == "incident"


def test_e2_input_order_does_not_change_the_chart_order():
    frame = _two_clusters().iloc[::-1].reset_index(drop=True)
    assert list(incident_count_data(frame).index) == ["#1 (cluster 15)", "#2 (cluster 3)"]


def test_e3_top_n_limits_the_bars():
    rows = [
        {
            "cluster_id": i,
            "size": 100 - i,
            "representative_message": f"msg {i}",
            "severity_score": 0.6,
            "frequency_score": 0.5,
            "risk_score": 0.5,
            "rank": i,
        }
        for i in range(1, 16)
    ]
    data = incident_count_data(_ranking(rows), top_n=4)
    assert len(data) == 4
    assert list(data.index) == ["#1 (cluster 1)", "#2 (cluster 2)", "#3 (cluster 3)", "#4 (cluster 4)"]


def test_e4_default_top_n_matches_the_ui_constant():
    assert DEFAULT_CHART_TOP_N == 10


@pytest.mark.parametrize("bad", [0, -1, True, False, "3", None, 2.5])
def test_e5_invalid_top_n_raises(bad):
    with pytest.raises(ValueError):
        incident_count_data(_two_clusters(), top_n=bad)


def test_e6_empty_ranking_gives_an_empty_chart_table():
    for frame in (pd.DataFrame(), pd.DataFrame(columns=OUTPUT_COLUMNS), None):
        data = incident_count_data(frame)
        assert data.empty
        assert list(data.columns) == ["log_count"]


def test_e7_ranking_without_size_is_not_invented():
    frame = pd.DataFrame({"cluster_id": [1], "rank": [1]})
    assert incident_count_data(frame).empty


def test_e8_missing_rank_or_cluster_still_produces_labels():
    assert list(incident_count_data(pd.DataFrame({"size": [4, 5]})).index) == ["row 0", "row 1"]
    frame = pd.DataFrame({"cluster_id": [7], "size": [4]})
    assert list(incident_count_data(frame).index) == ["cluster 7"]


def test_e9_non_dataframe_input_raises():
    with pytest.raises(TypeError):
        incident_count_data([1, 2, 3])


def test_e10_non_numeric_sizes_fall_back_to_zero_not_crash():
    frame = pd.DataFrame({"cluster_id": [1], "size": ["not-a-number"], "rank": [1]})
    assert list(incident_count_data(frame)["log_count"]) == [0]


# --- F. Severity chart data ---------------------------------------------------
def test_f1_distribution_counts_clusters_per_level():
    frame = _ranking(
        [
            {"cluster_id": 1, "size": 5, "severity_score": 1.0, "rank": 1},
            {"cluster_id": 2, "size": 4, "severity_score": 1.0, "rank": 2},
            {"cluster_id": 3, "size": 3, "severity_score": 0.6, "rank": 3},
        ]
    )
    data = severity_distribution_data(frame)
    assert list(data.index) == list(ui_support.SEVERITY_ORDER)   # full ladder, fixed order
    assert data.loc["FATAL", "incident_clusters"] == 2
    assert data.loc["ERROR", "incident_clusters"] == 1
    assert data.loc["DEBUG", "incident_clusters"] == 0            # a real zero, not a gap
    assert int(data["incident_clusters"].sum()) == 3


def test_f2_unknown_scores_get_their_own_category():
    frame = _ranking([{"cluster_id": 1, "size": 5, "severity_score": None, "rank": 1}])
    data = severity_distribution_data(frame)
    assert data.loc["UNKNOWN", "incident_clusters"] == 1
    assert list(data.index)[-1] == "UNKNOWN"


def test_f3_empty_or_missing_severity_gives_an_empty_chart_table():
    assert severity_distribution_data(pd.DataFrame()).empty
    assert severity_distribution_data(None).empty
    assert severity_distribution_data(pd.DataFrame({"cluster_id": [1]})).empty
    data = severity_distribution_data(pd.DataFrame(columns=OUTPUT_COLUMNS))
    assert data.empty
    assert list(data.columns) == ["incident_clusters"]


# --- G. Charts are derived from the real pipeline -----------------------------
def test_g1_chart_tables_match_a_real_pipeline_run(tmp_path):
    log = tmp_path / "input.log"
    log.write_text(GOOD_LOG, encoding="utf-8")
    result = run_pipeline(log_path=str(log))
    ranking = result.ranking

    counts = incident_count_data(ranking)
    severities = severity_distribution_data(ranking)

    assert not counts.empty and not severities.empty
    # Every bar is a real cluster size from the ranker, in rank order.
    assert list(counts["log_count"]) == list(ranking.sort_values("rank")["size"].astype(int))[: len(counts)]
    assert sum(counts["log_count"]) <= int(ranking["size"].sum())
    # The severity bars account for exactly the clusters the ranker produced.
    assert int(severities["incident_clusters"].sum()) == len(ranking)
    # No fabricated levels: each non-zero bar matches the ranker's own scores.
    for level, count in severities["incident_clusters"].items():
        expected = int(
            (ranking["severity_score"].map(severity_label) == level).sum()
        )
        assert count == expected


def test_g2_end_to_end_input_helpers_feed_the_pipeline(tmp_path):
    text, warning = validate_pasted_log(GOOD_LOG)
    assert warning is None
    path = write_temp_log(text)
    try:
        result = run_pipeline(log_path=path)
        assert result.n_logs == 5
        assert not incident_count_data(result.ranking).empty
    finally:
        remove_temp_log(path)
    assert not os.path.exists(path)


# --- H. Cluster detail view -----------------------------------------------------
def _clustered():
    """A tiny stand-in for ``PipelineResult.clustered``.

    Same columns the pipeline produces (parser output + normalized message +
    cluster_id), and deliberately out of rank order so the filters can be checked
    against the original row order.
    """
    return pd.DataFrame(
        {
            "line_id": [0, 1, 2, 3],
            "timestamp": ["2026-10-01T00:00:00Z", "2026-10-01T00:00:01Z", "2026-10-01T00:00:02Z", "2026-10-01T00:00:03Z"],
            "level": ["ERROR", "ERROR", "INFO", "FATAL"],
            "service": ["DatabaseService", "DatabaseService", "ApiGateway", "PaymentService"],
            "request_id": ["req-0", "req-1", "req-2", "req-3"],
            "message": [
                "Database connection failed after 3s",
                "Database connection failed after 7s",
                "Request completed in 12ms",
                "Payment declined by provider PayPal for order 12",
            ],
            "message_normalized": [
                "database connection failed after <number>s",
                "database connection failed after <number>s",
                "request completed in <number>ms",
                "payment declined by provider <alnum> for order <number>",
            ],
            "parsed": [True, True, True, True],
            "cluster_id": [7, 7, 3, 9],
        }
    )


def _explanations():
    return [
        {"cluster_id": 7, "rank": 1, "summary": "sevens", "evidence": {"occurrence_count": 2}},
        {"cluster_id": 9, "rank": 2, "summary": "nines", "evidence": {"occurrence_count": 1}},
        {"cluster_id": 3, "rank": 3, "summary": "threes", "evidence": {"occurrence_count": 1}},
    ]


def test_h1_cluster_options_come_from_the_ranking_in_rank_order():
    ranking = _two_clusters()
    options = cluster_options(ranking.sort_values("rank", ascending=False))   # input order is irrelevant
    assert list(options) == [15, 3]                                          # best rank first
    assert all(isinstance(key, int) for key in options)
    assert options[15] == "Cluster 15 - rank 1 - 76 log entries - risk 1.00"
    assert options[3] == "Cluster 3 - rank 2 - 12 log entries - risk 0.30"


def test_h2_cluster_options_are_empty_when_there_is_nothing_to_select():
    assert cluster_options(None) == {}
    assert cluster_options(pd.DataFrame()) == {}
    assert cluster_options(pd.DataFrame(columns=OUTPUT_COLUMNS)) == {}
    assert cluster_options(pd.DataFrame({"size": [1]})) == {}                # no cluster_id column


def test_h3_cluster_options_skip_rows_without_a_usable_cluster_id():
    ranking = _ranking(
        [
            {"cluster_id": 4, "size": 2, "severity_score": 0.6, "frequency_score": 1.0, "risk_score": 0.8, "rank": 1},
            {"cluster_id": None, "size": 1, "severity_score": 0.2, "frequency_score": 0.5, "risk_score": 0.4, "rank": 2},
        ]
    )
    assert list(cluster_options(ranking)) == [4]


def test_h4_cluster_options_reject_a_non_dataframe():
    with pytest.raises(TypeError, match="ranking must be a pandas DataFrame"):
        cluster_options([{"cluster_id": 1}])


def test_h5_cluster_log_entries_return_only_that_clusters_rows_in_order():
    frame = _clustered()
    entries = cluster_log_entries(frame, 7)
    assert entries["line_id"].tolist() == [0, 1]
    assert entries["message"].tolist() == [
        "Database connection failed after 3s",
        "Database connection failed after 7s",
    ]
    assert 3 not in set(entries["line_id"])                                  # no other cluster leaks in


def test_h6_original_messages_are_preserved_exactly():
    frame = _clustered()
    entries = cluster_log_entries(frame, 7)
    # The original text is untouched (not the normalized form), byte for byte.
    assert entries["message"].tolist() == frame.loc[[0, 1], "message"].tolist()
    assert all(original != normalized for original, normalized in zip(entries["message"], entries["message_normalized"]))
    # The normalized column is carried alongside, never substituted.
    assert entries["message_normalized"].tolist() == frame.loc[[0, 1], "message_normalized"].tolist()


def test_h7_cluster_log_entries_show_only_metadata_that_exists():
    frame = _clustered()
    entries = cluster_log_entries(frame, 9)
    assert list(entries.columns) == list(ui_support.CLUSTER_DETAIL_COLUMNS)
    assert "cluster_id" not in entries.columns                               # implied by the selection
    assert "parsed" not in entries.columns                                   # not part of the detail view
    trimmed = cluster_log_entries(frame[["cluster_id", "message", "level"]], 9)
    assert list(trimmed.columns) == ["level", "message"]                     # only what exists, in display order
    assert cluster_log_entries(frame[["cluster_id"]], 9).empty               # nothing worth showing


def test_h8_cluster_log_entries_are_empty_for_any_unusable_selection():
    frame = _clustered()
    assert cluster_log_entries(frame, 999).empty                             # unknown id
    assert cluster_log_entries(frame, None).empty                            # no selection
    assert cluster_log_entries(None, 7).empty                                # no data
    assert cluster_log_entries(pd.DataFrame(), 7).empty
    assert cluster_log_entries(frame[["line_id", "message"]], 7).empty       # no cluster_id column
    assert cluster_log_entries(frame, "not-a-number").empty                  # not an id at all


def test_h9_cluster_log_entries_reject_a_non_dataframe():
    with pytest.raises(TypeError, match="clustered must be a pandas DataFrame"):
        cluster_log_entries("nope", 7)


def test_h10_cluster_log_entries_accept_ids_of_other_numeric_types():
    frame = _clustered()
    expected = cluster_log_entries(frame, 7)
    assert cluster_log_entries(frame, "7").equals(expected)
    assert cluster_log_entries(frame, 7.0).equals(expected)
    assert cluster_log_entries(frame.astype({"cluster_id": "float64"}), 7).equals(expected)


def test_h11_the_input_frame_is_never_modified():
    frame = _clustered()
    before = frame.copy(deep=True)
    cluster_log_entries(frame, 7)
    pd.testing.assert_frame_equal(frame, before)


def test_h12_find_explanation_returns_the_existing_explanation_object():
    explanations = _explanations()
    found = find_explanation(explanations, 9)
    assert found is explanations[1]                                          # the same dict, not a copy
    assert found["summary"] == "nines"


def test_h13_find_explanation_matches_whatever_numeric_shape_the_id_arrives_in():
    explanations = _explanations()
    for value in (7, "7", 7.0):
        assert find_explanation(explanations, value)["summary"] == "sevens"


def test_h14_find_explanation_is_none_when_there_is_nothing_to_reuse():
    assert find_explanation(_explanations(), 999) is None                    # unknown cluster
    assert find_explanation(_explanations(), None) is None                   # no selection
    assert find_explanation([], 7) is None                                   # no explanations at all
    assert find_explanation(None, 7) is None
    assert find_explanation("not-a-list", 7) is None


def test_h15_find_explanation_skips_malformed_entries():
    explanations = ["nope", 42, None, {"rank": 1}, {"cluster_id": 5, "summary": "five"}]
    assert find_explanation(explanations, 5)["summary"] == "five"
    assert find_explanation(explanations, 1) is None                         # {"rank": 1} has no cluster_id


def test_h16_detail_helpers_agree_with_a_real_pipeline_run(tmp_path):
    log = tmp_path / "input.log"
    log.write_text(GOOD_LOG, encoding="utf-8")
    result = run_pipeline(log_path=str(log))
    explanations = explain_ranked_incidents(result.ranking)

    options = cluster_options(result.ranking)
    assert set(options) == set(int(c) for c in result.ranking["cluster_id"])
    assert list(options) == list(result.ranking.sort_values("rank")["cluster_id"].astype(int))

    seen_lines = []
    for cluster_id in options:
        entries = cluster_log_entries(result.clustered, cluster_id)
        expected = result.clustered[result.clustered["cluster_id"] == cluster_id]
        assert len(entries) == len(expected) == int(
            result.ranking.loc[result.ranking["cluster_id"] == cluster_id, "size"].iloc[0]
        )
        assert entries["message"].tolist() == expected["message"].tolist()
        # The explanation the UI would show is the one the explainer produced.
        assert find_explanation(explanations, cluster_id)["cluster_id"] == cluster_id
        seen_lines.extend(entries["line_id"].tolist())

    # Every row of every selectable cluster is shown exactly once, and nothing else.
    selectable = result.clustered[result.clustered["cluster_id"].isin(options)]
    assert sorted(seen_lines) == sorted(selectable["line_id"].tolist())


# A log whose lines mix timestamp spellings and bracket layouts, including one
# unstructured line. The first two rows mirror the reported cache-service case.
MIXED_FORMAT_LOG = """\
2026-10-26 10:07:30 ERROR [cache-service] Redis connection refused to cache-01.internal:6379
2026-10-26 10:07:34 ERROR [cache-service] Redis connection refused to cache-01.internal:6379
this line is not a valid log entry at all
2026-10-26T10:07:40Z WARNING [req-abc123] [AuthService] Authentication failed for user alice
2026-10-26 10:07:45 ERROR [req-def456] [PaymentService] Payment declined by provider PayPal
2026-10-26 10:07:50 INFO Request completed in 12ms
"""


def test_h17_cluster_detail_keeps_structured_fields_for_space_timestamp_logs(tmp_path):
    log = tmp_path / "mixed.log"
    log.write_text(MIXED_FORMAT_LOG, encoding="utf-8")
    result = run_pipeline(log_path=str(log))

    clustered = result.clustered.set_index("line_id")
    # The parser now extracts what the line actually contains ...
    assert clustered.loc[0, "timestamp"] == "2026-10-26 10:07:30"
    assert clustered.loc[0, "level"] == "ERROR"
    assert clustered.loc[0, "service"] == "cache-service"
    assert pd.isna(clustered.loc[0, "request_id"])            # genuinely absent, not invented
    assert clustered.loc[0, "message"] == "Redis connection refused to cache-01.internal:6379"
    assert clustered.loc[1, "timestamp"] == "2026-10-26 10:07:34"
    # The documented format still parses its request id and service.
    assert clustered.loc[3, "request_id"] == "req-abc123"
    assert clustered.loc[3, "service"] == "AuthService"
    assert pd.isna(clustered.loc[5, "service"])
    assert pd.isna(clustered.loc[5, "request_id"])

    # ... and the cluster-detail table shows the same values, not None.
    cluster_id = int(clustered.loc[0, "cluster_id"])
    entries = cluster_log_entries(result.clustered, cluster_id).set_index("line_id")
    assert set(entries.index) == {0, 1}
    for line_id, stamp in ((0, "2026-10-26 10:07:30"), (1, "2026-10-26 10:07:34")):
        assert entries.loc[line_id, "timestamp"] == stamp
        assert entries.loc[line_id, "level"] == "ERROR"
        assert entries.loc[line_id, "service"] == "cache-service"
        assert pd.isna(entries.loc[line_id, "request_id"])
    assert entries.loc[0, "message"] == "Redis connection refused to cache-01.internal:6379"
    assert entries.loc[1, "message"] == "Redis connection refused to cache-01.internal:6379"


def test_h18_cluster_detail_preserves_line_ids_and_raw_unparsed_messages(tmp_path):
    log = tmp_path / "mixed.log"
    log.write_text(MIXED_FORMAT_LOG, encoding="utf-8")
    result = run_pipeline(log_path=str(log))

    clustered = result.clustered.set_index("line_id")
    assert clustered.index.tolist() == list(range(6))         # one row per non-blank source line
    malformed_cluster = int(clustered.loc[2, "cluster_id"])
    detail = cluster_log_entries(result.clustered, malformed_cluster)

    row = detail[detail["line_id"] == 2].iloc[0]
    assert row["message"] == "this line is not a valid log entry at all"
    assert pd.isna(row["timestamp"])
    assert pd.isna(row["level"])
    assert pd.isna(row["service"])
    assert pd.isna(row["request_id"])
    assert detail["line_id"].tolist() == sorted(detail["line_id"].tolist())


def test_h19_cluster_label_pluralizes_a_single_log_entry():
    # A cluster holding one row must read "1 log entry", never "1 log entries".
    options = cluster_options(
        _ranking(
            [
                {"cluster_id": 4, "size": 1, "risk_score": 0.2, "rank": 1},
                {"cluster_id": 5, "size": 2, "risk_score": 0.1, "rank": 2},
            ]
        )
    )
    assert options[4] == "Cluster 4 - rank 1 - 1 log entry - risk 0.20"
    assert options[5] == "Cluster 5 - rank 2 - 2 log entries - risk 0.10"
