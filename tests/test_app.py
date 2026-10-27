"""Day 10 / Day 11 / Day 16 UI tests for app.py.

Covers the log input modes, the dashboard charts, and the cluster detail view.

ALL tests are offline: the pipeline runs on a tiny synthetic log, and no OpenAI
key is configured or requested. The provider boundary is never reached.

Streamlit is a declared project dependency, but it is not installed in every
interpreter, so each test is skipped individually where it is missing rather than
failing the collection of the whole module.
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from src.ui_support import cluster_log_entries, cluster_options, find_explanation

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

try:
    from streamlit.testing.v1 import AppTest

    _STREAMLIT_AVAILABLE = True
except ImportError:
    AppTest = None
    _STREAMLIT_AVAILABLE = False

needs_streamlit = pytest.mark.skipif(
    not _STREAMLIT_AVAILABLE,
    reason="streamlit is not installed in this interpreter",
)

GOOD_LOG = """\
2026-10-01T00:00:00Z ERROR [req-1] [DatabaseService] Database connection failed after 3s
2026-10-01T00:00:01Z ERROR [req-2] [DatabaseService] Database connection failed after 7s
2026-10-01T00:00:02Z FATAL [req-3] [PaymentService] Payment declined by provider PayPal for order 12
2026-10-01T00:00:03Z WARNING [req-4] [ApiGateway] Upstream request timed out after 5s
2026-10-01T00:00:04Z INFO [req-5] [ApiGateway] Request completed in 12ms
"""

UPLOAD_MODE = "Upload a log file"
PASTE_MODE = "Paste log text"
PATH_MODE = "Use a file path"

# A second, completely unrelated log so a stale-detail regression would be
# obvious: none of its messages appear in GOOD_LOG.
OTHER_LOG = """\
2026-11-01T00:00:00Z ERROR [req-9] [SearchService] Search index shard unavailable
2026-11-01T00:00:01Z ERROR [req-10] [SearchService] Search index shard unavailable
2026-11-01T00:00:02Z CRITICAL [req-11] [CacheService] Cache eviction storm detected
"""


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)


@pytest.fixture
def app_module():
    """Import app.py only in tests that actually run (Streamlit must exist)."""
    import app

    return app


def _upload(name, text=GOOD_LOG):
    """A stand-in for Streamlit's UploadedFile: only .name and .getvalue() are used."""
    data = text.encode("utf-8") if isinstance(text, str) else text
    return SimpleNamespace(name=name, getvalue=lambda: data)


def _new_app():
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def _leftover_temp_logs():
    return [
        name
        for name in os.listdir(tempfile.gettempdir())
        if name.startswith("incidentinsight-") and name.endswith(".log")
    ]


def _charts(at):
    """The rendered bar charts (Streamlit exposes them as vega-lite elements)."""
    return at.get("vega_lite_chart")


# --- A. Input resolution (app.py logic, no rendering) -------------------------
@needs_streamlit
def test_a1_pasted_text_becomes_a_temporary_log_file(app_module):
    path, warning, owns_temp = app_module._prepare_log_input(PASTE_MODE, None, GOOD_LOG, "")
    try:
        assert owns_temp is True
        assert warning is None
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as handle:
            assert handle.read() == GOOD_LOG.strip()
    finally:
        app_module.remove_temp_log(path)


@needs_streamlit
def test_a2_uploaded_file_becomes_a_temporary_log_file(app_module):
    path, warning, owns_temp = app_module._prepare_log_input(UPLOAD_MODE, _upload("app.log"), "", "")
    try:
        assert owns_temp is True
        assert warning is None
        with open(path, encoding="utf-8") as handle:
            assert handle.read() == GOOD_LOG
    finally:
        app_module.remove_temp_log(path)


@needs_streamlit
def test_a3_path_mode_creates_no_temporary_file(app_module):
    path, warning, owns_temp = app_module._prepare_log_input(PATH_MODE, None, "", "  data/raw/app.log ")
    assert (path, warning, owns_temp) == ("data/raw/app.log", None, False)


@needs_streamlit
def test_a4_upload_without_a_file_is_reported(app_module):
    with pytest.raises(ValueError, match="Choose a log file to upload first"):
        app_module._prepare_log_input(UPLOAD_MODE, None, "", "")


@needs_streamlit
def test_a5_unsupported_upload_type_is_reported(app_module):
    with pytest.raises(ValueError, match="Unsupported file type"):
        app_module._prepare_log_input(UPLOAD_MODE, _upload("logs.csv"), "", "")


@needs_streamlit
def test_a6_empty_paste_is_reported(app_module):
    with pytest.raises(ValueError, match="Nothing was pasted"):
        app_module._prepare_log_input(PASTE_MODE, None, "   \n ", "")


@needs_streamlit
def test_a7_blank_path_is_reported(app_module):
    with pytest.raises(ValueError, match="Enter a log file path"):
        app_module._prepare_log_input(PATH_MODE, None, "", "   ")


@needs_streamlit
def test_a8_non_utf8_upload_still_runs_but_warns(app_module):
    raw = b"2026-10-01T00:00:00Z ERROR [req-1] [Svc] failed \xff\xfe badly"
    path, warning, owns_temp = app_module._prepare_log_input(UPLOAD_MODE, _upload("app.log", raw), "", "")
    try:
        assert owns_temp is True
        assert warning is not None and "not valid UTF-8" in warning
        assert os.path.isfile(path)
    finally:
        app_module.remove_temp_log(path)


# --- B. The three input modes are reachable in the UI -------------------------
@needs_streamlit
def test_b1_all_three_input_modes_are_offered():
    at = _new_app()
    assert not at.exception
    assert len(at.sidebar.radio) == 1
    assert set(at.sidebar.radio[0].options) == {UPLOAD_MODE, PASTE_MODE, PATH_MODE}


@needs_streamlit
def test_b2_path_mode_is_the_default_and_still_works():
    at = _new_app()
    assert at.sidebar.radio[0].value == PATH_MODE
    assert any(t.label == "Log file path" for t in at.sidebar.text_input)


@needs_streamlit
def test_b3_paste_mode_shows_a_text_area_instead_of_a_path():
    at = _new_app()
    at.sidebar.radio[0].set_value(PASTE_MODE).run()
    assert not at.exception
    assert len(at.sidebar.text_area) == 1
    assert at.sidebar.text_input == []
    assert at.sidebar.file_uploader == []


@needs_streamlit
def test_b4_upload_mode_shows_a_file_uploader():
    at = _new_app()
    at.sidebar.radio[0].set_value(UPLOAD_MODE).run()
    assert not at.exception
    assert len(at.sidebar.file_uploader) == 1
    assert at.sidebar.text_input == []


@needs_streamlit
def test_b5_run_analysis_button_is_always_present():
    for mode in (PATH_MODE, PASTE_MODE, UPLOAD_MODE):
        at = _new_app()
        if mode != PATH_MODE:
            at.sidebar.radio[0].set_value(mode).run()
        assert any(b.label == "Run analysis" for b in at.sidebar.button)


# --- C. Pasted logs are analysed end to end ----------------------------------
@needs_streamlit
def test_c1_pasted_log_runs_the_existing_workflow():
    at = _new_app()
    at.sidebar.radio[0].set_value(PASTE_MODE).run()
    at.sidebar.text_area[0].set_value(GOOD_LOG).run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert any(s.value == "Analysis summary" for s in at.subheader)
    labels = {m.label: m.value for m in at.metric}
    assert labels["Log entries"] == "5"
    assert int(labels["Incident clusters"]) >= 1
    assert int(labels["Explanations"]) >= 1


@needs_streamlit
def test_c2_pasted_log_temp_file_is_deleted_after_the_analysis():
    before = set(_leftover_temp_logs())
    at = _new_app()
    at.sidebar.radio[0].set_value(PASTE_MODE).run()
    at.sidebar.text_area[0].set_value(GOOD_LOG).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert set(_leftover_temp_logs()) == before


@needs_streamlit
def test_c3_empty_paste_shows_an_error_and_no_analysis():
    at = _new_app()
    at.sidebar.radio[0].set_value(PASTE_MODE).run()
    at.sidebar.text_area[0].set_value("   ").run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Nothing was pasted" in e.value for e in at.sidebar.error)
    assert any("No analysis yet" in i.value for i in at.info)


@needs_streamlit
def test_c4_the_workflow_receives_the_pasted_text(tmp_path):
    import src.workflow as wf

    seen = {}
    real = wf.run_workflow

    def capture(*args, **kwargs):
        seen["log_path"] = kwargs.get("log_path") or (args[0] if args else None)
        with open(seen["log_path"], encoding="utf-8") as handle:
            seen["text"] = handle.read()
        return real(*args, **kwargs)

    with mock.patch.object(wf, "run_workflow", side_effect=capture):
        at = _new_app()
        at.sidebar.radio[0].set_value(PASTE_MODE).run()
        at.sidebar.text_area[0].set_value(GOOD_LOG).run()
        at.sidebar.button[0].click().run()

    assert not at.exception
    assert seen["text"] == GOOD_LOG.strip()
    # The temporary file handed to the workflow is gone once the result is held.
    assert not os.path.exists(seen["log_path"])


@needs_streamlit
def test_c5_switching_input_mode_does_not_rerun_the_analysis(tmp_path):
    import src.workflow as wf

    real = wf.run_workflow
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    with mock.patch.object(wf, "run_workflow", side_effect=counting):
        at = _new_app()
        at.sidebar.radio[0].set_value(PASTE_MODE).run()
        assert calls["n"] == 0
        at.sidebar.text_area[0].set_value(GOOD_LOG).run()
        assert calls["n"] == 0                       # typing does not analyse
        at.sidebar.button[0].click().run()
        assert calls["n"] == 1
        at.sidebar.radio[0].set_value(UPLOAD_MODE).run()
        assert calls["n"] == 1                       # switching mode does not re-analyse
        assert any(s.value == "Analysis summary" for s in at.subheader)


# --- D. Charts (Day 11) -------------------------------------------------------
@needs_streamlit
def test_d1_no_charts_before_an_analysis_exists():
    at = _new_app()
    assert not at.exception
    assert _charts(at) == []
    assert not any(s.value == "Incident charts" for s in at.subheader)


@needs_streamlit
def test_d2_both_charts_render_from_the_real_ranking(tmp_path):
    log = tmp_path / "input.log"
    log.write_text(GOOD_LOG, encoding="utf-8")
    at = _new_app()
    at.sidebar.text_input[0].set_value(str(log)).run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert any(s.value == "Incident charts" for s in at.subheader)
    assert len(_charts(at)) == 2
    headings = [m.value for m in at.markdown]
    assert any("Log entries per incident" in h for h in headings)
    assert any("maximum severity level" in h for h in headings)


@needs_streamlit
def test_d3_charts_survive_a_chat_turn_without_recomputing(tmp_path):
    log = tmp_path / "input.log"
    log.write_text(GOOD_LOG, encoding="utf-8")
    at = _new_app()
    at.sidebar.text_input[0].set_value(str(log)).run()
    at.sidebar.button[0].click().run()
    at.chat_input[0].set_value("What is the highest-ranked incident?").run()
    assert not at.exception
    assert len(_charts(at)) == 2


@needs_streamlit
def test_d4_ranked_table_and_explanations_are_kept(tmp_path):
    log = tmp_path / "input.log"
    log.write_text(GOOD_LOG, encoding="utf-8")
    at = _new_app()
    at.sidebar.text_input[0].set_value(str(log)).run()
    at.sidebar.button[0].click().run()
    headings = [s.value for s in at.subheader]
    assert any(h.startswith("Top ") and "incident clusters" in h for h in headings)
    assert "Incident charts" in headings
    assert "Incident explanations" in headings
    # The ranked table is still the FIRST dataframe rendered (the cluster detail
    # view added its own tables below it).
    ranked = at.dataframe[0].value
    assert list(ranked.columns) == ["rank", "cluster_id", "size", "frequency_score", "severity_score", "risk_score"]
    assert len(ranked) >= 1


@needs_streamlit
def test_d5_a_failed_analysis_shows_no_charts(tmp_path):
    at = _new_app()
    at.sidebar.text_input[0].set_value(str(tmp_path / "missing.log")).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Analysis failed" in e.value for e in at.error)
    assert _charts(at) == []


# --- E. Cluster detail view (Day 16) ------------------------------------------
_DETAIL_HEADING = "Cluster detail"
_EXPLANATION_HEADING = "**Explanation for this cluster**"


def _write_log(tmp_path, text=GOOD_LOG, name="input.log"):
    log = tmp_path / name
    log.write_text(text, encoding="utf-8")
    return log


def _analysed_app(tmp_path, text=GOOD_LOG, name="input.log"):
    """An app that has already analysed `text` through the real (offline) workflow."""
    at = _new_app()
    at.sidebar.text_input[0].set_value(str(_write_log(tmp_path, text, name))).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.selectbox, "the cluster selector should exist once an analysis succeeded"
    return at


def _counting_workflow():
    """Patch src.workflow.run_workflow with a call-counting passthrough."""
    import src.workflow as wf

    real = wf.run_workflow
    calls = {"n": 0}

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    return mock.patch.object(wf, "run_workflow", side_effect=counting), calls


def _entries_of(result, cluster_id):
    """The rows the pipeline itself assigned to `cluster_id`, in original order."""
    clustered = result.analysis.clustered
    return clustered[clustered["cluster_id"] == cluster_id]


def _explanation_shown(at):
    """The markdown rendered directly under the detail view's explanation heading."""
    texts = [m.value for m in at.markdown]
    return texts[texts.index(_EXPLANATION_HEADING) + 1]


@needs_streamlit
def test_e1_no_cluster_selector_before_an_analysis_exists():
    at = _new_app()
    assert not at.exception
    assert at.selectbox == []
    assert not any(s.value == _DETAIL_HEADING for s in at.subheader)


@needs_streamlit
def test_e2_selector_lists_the_current_ranking_in_rank_order(tmp_path):
    at = _analysed_app(tmp_path)
    result = at.session_state["analysis_result"]
    options = cluster_options(result.analysis.ranking)
    assert len(options) >= 2, "GOOD_LOG should produce more than one cluster"
    assert len(at.selectbox) == 1
    assert at.selectbox[0].label == "Incident cluster"
    # Every ranked cluster is selectable, labelled from the ranker's own numbers,
    # best-ranked first - and the best-ranked one is preselected.
    assert at.selectbox[0].options == [options[cluster_id] for cluster_id in options]
    assert at.selectbox[0].value == next(iter(options))


@needs_streamlit
def test_e3_selecting_a_cluster_shows_only_its_own_entries(tmp_path):
    at = _analysed_app(tmp_path)
    result = at.session_state["analysis_result"]
    selected = at.selectbox[0].value
    shown = at.dataframe[1].value
    expected = _entries_of(result, selected)

    assert len(shown) == len(expected) > 0
    assert shown["line_id"].tolist() == expected["line_id"].tolist()
    # Nothing from any other cluster is displayed.
    other_lines = set(result.analysis.clustered["line_id"]) - set(expected["line_id"])
    assert set(shown["line_id"]).isdisjoint(other_lines)


@needs_streamlit
def test_e4_original_messages_are_preserved_exactly(tmp_path):
    at = _analysed_app(tmp_path)
    result = at.session_state["analysis_result"]
    selected = at.selectbox[0].value
    expected = _entries_of(result, selected)
    shown = at.dataframe[1].value

    assert shown["message"].tolist() == expected["message"].tolist()
    # The normalized text is supplementary: it never replaces the original,
    # and the full view carries both side by side.
    assert "message_normalized" not in shown.columns
    full = at.dataframe[2].value
    assert list(full.columns)[-1] == "message_normalized"
    assert full["message"].tolist() == expected["message"].tolist()
    assert full["message_normalized"].tolist() == expected["message_normalized"].tolist()
    assert any("All " in e.label for e in at.get("expander"))


@needs_streamlit
def test_e5_real_metadata_is_displayed_and_nothing_is_invented(tmp_path):
    at = _analysed_app(tmp_path)
    result = at.session_state["analysis_result"]
    expected = _entries_of(result, at.selectbox[0].value)
    shown = at.dataframe[1].value

    for column in ("line_id", "timestamp", "level", "service", "request_id", "message"):
        assert column in shown.columns
    assert shown["level"].tolist() == expected["level"].tolist()
    assert shown["service"].tolist() == expected["service"].tolist()
    assert shown["timestamp"].tolist() == expected["timestamp"].tolist()
    # The preview is capped, and the count shown is the real row count.
    assert len(shown) <= 25
    assert any(f"**{len(expected)} log " in m.value for m in at.markdown)


@needs_streamlit
def test_e6_the_selected_clusters_own_explanation_is_displayed(tmp_path):
    at = _analysed_app(tmp_path)
    result = at.session_state["analysis_result"]
    selected = at.selectbox[0].value
    expected = find_explanation(result.explanations, selected)
    assert expected is not None
    assert _explanation_shown(at) == expected["summary"]


@needs_streamlit
def test_e7_switching_cluster_never_reruns_the_pipeline(tmp_path):
    patch, calls = _counting_workflow()
    log = _write_log(tmp_path)
    with patch:
        at = _new_app()
        at.sidebar.text_input[0].set_value(str(log)).run()
        at.sidebar.button[0].click().run()
        assert calls["n"] == 1

        result = at.session_state["analysis_result"]
        cluster_ids = list(cluster_options(result.analysis.ranking))
        assert len(cluster_ids) >= 2
        first_lines = at.dataframe[1].value["line_id"].tolist()

        at.selectbox[0].set_value(cluster_ids[1]).run()

        assert calls["n"] == 1, "selecting a cluster must not re-run the analysis"
        assert at.selectbox[0].value == cluster_ids[1]
        second = at.dataframe[1].value
        expected = _entries_of(result, cluster_ids[1])
        assert second["line_id"].tolist() == expected["line_id"].tolist()
        assert second["message"].tolist() == expected["message"].tolist()
        assert set(first_lines).isdisjoint(set(second["line_id"]))
        # The explanation followed the selection, and was looked up, not rebuilt.
        assert _explanation_shown(at) == find_explanation(result.explanations, cluster_ids[1])["summary"]


@needs_streamlit
def test_e8_a_new_analysis_cannot_reuse_the_previous_clusters_details(tmp_path):
    patch, calls = _counting_workflow()
    first_log = _write_log(tmp_path, GOOD_LOG, "first.log")
    second_log = _write_log(tmp_path, OTHER_LOG, "second.log")
    with patch:
        at = _new_app()
        at.sidebar.text_input[0].set_value(str(first_log)).run()
        at.sidebar.button[0].click().run()
        old_result = at.session_state["analysis_result"]
        old_ids = list(cluster_options(old_result.analysis.ranking))
        # Deliberately pick a cluster that is NOT the default selection.
        at.selectbox[0].set_value(old_ids[-1]).run()
        assert at.selectbox[0].value == old_ids[-1]
        stale_messages = set(at.dataframe[1].value["message"])
        assert calls["n"] == 1

        at.sidebar.text_input[0].set_value(str(second_log)).run()
        at.sidebar.button[0].click().run()

        assert calls["n"] == 2
        assert not at.exception
        new_result = at.session_state["analysis_result"]
        new_ids = list(cluster_options(new_result.analysis.ranking))
        # The stale selection was reconciled against the new result.
        assert at.selectbox[0].value in new_ids
        assert at.session_state["selected_cluster"] == at.selectbox[0].value
        shown = at.dataframe[1].value
        expected = _entries_of(new_result, at.selectbox[0].value)
        assert shown["line_id"].tolist() == expected["line_id"].tolist()
        assert set(shown["message"]).isdisjoint(stale_messages)
        assert _explanation_shown(at) == find_explanation(
            new_result.explanations, at.selectbox[0].value
        )["summary"]


@needs_streamlit
def test_e9_a_stored_selection_that_no_longer_exists_falls_back(tmp_path):
    at = _analysed_app(tmp_path)
    result = at.session_state["analysis_result"]
    best = next(iter(cluster_options(result.analysis.ranking)))
    at.session_state["selected_cluster"] = 999999   # a cluster id from no run at all
    at.run()
    assert not at.exception
    assert at.selectbox[0].value == best
    shown = at.dataframe[1].value
    assert shown["line_id"].tolist() == _entries_of(result, best)["line_id"].tolist()


@needs_streamlit
def test_e10_a_cluster_with_no_stored_entries_is_reported_not_crashed(tmp_path):
    import src.ui_support as ui

    at = _analysed_app(tmp_path)
    with mock.patch.object(ui, "cluster_log_entries", return_value=pd.DataFrame()):
        at.run()
    assert not at.exception
    assert any("No log entries are stored for this cluster" in c.value for c in at.caption)


@needs_streamlit
def test_e11_a_missing_explanation_is_reported_not_crashed(tmp_path):
    import src.ui_support as ui

    at = _analysed_app(tmp_path)
    with mock.patch.object(ui, "find_explanation", return_value=None):
        at.run()
    assert not at.exception
    assert any("No explanation is available for this cluster" in c.value for c in at.caption)


@needs_streamlit
def test_e12_a_failed_analysis_shows_no_cluster_detail(tmp_path):
    at = _new_app()
    at.sidebar.text_input[0].set_value(str(tmp_path / "missing.log")).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Analysis failed" in e.value for e in at.error)
    assert at.selectbox == []
    assert not any(s.value == _DETAIL_HEADING for s in at.subheader)


@needs_streamlit
def test_e13_the_existing_dashboard_sections_are_untouched(tmp_path):
    at = _analysed_app(tmp_path)
    headings = [s.value for s in at.subheader]
    for expected in (
        "Analysis summary",
        "Incident charts",
        _DETAIL_HEADING,
        "Incident explanations",
        "Incident chatbot",
    ):
        assert expected in headings
    # The detail view sits between the charts and the existing explanations list.
    assert headings.index("Incident charts") < headings.index(_DETAIL_HEADING) < headings.index(
        "Incident explanations"
    )
    assert len(_charts(at)) == 2
    assert len(at.metric) == 3
    assert len(at.chat_input) == 1
    # The per-rank explanation expanders are still rendered for every explanation.
    result = at.session_state["analysis_result"]
    assert len(at.get("expander")) == len(result.explanations) + 1   # + the detail view's own


@needs_streamlit
def test_e14_the_detail_view_needs_no_helper_that_is_not_in_ui_support(tmp_path):
    """The app must delegate the lookup, not reimplement it."""
    import inspect

    import app as app_module

    source = inspect.getsource(app_module._render_cluster_detail)
    assert "cluster_options(" in source
    assert "cluster_log_entries(" in source
    assert "find_explanation(" in source
    assert "run_workflow(" not in source
    assert "run_pipeline(" not in source
    # And the helpers it calls are the same objects ui_support exports.
    assert app_module.cluster_log_entries is cluster_log_entries
