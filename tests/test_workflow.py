"""Day 14 integration tests for the workflow orchestrator.

These tests exercise the wiring between the existing stages without depending on
a live LLM provider, a real network, or the project's generated dataset. The
stage boundaries (run_pipeline, explain_ranked_incidents,
explain_incident_with_llm, save_incidents) are mocked where a failure must be
simulated deterministically; a few tests use the REAL downstream modules (with
tmp_path databases and no API key) to prove genuine end-to-end integration.
"""

from unittest import mock

import pandas as pd
import pytest

from src import config, workflow
from src.database import get_incidents
from src.llm_explainer import SOURCE_FALLBACK, SOURCE_OPENAI
from src.pipeline import PipelineResult
from src.workflow import (
    LLM_ENHANCED,
    LLM_FALLBACK,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    WorkflowResult,
    run_workflow,
)

# Ranking columns produced by src/ranker.py (OUTPUT_COLUMNS).
_RANKING_COLUMNS = [
    "cluster_id",
    "size",
    "representative_message",
    "severity_score",
    "frequency_score",
    "risk_score",
    "rank",
]


def _ranking_df():
    """A small, realistic ranked-incident table (two clusters)."""
    return pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "size": 42,
                "representative_message": "Database connection failed after 3s",
                "severity_score": 1.0,
                "frequency_score": 0.9,
                "risk_score": 0.95,
                "rank": 1,
            },
            {
                "cluster_id": 1,
                "size": 7,
                "representative_message": "Payment gateway timeout while charging card",
                "severity_score": 0.8,
                "frequency_score": 0.3,
                "risk_score": 0.55,
                "rank": 2,
            },
        ],
        columns=_RANKING_COLUMNS,
    )


def _fake_result(ranking=None):
    """A PipelineResult-shaped object the workflow can consume."""
    ranking = _ranking_df() if ranking is None else ranking
    return PipelineResult(
        clustered=ranking.copy(),
        ranking=ranking,
        n_logs=int(ranking["size"].sum()) if len(ranking) else 0,
        n_clusters=int(len(ranking)),
    )


# --- 1. Normal workflow -------------------------------------------------------
def test_normal_workflow_succeeds():
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()):
        result = run_workflow(log_path="ignored.log")

    assert isinstance(result, WorkflowResult)
    assert result.success is True
    assert result.analysis_status == STATUS_SUCCESS
    assert result.explanation_status == STATUS_SUCCESS
    assert result.llm_status == STATUS_SKIPPED
    assert result.persistence_status == STATUS_SKIPPED
    assert result.persisted_count is None
    assert len(result.explanations) == 2
    # Explanations carry the Day 11 contract (no explanation_source when no LLM).
    assert set(result.explanations[0]) == {
        "cluster_id",
        "rank",
        "summary",
        "evidence",
        "suggested_checks",
        "limitations",
    }


def test_normal_workflow_preserves_pipeline_result():
    fake = _fake_result()
    with mock.patch.object(workflow, "run_pipeline", return_value=fake):
        result = run_workflow(log_path="ignored.log")
    assert result.analysis is fake


# --- 2. Pipeline invocation count --------------------------------------------
def test_pipeline_called_exactly_once_plain():
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()) as rp:
        run_workflow(log_path="ignored.log")
    assert rp.call_count == 1


def test_pipeline_called_once_with_llm_and_persist(tmp_path):
    db = tmp_path / "wf.db"
    enhanced = {"explanation_source": SOURCE_OPENAI}
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()) as rp, \
         mock.patch.object(workflow, "explain_incident_with_llm", return_value=enhanced), \
         mock.patch.object(workflow, "save_incidents", return_value=2):
        run_workflow(log_path="ignored.log", use_llm=True, persist=True, db_path=str(db))
    assert rp.call_count == 1


def test_pipeline_args_forwarded():
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()) as rp:
        run_workflow(log_path="custom.log", distance_threshold=0.3, column="some_col")
    rp.assert_called_once_with(
        log_path="custom.log", distance_threshold=0.3, column="some_col"
    )


def test_default_log_path_uses_config():
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()) as rp:
        run_workflow()
    assert rp.call_args.kwargs["log_path"] == config.RAW_LOG_PATH


# --- 3. Pipeline failure ------------------------------------------------------
def test_pipeline_failure_reported_not_masked():
    with mock.patch.object(workflow, "run_pipeline", side_effect=ValueError("no log entries")), \
         mock.patch.object(workflow, "explain_ranked_incidents") as eri:
        result = run_workflow(log_path="ignored.log")

    assert result.success is False
    assert result.analysis_status == STATUS_FAILED
    assert result.analysis is None
    assert result.explanations == []
    assert result.explanation_status == STATUS_SKIPPED
    assert "analysis" in result.errors
    assert "no log entries" in result.errors["analysis"]
    # Downstream stages must not run on a failed analysis (no fabricated output).
    eri.assert_not_called()


def test_pipeline_filenotfound_is_analysis_failure():
    with mock.patch.object(workflow, "run_pipeline", side_effect=FileNotFoundError("missing")):
        result = run_workflow(log_path="nope.log")
    assert result.success is False
    assert result.analysis_status == STATUS_FAILED
    assert "FileNotFoundError" in result.errors["analysis"]


def test_real_empty_log_file_is_analysis_failure(tmp_path):
    empty = tmp_path / "empty.log"
    empty.write_text("", encoding="utf-8")
    result = run_workflow(log_path=empty)  # real run_pipeline
    assert result.success is False
    assert result.analysis_status == STATUS_FAILED
    assert result.explanations == []


# --- 4. LLM fallback and enhancement -----------------------------------------
def test_llm_fallback_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()):
        result = run_workflow(log_path="ignored.log", use_llm=True)

    assert result.success is True
    assert result.llm_status == LLM_FALLBACK
    assert "llm" in result.errors
    assert "missing_api_key" in result.errors["llm"]
    assert all(e["explanation_source"] == SOURCE_FALLBACK for e in result.explanations)


def test_llm_enhanced_when_all_openai():
    def fake_llm(record, **kwargs):
        return {"cluster_id": record["cluster_id"], "explanation_source": SOURCE_OPENAI}

    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()), \
         mock.patch.object(workflow, "explain_incident_with_llm", side_effect=fake_llm):
        result = run_workflow(log_path="ignored.log", use_llm=True)

    assert result.llm_status == LLM_ENHANCED
    assert "llm" not in result.errors
    assert all(e["explanation_source"] == SOURCE_OPENAI for e in result.explanations)


def test_llm_mixed_sources_report_fallback():
    calls = {"n": 0}

    def fake_llm(record, **kwargs):
        calls["n"] += 1
        source = SOURCE_OPENAI if calls["n"] == 1 else SOURCE_FALLBACK
        out = {"cluster_id": record["cluster_id"], "explanation_source": source}
        if source == SOURCE_FALLBACK:
            out["fallback_reason"] = "timeout"
        return out

    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()), \
         mock.patch.object(workflow, "explain_incident_with_llm", side_effect=fake_llm):
        result = run_workflow(log_path="ignored.log", use_llm=True)

    assert result.llm_status == LLM_FALLBACK
    assert "timeout" in result.errors["llm"]


def test_llm_kwargs_forwarded():
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()), \
         mock.patch.object(
             workflow, "explain_incident_with_llm",
             return_value={"explanation_source": SOURCE_OPENAI},
         ) as llm:
        run_workflow(
            log_path="ignored.log", use_llm=True,
            llm_api_key="k", llm_model="m", llm_timeout=3.5,
        )
    assert llm.call_count == 2
    for call in llm.call_args_list:
        assert call.kwargs == {"api_key": "k", "model": "m", "timeout": 3.5}


def test_llm_disabled_does_not_call_llm():
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()), \
         mock.patch.object(workflow, "explain_incident_with_llm") as llm:
        run_workflow(log_path="ignored.log", use_llm=False)
    llm.assert_not_called()


# --- 5. Analysis preservation after LLM failure ------------------------------
def test_analysis_preserved_when_llm_stage_raises():
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()), \
         mock.patch.object(workflow, "explain_incident_with_llm", side_effect=RuntimeError("boom")):
        result = run_workflow(log_path="ignored.log", use_llm=True)

    assert result.success is True                    # optional stage failure
    assert result.analysis_status == STATUS_SUCCESS
    assert result.analysis is not None
    assert result.llm_status == STATUS_FAILED
    assert "boom" in result.errors["llm"]
    # Deterministic explanations are retained despite the LLM failure.
    assert len(result.explanations) == 2
    assert "explanation_source" not in result.explanations[0]


# --- 6. Persistence disabled --------------------------------------------------
def test_persistence_disabled_skips_db():
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()), \
         mock.patch.object(workflow, "save_incidents") as save:
        result = run_workflow(log_path="ignored.log", persist=False)
    save.assert_not_called()
    assert result.persistence_status == STATUS_SKIPPED
    assert result.persisted_count is None


def test_persist_true_without_db_path_raises():
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()):
        with pytest.raises(ValueError, match="db_path is required"):
            run_workflow(log_path="ignored.log", persist=True)


# --- 7. Persistence failure ---------------------------------------------------
def test_persistence_failure_preserves_analysis():
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()), \
         mock.patch.object(workflow, "save_incidents", side_effect=OSError("disk full")):
        result = run_workflow(log_path="ignored.log", persist=True, db_path="x.db")

    assert result.success is True                    # persistence is optional
    assert result.analysis_status == STATUS_SUCCESS
    assert result.analysis is not None
    assert len(result.explanations) == 2             # not discarded
    assert result.persistence_status == STATUS_FAILED
    assert result.persisted_count is None
    assert "disk full" in result.errors["persistence"]


# --- 8. Persistence success ---------------------------------------------------
def test_persistence_success_calls_save_once():
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()), \
         mock.patch.object(workflow, "save_incidents", return_value=2) as save:
        result = run_workflow(log_path="ignored.log", persist=True, db_path="out.db")

    save.assert_called_once()
    args, _ = save.call_args
    assert args[1] == "out.db"
    assert len(args[0]) == 2                         # the explanations list
    assert result.persistence_status == STATUS_SUCCESS
    assert result.persisted_count == 2


def test_persistence_real_sqlite_roundtrip(tmp_path):
    db = tmp_path / "real.db"
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()):
        result = run_workflow(log_path="ignored.log", persist=True, db_path=db)

    assert result.persistence_status == STATUS_SUCCESS
    assert result.persisted_count == 2
    stored = get_incidents(db)                        # real DB read-back
    assert len(stored) == 2
    assert {row["cluster_id"] for row in stored} == {0, 1}


# --- 9. Explanation failure ---------------------------------------------------
def test_explanation_failure_reported_accurately():
    fake = _fake_result()
    with mock.patch.object(workflow, "run_pipeline", return_value=fake), \
         mock.patch.object(workflow, "explain_ranked_incidents", side_effect=ValueError("bad row")):
        result = run_workflow(log_path="ignored.log")

    assert result.success is False
    assert result.analysis_status == STATUS_SUCCESS   # analysis still succeeded
    assert result.analysis is fake                   # and is preserved
    assert result.explanation_status == STATUS_FAILED
    assert result.explanations == []
    assert result.llm_status == STATUS_SKIPPED
    assert result.persistence_status == STATUS_SKIPPED
    assert "bad row" in result.errors["explanation"]


def test_explanation_failure_skips_persistence(tmp_path):
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()), \
         mock.patch.object(workflow, "explain_ranked_incidents", side_effect=ValueError("x")), \
         mock.patch.object(workflow, "save_incidents") as save:
        run_workflow(log_path="ignored.log", persist=True, db_path=str(tmp_path / "d.db"))
    save.assert_not_called()


# --- 10. Invalid input --------------------------------------------------------
def test_invalid_log_path_type_is_analysis_failure():
    # Path(int) raises TypeError inside run_pipeline -> reported, not propagated.
    with mock.patch.object(workflow, "run_pipeline", side_effect=TypeError("bad path")):
        result = run_workflow(log_path=123)
    assert result.success is False
    assert result.analysis_status == STATUS_FAILED


# --- 11. Result structure distinguishable ------------------------------------
def test_result_fields_are_distinguishable():
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()):
        result = run_workflow(log_path="ignored.log")
    for field_name in (
        "success", "analysis_status", "analysis", "explanation_status",
        "explanations", "llm_status", "persistence_status", "persisted_count",
        "errors",
    ):
        assert hasattr(result, field_name)


def test_empty_ranking_yields_empty_explanations():
    empty = _ranking_df().iloc[0:0]
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result(empty)):
        result = run_workflow(log_path="ignored.log", use_llm=True, persist=True, db_path="x.db")
    assert result.success is True
    assert result.explanations == []
    assert result.llm_status == STATUS_SKIPPED        # nothing to enhance
    assert result.persistence_status == STATUS_SKIPPED  # nothing to save


def test_persist_requested_but_nothing_to_save():
    empty = _ranking_df().iloc[0:0]
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result(empty)), \
         mock.patch.object(workflow, "save_incidents") as save:
        result = run_workflow(log_path="ignored.log", persist=True, db_path="x.db")
    save.assert_not_called()
    assert result.persistence_status == STATUS_SKIPPED
    assert "persistence" in result.errors


# --- 12. No unintended repetition --------------------------------------------
def test_no_second_pipeline_or_explanation_pass(tmp_path):
    with mock.patch.object(workflow, "run_pipeline", return_value=_fake_result()) as rp, \
         mock.patch.object(workflow, "explain_ranked_incidents", wraps=workflow.explain_ranked_incidents) as eri, \
         mock.patch.object(workflow, "explain_incident_with_llm", return_value={"explanation_source": SOURCE_OPENAI}), \
         mock.patch.object(workflow, "save_incidents", return_value=2):
        run_workflow(log_path="ignored.log", use_llm=True, persist=True, db_path=str(tmp_path / "d.db"))
    assert rp.call_count == 1
    assert eri.call_count == 1


# --- Real end-to-end smoke test on the generated dataset (optional) ----------
needs_generated_data = pytest.mark.skipif(
    not config.RAW_LOG_PATH.exists(), reason="run `python data_generator.py` first"
)


@needs_generated_data
def test_smoke_real_pipeline_offline_no_persistence():
    # Real pipeline + real deterministic explanations, LLM and persistence off.
    result = run_workflow()
    assert result.success is True
    assert result.analysis_status == STATUS_SUCCESS
    assert result.explanation_status == STATUS_SUCCESS
    assert result.llm_status == STATUS_SKIPPED
    assert result.persistence_status == STATUS_SKIPPED
    assert result.analysis.n_clusters == len(result.explanations)
