"""Day 14 - End-to-end workflow orchestration and consistent error handling.

A THIN coordination layer over the existing, already-tested modules. It runs the
analysis pipeline once and wires its output through the downstream stages, so a
caller (or a future UI) gets one structured result instead of juggling several
modules and their individual exceptions.

Stages, in order:
    1. Core analysis      -> src.pipeline.run_pipeline()            [required]
    2. Deterministic expl. -> src.incident_explainer.explain_ranked_incidents()  [required]
    3. Optional LLM enh.   -> src.llm_explainer.explain_incident_with_llm()      [optional]
    4. Optional persistence-> src.database.save_incidents()                       [optional]

Design rules (approved Day 14):
  * ORCHESTRATE ONLY. Every algorithm stays in its own module and is reused
    as-is; nothing is reimplemented here. No second clustering, ranking,
    explanation engine, database layer, or LLM client is introduced.
  * The pipeline runs EXACTLY ONCE per workflow request. Explanations and
    persistence consume that single result - they never re-run the pipeline.
  * Failures are stage-specific and never masked. A core-analysis failure is
    reported as such (no fabricated incidents); an optional LLM or persistence
    failure is visible in its own status field and does NOT erase a successful
    core analysis.
  * Optional stages are OFF by default. The workflow is fully usable offline:
    with ``use_llm=False`` and ``persist=False`` it never touches the network or
    the database, and no API key is required.
  * Original exceptions are summarized (type + message) into ``errors`` for
    debugging; nothing sensitive (no API keys) is ever recorded.

Overall success contract:
    ``result.success`` is True ONLY when the two REQUIRED stages complete -
    core analysis and deterministic explanations. The optional LLM and
    persistence stages report their own status and never, by themselves, make
    ``success`` False (an LLM fallback or a persistence failure is surfaced in
    ``llm_status`` / ``persistence_status`` while ``success`` stays True).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src import config
from src.database import save_incidents
from src.incident_explainer import explain_ranked_incidents
from src.llm_explainer import (
    DEFAULT_TIMEOUT,
    SOURCE_FALLBACK,
    SOURCE_OPENAI,
    explain_incident_with_llm,
)
from src.log_normalizer import NORMALIZED_COLUMN
from src.pipeline import PipelineResult, run_pipeline

# --- Stage status vocabulary (documented string constants) --------------------
STATUS_SUCCESS = "success"    # the stage ran and produced its result
STATUS_FAILED = "failed"      # the stage ran but errored
STATUS_SKIPPED = "skipped"    # the stage was not run (disabled, or nothing to do)

# LLM-specific statuses (only meaningful when use_llm=True).
LLM_ENHANCED = "enhanced"     # every incident got an OpenAI explanation
LLM_FALLBACK = "fallback"     # at least one incident used the deterministic fallback


def _describe(exc: BaseException) -> str:
    """A short, non-sensitive 'Type: message' summary of an exception.

    Used to preserve the original failure for debugging/testing without ever
    recording request headers, API keys, or raw provider bodies.
    """
    return f"{type(exc).__name__}: {exc}"


@dataclass
class WorkflowResult:
    """Structured outcome of one ``run_workflow()`` request.

    Attributes:
        success: True only when BOTH required stages (core analysis and
            deterministic explanations) completed. Optional-stage problems do
            not, by themselves, make this False.
        analysis_status: "success" or "failed" for the core pipeline stage.
        analysis: the PipelineResult when analysis succeeded, else None. It is
            preserved even if a later optional stage fails, so a successful core
            analysis is never discarded.
        explanation_status: "success", "failed", or "skipped" (skipped only when
            core analysis failed, so there was nothing to explain).
        explanations: the incident explanation dicts. These are the LLM-enhanced
            explanations when LLM enhancement fully succeeded, otherwise the
            deterministic explanations (including per-incident LLM fallbacks).
        llm_status: "skipped" (LLM disabled or nothing to enhance), "enhanced"
            (all incidents enhanced by OpenAI), "fallback" (deterministic
            fallback used for at least one incident), or "failed" (the LLM stage
            itself errored; deterministic explanations were retained).
        persistence_status: "skipped" (persistence disabled or nothing to save),
            "success", or "failed".
        persisted_count: number of records written when persistence succeeded,
            else None.
        errors: stage-name -> non-sensitive message. Populated for any stage
            failure, and for the "llm" key when a fallback was used (to describe
            why). Never contains secrets.
    """

    success: bool = False
    analysis_status: str = STATUS_FAILED
    analysis: PipelineResult | None = None
    explanation_status: str = STATUS_SKIPPED
    explanations: list = field(default_factory=list)
    llm_status: str = STATUS_SKIPPED
    persistence_status: str = STATUS_SKIPPED
    persisted_count: int | None = None
    errors: dict = field(default_factory=dict)


def run_workflow(
    log_path: "str | None" = None,
    *,
    distance_threshold: float = config.CLUSTER_DISTANCE_THRESHOLD,
    column: str = NORMALIZED_COLUMN,
    use_llm: bool = False,
    llm_api_key: "str | None" = None,
    llm_model: "str | None" = None,
    llm_timeout: float = DEFAULT_TIMEOUT,
    persist: bool = False,
    db_path: "str | None" = None,
) -> WorkflowResult:
    """Coordinate analysis -> explanations -> optional LLM -> optional persist.

    Args:
        log_path: path to the raw log file. Defaults to the generated dataset
            (``config.RAW_LOG_PATH``) exactly like ``run_pipeline``.
        distance_threshold: clustering threshold, passed straight to the
            existing pipeline (validated there). Not re-tuned here.
        column: text column to cluster on, passed through to the pipeline.
        use_llm: when True, enhance each incident with the existing optional
            OpenAI integration (which safely falls back to the deterministic
            explanation when no key is present or the provider fails). Default
            False keeps the workflow fully offline.
        llm_api_key / llm_model / llm_timeout: forwarded unchanged to
            ``explain_incident_with_llm``. The key is never logged or stored.
        persist: when True, save the final explanations via the existing
            database API. Default False skips persistence entirely.
        db_path: target SQLite file; REQUIRED when ``persist`` is True.

    Returns:
        A WorkflowResult. It never raises for a core-analysis, explanation, LLM,
        or persistence failure - those are reported in the result. It raises only
        for a clear caller error in the workflow's OWN arguments.

    Raises:
        ValueError: ``persist`` is True but ``db_path`` was not supplied. This is
            a caller/programmer error and is detected before any work is done.

    The pipeline is invoked exactly once. Explanations and persistence reuse
    that single result and never re-run clustering or ranking.
    """
    # --- Workflow argument validation (fail fast, before doing any work) ------
    if persist and db_path is None:
        raise ValueError("db_path is required when persist=True")

    errors: dict = {}

    # --- Stage 1: core analysis (run the pipeline EXACTLY once) ---------------
    effective_log_path = config.RAW_LOG_PATH if log_path is None else log_path
    try:
        analysis = run_pipeline(
            log_path=effective_log_path,
            distance_threshold=distance_threshold,
            column=column,
        )
    except Exception as exc:  # noqa: BLE001 - orchestrator reports, never masks
        # Core analysis failed: report it honestly with NO fabricated incidents.
        errors["analysis"] = _describe(exc)
        return WorkflowResult(
            success=False,
            analysis_status=STATUS_FAILED,
            analysis=None,
            explanation_status=STATUS_SKIPPED,
            explanations=[],
            llm_status=STATUS_SKIPPED,
            persistence_status=STATUS_SKIPPED,
            persisted_count=None,
            errors=errors,
        )

    # --- Stage 2: deterministic explanations (required, trusted baseline) -----
    try:
        explanations = explain_ranked_incidents(analysis.ranking)
    except Exception as exc:  # noqa: BLE001 - keep the successful analysis
        errors["explanation"] = _describe(exc)
        return WorkflowResult(
            success=False,
            analysis_status=STATUS_SUCCESS,
            analysis=analysis,  # core analysis is preserved despite the failure
            explanation_status=STATUS_FAILED,
            explanations=[],
            llm_status=STATUS_SKIPPED,
            persistence_status=STATUS_SKIPPED,
            persisted_count=None,
            errors=errors,
        )

    explanation_status = STATUS_SUCCESS
    llm_status = STATUS_SKIPPED

    # --- Stage 3: optional LLM enhancement (reuses the existing integration) --
    # Only attempted when requested and when there is at least one incident. On
    # any LLM-stage error we KEEP the deterministic explanations computed above.
    if use_llm and explanations:
        try:
            records = analysis.ranking.to_dict("records")
            enhanced = [
                explain_incident_with_llm(
                    record,
                    api_key=llm_api_key,
                    model=llm_model,
                    timeout=llm_timeout,
                )
                for record in records
            ]
        except Exception as exc:  # noqa: BLE001 - optional stage must not crash
            llm_status = STATUS_FAILED
            errors["llm"] = _describe(exc)
            # explanations remain the deterministic baseline
        else:
            explanations = enhanced
            sources = [item.get("explanation_source") for item in enhanced]
            if all(source == SOURCE_OPENAI for source in sources):
                llm_status = LLM_ENHANCED
            else:
                llm_status = LLM_FALLBACK
                reasons = sorted(
                    {
                        item.get("fallback_reason")
                        for item in enhanced
                        if item.get("explanation_source") == SOURCE_FALLBACK
                        and item.get("fallback_reason")
                    }
                )
                n_fallback = sum(1 for s in sources if s == SOURCE_FALLBACK)
                errors["llm"] = (
                    f"deterministic fallback used for {n_fallback} incident(s); "
                    f"reason(s): {', '.join(reasons) if reasons else 'unknown'}"
                )

    # --- Stage 4: optional SQLite persistence (explicit only) -----------------
    persistence_status = STATUS_SKIPPED
    persisted_count = None
    if persist and explanations:
        try:
            persisted_count = save_incidents(explanations, db_path)
            persistence_status = STATUS_SUCCESS
        except Exception as exc:  # noqa: BLE001 - never lose the in-memory result
            persistence_status = STATUS_FAILED
            errors["persistence"] = _describe(exc)
            # analysis and explanations are retained in the returned result
    elif persist and not explanations:
        # Persistence was requested but there was nothing valid to save.
        errors["persistence"] = "no explanations available to persist"

    # Overall success = both REQUIRED stages completed. Optional stages report
    # their own status and do not, by themselves, flip success to False.
    success = explanation_status == STATUS_SUCCESS

    return WorkflowResult(
        success=success,
        analysis_status=STATUS_SUCCESS,
        analysis=analysis,
        explanation_status=explanation_status,
        explanations=explanations,
        llm_status=llm_status,
        persistence_status=persistence_status,
        persisted_count=persisted_count,
        errors=errors,
    )
