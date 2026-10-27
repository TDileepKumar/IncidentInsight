"""IncidentInsight - Streamlit application (Day 16).

Launch with:

    streamlit run app.py

What this UI does:
  * Lets you supply logs by upload, by pasting text, or by file path, choose the
    clustering threshold, and run the analysis ONCE (via src.workflow.run_workflow).
  * Shows the ranked incident clusters, count/severity charts, and their
    deterministic explanations.
  * Lets you pick one incident cluster and inspect the original log entries that
    belong to it, together with the explanation already produced for it.
  * Provides an incident-aware chatbot (src.chatbot) that answers questions about
    the analysis that is already in memory.

Design notes (approved Day 16):
  * The pipeline runs ONLY when the "Run analysis" button is pressed. The result
    is kept in `st.session_state`, so Streamlit's automatic reruns (every widget
    interaction, including each chat message and each cluster selection) never
    re-analyse the logs.
  * The cluster detail view is a pure lookup into that stored result: it filters
    the clustered rows the pipeline already produced and reuses the explanation
    already generated for the selected cluster. It never re-clusters, re-reads the
    log file, regenerates an explanation, or calls the LLM.
  * Cluster ids are only meaningful for one analysis run, so the selection is
    resolved against the CURRENT result and reset whenever a new analysis replaces
    it. A stored id that no longer exists falls back to the best-ranked cluster.
  * All heavy lifting stays in the existing modules. This file only renders and
    wires user input to them - no algorithm is reimplemented here. Input
    validation, chart data, and cluster-detail lookups live in `src/ui_support.py`,
    which imports no Streamlit and is unit-tested on its own.
  * Uploaded or pasted logs are written to a private temporary file (the existing
    workflow takes a path) and deleted as soon as the result is in memory.
    Nothing is stored in the project tree or the database.
  * The OpenAI API key is read from the environment by the chatbot itself. This
    file never displays, logs, or stores the key; it only shows whether one is
    configured. Without a key the chatbot answers offline and is labelled as a
    deterministic fallback, never as an AI response.
"""

from __future__ import annotations

import os

import streamlit as st

from src import config
from src.chatbot import SOURCE_OPENAI, STATUS_FAILURE, chat_with_incidentinsight
from src.llm_explainer import API_KEY_ENV_VAR
from src.log_normalizer import NORMALIZED_COLUMN
from src.ui_support import (
    DETAIL_PREVIEW_ROWS,
    cluster_log_entries,
    cluster_options,
    find_explanation,
    incident_count_data,
    remove_temp_log,
    resolve_log_path,
    severity_distribution_data,
    validate_pasted_log,
    validate_uploaded_log,
    write_temp_log,
)
from src.workflow import run_workflow

# Session-state keys (kept in one place so they are easy to follow).
_KEY_RESULT = "analysis_result"
_KEY_HISTORY = "chat_history"      # role/content pairs sent back to the model
_KEY_TRANSCRIPT = "chat_transcript"  # what the UI displays (adds source labels)
_KEY_CLUSTER = "selected_cluster"   # cluster id chosen in the detail view

_TOP_N = 10
_DISPLAY_COLUMNS = ["rank", "cluster_id", "size", "frequency_score", "severity_score", "risk_score"]

# The three ways a user can supply logs. The path option is the default so the
# bundled synthetic dataset stays one click away.
_SOURCE_UPLOAD = "Upload a log file"
_SOURCE_PASTE = "Paste log text"
_SOURCE_PATH = "Use a file path"
_INPUT_SOURCES = (_SOURCE_UPLOAD, _SOURCE_PASTE, _SOURCE_PATH)


def _init_session_state() -> None:
    """Create the session-state slots once, so reruns never lose them."""
    if _KEY_RESULT not in st.session_state:
        st.session_state[_KEY_RESULT] = None
    if _KEY_HISTORY not in st.session_state:
        st.session_state[_KEY_HISTORY] = []
    if _KEY_TRANSCRIPT not in st.session_state:
        st.session_state[_KEY_TRANSCRIPT] = []
    if _KEY_CLUSTER not in st.session_state:
        st.session_state[_KEY_CLUSTER] = None


def _api_key_configured() -> bool:
    """Whether an OpenAI key is present in the environment (never its value)."""
    return bool(os.environ.get(API_KEY_ENV_VAR, "").strip())


def _prepare_log_input(source: str, uploaded, pasted: str, log_path: str) -> "tuple[str, str | None, bool]":
    """Resolve the chosen input into (path, warning, owns_temporary_file).

    Uploads and pasted text are written to a private temporary file because the
    existing workflow takes a log path; the caller deletes it once the result is
    in memory. A path input is used as given, so no temporary file is created.
    """
    if source == _SOURCE_UPLOAD:
        if uploaded is None:
            raise ValueError("Choose a log file to upload first (.log or .txt).")
        text, warning = validate_uploaded_log(uploaded.name, uploaded.getvalue())
        return write_temp_log(text), warning, True
    if source == _SOURCE_PASTE:
        text, warning = validate_pasted_log(pasted)
        return write_temp_log(text), warning, True
    return resolve_log_path(log_path), None, False


def _render_sidebar() -> None:
    """Analysis controls. Running the pipeline is an explicit button press."""
    st.sidebar.header("Analysis")
    source = st.sidebar.radio(
        "Log input",
        _INPUT_SOURCES,
        index=_INPUT_SOURCES.index(_SOURCE_PATH),
        help="Upload a log file, paste log lines, or point at a file on disk.",
    )

    uploaded = None
    pasted = ""
    log_path = ""
    if source == _SOURCE_UPLOAD:
        uploaded = st.sidebar.file_uploader(
            "Log file",
            type=["log", "txt"],
            help="Plain-text application logs. Nothing is kept after the analysis.",
        )
    elif source == _SOURCE_PASTE:
        pasted = st.sidebar.text_area(
            "Paste log lines",
            height=200,
            placeholder="2026-10-01T00:00:00Z ERROR [req-1] [DatabaseService] Database connection failed",
            help="One log entry per line, in the same format as the raw log file.",
        )
    else:
        log_path = st.sidebar.text_input(
            "Log file path",
            value=str(config.RAW_LOG_PATH),
            help="Path to the raw log file to analyse.",
        )

    threshold = st.sidebar.slider(
        "Clustering distance threshold",
        min_value=0.1,
        max_value=1.0,
        value=float(config.CLUSTER_DISTANCE_THRESHOLD),
        step=0.05,
        help="Smaller values produce more, tighter incident clusters.",
    )
    use_llm = st.sidebar.checkbox(
        "LLM-enhanced explanations",
        value=False,
        help="Optional. Falls back to the deterministic explanation without an API key.",
    )

    if st.sidebar.button("Run analysis", type="primary"):
        # The ONLY place the pipeline is invoked. The result is cached in session
        # state so subsequent reruns (including chat turns) reuse it.
        try:
            resolved_path, warning, owns_temp = _prepare_log_input(source, uploaded, pasted, log_path)
        except ValueError as exc:
            # Bad input is reported instead of analysed; the previous result stays.
            st.sidebar.error(str(exc))
            return
        if warning:
            st.sidebar.warning(warning)
        try:
            result = run_workflow(
                log_path=resolved_path,
                distance_threshold=threshold,
                use_llm=use_llm,
            )
        finally:
            if owns_temp:
                remove_temp_log(resolved_path)
        st.session_state[_KEY_RESULT] = result
        # A fresh analysis starts a fresh conversation context.
        st.session_state[_KEY_HISTORY] = []
        st.session_state[_KEY_TRANSCRIPT] = []
        # Cluster ids belong to one run only, so the detail-view selection must
        # not survive into the next result.
        st.session_state[_KEY_CLUSTER] = None


def _render_charts(ranking) -> None:
    """Count and severity charts, built only from the real ranking output."""
    counts = incident_count_data(ranking, top_n=_TOP_N)
    severities = severity_distribution_data(ranking)

    st.subheader("Incident charts")
    if counts.empty and severities.empty:
        st.caption("No incidents to chart.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Log entries per incident (top {len(counts)} by rank)**")
        if counts.empty:
            st.caption("No clusters to chart.")
        else:
            st.bar_chart(
                counts,
                x_label="Incident (rank, cluster id)",
                y_label="Log entries",
                sort=False,   # keep the ranker's order instead of sorting by height
            )
    with right:
        st.markdown("**Incident clusters by maximum severity level**")
        if severities.empty:
            st.caption("No severity scores to chart.")
        else:
            st.bar_chart(
                severities,
                x_label="Maximum severity level",
                y_label="Incident clusters",
                sort=False,   # keep the DEBUG -> FATAL ladder order
            )


def _render_explanation(explanation: dict) -> None:
    """Render one explanation dict.

    Shared by the per-rank expanders and the cluster detail view so both show the
    SAME already-generated explanation; neither path generates one.
    """
    st.write(explanation.get("summary", ""))
    source = explanation.get("explanation_source")
    if source == SOURCE_OPENAI:
        st.caption("Source: AI-generated (OpenAI)")
    elif source:
        st.caption("Source: deterministic explanation")
    evidence = explanation.get("evidence") or {}
    if evidence:
        st.markdown("**Evidence**")
        for key, value in evidence.items():
            st.markdown(f"- `{key}`: {value}")
    checks = explanation.get("suggested_checks") or []
    if checks:
        st.markdown("**Suggested checks**")
        for check in checks:
            st.markdown(f"- {check}")
    limitations = explanation.get("limitations") or []
    if limitations:
        st.markdown("**Limitations**")
        for limitation in limitations:
            st.markdown(f"- {limitation}")


def _render_cluster_detail(result) -> None:
    """Inspect the original log entries behind one selected incident cluster.

    A pure read of the analysis already in session state: selecting a different
    cluster only filters the stored rows and looks up the stored explanation. The
    selection is reconciled against the CURRENT ranking, so a cluster id left over
    from an earlier run can never show that run's logs or explanation.
    """
    st.subheader("Cluster detail")
    labels = cluster_options(result.analysis.ranking)
    if not labels:
        st.caption("No incident clusters to inspect in the current analysis.")
        return

    cluster_ids = list(labels)
    stored = st.session_state.get(_KEY_CLUSTER)
    index = cluster_ids.index(stored) if stored in cluster_ids else 0
    selected = st.selectbox(
        "Incident cluster",
        cluster_ids,
        index=index,
        format_func=lambda cluster_id: labels[cluster_id],
        help="Choose a ranked incident to see its original log entries.",
    )
    st.session_state[_KEY_CLUSTER] = selected

    entries = cluster_log_entries(result.analysis.clustered, selected)
    if entries.empty:
        st.caption("No log entries are stored for this cluster in the current analysis.")
    else:
        preview = entries[[c for c in entries.columns if c != NORMALIZED_COLUMN]]
        noun = "entry" if len(entries) == 1 else "entries"
        st.markdown(f"**{len(entries)} log {noun} in this cluster**")
        st.dataframe(preview.head(DETAIL_PREVIEW_ROWS))
        if len(entries) > DETAIL_PREVIEW_ROWS:
            st.caption(f"Showing the first {DETAIL_PREVIEW_ROWS} of {len(entries)} entries.")
        with st.expander(f"All {len(entries)} entries (with normalized messages)", expanded=False):
            st.dataframe(entries)

    st.markdown("**Explanation for this cluster**")
    explanation = find_explanation(result.explanations, selected)
    if explanation is None:
        st.caption("No explanation is available for this cluster in the current analysis run.")
    else:
        _render_explanation(explanation)


def _render_analysis(result) -> None:
    """Show the ranked incidents and their explanations."""
    if result is None:
        st.info("No analysis yet. Choose a log file in the sidebar and click **Run analysis**.")
        return

    if result.analysis_status != "success":
        st.error(f"Analysis failed: {result.errors.get('analysis', 'unknown error')}")
        return

    st.subheader("Analysis summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Log entries", result.analysis.n_logs)
    col2.metric("Incident clusters", result.analysis.n_clusters)
    col3.metric("Explanations", len(result.explanations))

    if result.llm_status == "fallback":
        st.caption(
            "LLM enhancement unavailable - deterministic explanations are shown "
            f"({result.errors.get('llm', '')})."
        )

    st.subheader(f"Top {min(_TOP_N, len(result.analysis.ranking))} incident clusters")
    ranking = result.analysis.ranking
    columns = [c for c in _DISPLAY_COLUMNS if c in ranking.columns]
    st.dataframe(ranking.head(_TOP_N)[columns])

    _render_charts(ranking)
    _render_cluster_detail(result)

    st.subheader("Incident explanations")
    for explanation in result.explanations[:_TOP_N]:
        title = f"Rank {explanation.get('rank')} - cluster {explanation.get('cluster_id')}"
        with st.expander(title, expanded=False):
            _render_explanation(explanation)


def _render_chatbot(result) -> None:
    """The incident-aware chatbot section.

    The chatbot receives the analysis result already held in session state, so a
    question never re-runs the pipeline. Conversation history is per-session (it
    lives in this browser session's state) and is never written to the database.
    """
    st.subheader("Incident chatbot")
    if _api_key_configured():
        st.caption("OpenAI API key: configured. Answers will be AI-generated.")
    else:
        st.caption(
            f"OpenAI API key: not configured ({API_KEY_ENV_VAR}). Answers are "
            "deterministic offline responses built from the analysis, and are "
            "labelled as such."
        )

    # Replay the existing transcript so it survives Streamlit reruns.
    for entry in st.session_state[_KEY_TRANSCRIPT]:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("label"):
                st.caption(entry["label"])
            if entry.get("error"):
                st.warning(entry["error"])

    question = st.chat_input("Ask about the analysed incidents...")
    if not question:
        return

    # Echo the user's turn immediately.
    st.session_state[_KEY_TRANSCRIPT].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # History is passed in (bounded by the chatbot); the result is appended after.
    response = chat_with_incidentinsight(
        question,
        context=result,
        history=list(st.session_state[_KEY_HISTORY]),
    )

    if response.status == STATUS_FAILURE:
        label = "Could not answer"
    elif response.response_source == SOURCE_OPENAI:
        label = "AI-generated (OpenAI)"
    else:
        label = "Deterministic fallback (no AI model was used)"

    st.session_state[_KEY_HISTORY].append({"role": "user", "content": question})
    st.session_state[_KEY_HISTORY].append({"role": "assistant", "content": response.answer})
    st.session_state[_KEY_TRANSCRIPT].append(
        {
            "role": "assistant",
            "content": response.answer,
            "label": label,
            "error": response.error if response.status == STATUS_FAILURE else None,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(response.answer)
        st.caption(label)
        if response.status == STATUS_FAILURE and response.error:
            st.warning(response.error)


def main() -> None:
    """Render the application."""
    st.set_page_config(page_title="IncidentInsight", page_icon=None, layout="wide")
    _init_session_state()

    st.title("IncidentInsight")
    st.caption("Application log analysis, incident ranking, and AI-assisted troubleshooting.")

    _render_sidebar()
    result = st.session_state[_KEY_RESULT]
    _render_analysis(result)

    st.divider()
    _render_chatbot(result)


if __name__ == "__main__":
    main()
