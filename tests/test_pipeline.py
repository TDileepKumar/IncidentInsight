"""Day 9 tests for pipeline orchestration and the CLI entry point."""

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

import run
from src import config, pipeline
from src.clusterer import CLUSTER_COLUMN
from src.log_normalizer import NORMALIZED_COLUMN
from src.pipeline import PipelineResult, run_pipeline
from src.ranker import OUTPUT_COLUMNS, rank_clusters

needs_generated_data = pytest.mark.skipif(
    not config.RAW_LOG_PATH.exists(), reason="run `python data_generator.py` first"
)

# Valid, parseable log lines in the format the Day-3 parser expects.
GOOD_LOG = """\
2026-10-01T00:00:00Z ERROR [req-1] [DatabaseService] Database connection failed after 3s
2026-10-01T00:00:01Z ERROR [req-2] [DatabaseService] Database connection failed after 7s
2026-10-01T00:00:02Z FATAL [req-3] [PaymentService] Payment declined by provider PayPal for order 12
2026-10-01T00:00:03Z WARNING [req-4] [ApiGateway] Upstream request timed out after 5s
"""

# Parseable lines whose messages carry NO TF-IDF features (punctuation only).
BLANK_MESSAGE_LOG = """\
2026-10-01T00:00:00Z ERROR [req-1] [Svc] ***
2026-10-01T00:00:01Z ERROR [req-2] [Svc] ---
"""


def _write(tmp_path, text, name="input.log"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- 1. Structured result -----------------------------------------------------
def test_valid_file_returns_structured_result(tmp_path):
    path = _write(tmp_path, GOOD_LOG)
    result = run_pipeline(path)

    assert isinstance(result, PipelineResult)
    assert isinstance(result.clustered, pd.DataFrame)
    assert isinstance(result.ranking, pd.DataFrame)
    assert result.n_logs == 4
    assert 2 <= result.n_clusters <= 4
    assert list(result.ranking.columns) == OUTPUT_COLUMNS
    assert len(result.ranking) == result.n_clusters


# --- 2. Clustered DataFrame contents ------------------------------------------
def test_clustered_dataframe_has_expected_columns(tmp_path):
    result = run_pipeline(_write(tmp_path, GOOD_LOG))
    cols = result.clustered.columns
    assert CLUSTER_COLUMN in cols
    assert NORMALIZED_COLUMN in cols
    assert "level" in cols
    assert "line_id" in cols
    assert len(result.clustered) == 4
    # Every row received a label.
    assert result.clustered[CLUSTER_COLUMN].notna().all()


# --- 3. Ranking consistent with rank_clusters() -------------------------------
def test_ranking_matches_rank_clusters(tmp_path):
    result = run_pipeline(_write(tmp_path, GOOD_LOG))
    expected = rank_clusters(result.clustered)
    assert result.ranking.equals(expected)


# --- 4. Missing file ----------------------------------------------------------
def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path / "does_not_exist.log")


def test_directory_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(tmp_path)  # a directory, not a file


# --- 5. Empty input -----------------------------------------------------------
def test_empty_file_raises(tmp_path):
    with pytest.raises(ValueError, match="no log entries"):
        run_pipeline(_write(tmp_path, ""))


def test_blank_lines_only_raises(tmp_path):
    with pytest.raises(ValueError, match="no log entries"):
        run_pipeline(_write(tmp_path, "\n\n   \n"))


# --- 6. Unusable messages -----------------------------------------------------
def test_all_unusable_messages_raise_clear_error(tmp_path):
    with pytest.raises(ValueError, match="features"):
        run_pipeline(_write(tmp_path, BLANK_MESSAGE_LOG))


# --- 7. Clustering performed exactly once -------------------------------------
def test_clustering_called_once(tmp_path):
    path = _write(tmp_path, GOOD_LOG)
    with mock.patch.object(pipeline, "cluster_dataframe", wraps=pipeline.cluster_dataframe) as spy:
        run_pipeline(path)
    assert spy.call_count == 1


# --- Threshold passthrough ----------------------------------------------------
def test_threshold_is_passed_through(tmp_path):
    path = _write(tmp_path, GOOD_LOG)
    with mock.patch.object(pipeline, "cluster_dataframe", wraps=pipeline.cluster_dataframe) as spy:
        run_pipeline(path, distance_threshold=0.4)
    assert spy.call_args.kwargs.get("distance_threshold") == 0.4


def test_invalid_threshold_propagates(tmp_path):
    with pytest.raises(ValueError):
        run_pipeline(_write(tmp_path, GOOD_LOG), distance_threshold=-1)


# --- 8/9. CLI behaviour --------------------------------------------------------
def test_cli_valid_input(capsys, tmp_path):
    rc = run.main([str(_write(tmp_path, GOOD_LOG))])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Processed 4 log entries" in out
    assert "incident clusters" in out


def test_cli_missing_file(capsys, tmp_path):
    rc = run.main([str(tmp_path / "nope.log")])
    assert rc == 1
    assert "Error" in capsys.readouterr().err


def test_cli_unusable_input(capsys, tmp_path):
    rc = run.main([str(_write(tmp_path, BLANK_MESSAGE_LOG, "blank.log"))])
    assert rc == 1
    assert "Error" in capsys.readouterr().err


def test_cli_invalid_top(capsys, tmp_path):
    rc = run.main([str(_write(tmp_path, GOOD_LOG)), "--top", "0"])
    assert rc == 2
    assert "Error" in capsys.readouterr().err


def test_cli_threshold_flag(capsys, tmp_path):
    rc = run.main([str(_write(tmp_path, GOOD_LOG)), "--threshold", "0.4", "--top", "2"])
    assert rc == 0
    assert "threshold=0.4" in capsys.readouterr().out


# --- 10. Default path (no working-directory assumption) ------------------------
def test_default_path_is_absolute_and_from_config():
    default = run.build_parser().parse_args([]).log_path
    assert default == str(config.RAW_LOG_PATH)
    assert Path(default).is_absolute()


# --- Integration with the real generated dataset -------------------------------
@needs_generated_data
def test_pipeline_on_generated_dataset():
    result = run_pipeline(config.RAW_LOG_PATH)
    assert result.n_logs == 1000
    assert result.clustered["line_id"].tolist() == list(range(1000))
    assert result.n_clusters > 1
    assert list(result.ranking.columns) == OUTPUT_COLUMNS
    assert result.ranking["rank"].tolist() == list(range(1, result.n_clusters + 1))
    # Every log is counted exactly once across clusters.
    assert result.ranking["size"].sum() == 1000
