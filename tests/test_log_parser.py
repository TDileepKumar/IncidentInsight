"""Day 3 focused tests for the log parser.

Covers: happy-path parsing, the fixture file, malformed / unknown-level
handling, empty input, and (when the generated data is present) that all 1,000
synthetic lines parse and round-trip against ground_truth.csv.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.log_parser import COLUMNS, VALID_LEVELS, parse_file, parse_line, parse_logs

FIXTURE = Path(__file__).parent / "fixtures" / "sample.log"
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# The two big-file tests need the Day 2 output; skip cleanly if it is absent.
needs_generated_data = pytest.mark.skipif(
    not config.RAW_LOG_PATH.exists(), reason="run `python data_generator.py` first"
)


def test_parse_known_line():
    line = "2026-10-04T06:09:02Z FATAL [req-ba53cce8] [DatabaseService] Database connection pool exhausted"
    rec = parse_line(line)
    assert rec["parsed"] is True
    assert rec["timestamp"] == "2026-10-04T06:09:02Z"
    assert rec["level"] == "FATAL"
    assert rec["request_id"] == "req-ba53cce8"
    assert rec["service"] == "DatabaseService"
    assert rec["message"] == "Database connection pool exhausted"


def test_space_separated_timestamp_without_request_id_bracket():
    line = "2026-10-26 10:07:30 ERROR [cache-service] Redis connection refused to cache-01.internal:6379"
    rec = parse_line(line)
    assert rec["parsed"] is True
    assert rec["timestamp"] == "2026-10-26 10:07:30"
    assert rec["level"] == "ERROR"
    assert rec["service"] == "cache-service"
    assert rec["request_id"] is None          # the source line carries no request id
    assert rec["message"] == "Redis connection refused to cache-01.internal:6379"


def test_iso_timestamp_without_request_id_bracket():
    rec = parse_line("2026-10-26T10:07:34Z WARNING [cache-service] Cache miss for key user:42")
    assert rec["parsed"] is True
    assert rec["timestamp"] == "2026-10-26T10:07:34Z"
    assert rec["level"] == "WARNING"
    assert rec["service"] == "cache-service"
    assert rec["request_id"] is None


def test_single_request_id_bracket_is_not_mistaken_for_a_service():
    rec = parse_line("2026-10-26 10:07:30 ERROR [req-533f5a] Redis connection refused")
    assert rec["parsed"] is True
    assert rec["request_id"] == "req-533f5a"
    assert rec["service"] is None


def test_missing_optional_fields_stay_none():
    rec = parse_line("2026-10-26 10:07:30 INFO Request completed in 12ms")
    assert rec["parsed"] is True
    assert rec["timestamp"] == "2026-10-26 10:07:30"
    assert rec["level"] == "INFO"
    assert rec["service"] is None
    assert rec["request_id"] is None
    assert rec["message"] == "Request completed in 12ms"


def test_timestamp_variants_are_kept_verbatim():
    for stamp in (
        "2026-10-26 10:07:30.123",
        "2026-10-26 10:07:30,123",
        "2026-10-26 10:07:30+05:30",
        "2026-10-26T10:07:30.123456Z",
    ):
        rec = parse_line(f"{stamp} ERROR [cache-service] boom")
        assert rec["parsed"] is True
        assert rec["timestamp"] == stamp


def test_space_format_keeps_line_id_alignment_and_parses_neighbours():
    text = (
        "2026-10-26 10:07:30 ERROR [cache-service] Redis connection refused\n"
        "this line is not a valid log entry at all\n"
        "2026-10-26 10:07:34 ERROR [cache-service] Redis connection refused\n"
    )
    df = parse_logs(text)
    assert df["line_id"].tolist() == [0, 1, 2]
    assert df["parsed"].tolist() == [True, False, True]
    assert df.loc[0, "timestamp"] == "2026-10-26 10:07:30"
    assert df.loc[0, "message"] == "Redis connection refused"
    assert df.loc[1, "message"] == "this line is not a valid log entry at all"   # raw text preserved
    assert pd.isna(df.loc[1, "level"])
    assert df.loc[2, "timestamp"] == "2026-10-26 10:07:34"
    assert df.loc[2, "message"] == "Redis connection refused"


def test_message_keeps_brackets_and_colons():
    line = "2026-10-05T04:17:56Z ERROR [req-c952199e] [DatabaseService] connect to db-replica-2.internal:5433 [retry] failed"
    rec = parse_line(line)
    assert rec["parsed"] is True
    assert rec["message"] == "connect to db-replica-2.internal:5433 [retry] failed"


def test_fixture_counts():
    df = parse_file(FIXTURE)
    assert list(df.columns) == COLUMNS
    assert len(df) == 8                 # 6 valid + 1 malformed + 1 unknown level
    assert int(df["parsed"].sum()) == 6
    assert int((~df["parsed"]).sum()) == 2


def test_malformed_line_is_flagged_not_dropped():
    rec = parse_line("this line is not a valid log entry at all")
    assert rec["parsed"] is False
    assert rec["message"] == "this line is not a valid log entry at all"
    assert rec["level"] is None


def test_unknown_level_is_flagged():
    rec = parse_line("2026-10-03T10:00:00Z BANANA [req-abc12345] [TestService] weird level")
    assert rec["parsed"] is False
    assert rec["message"] == "2026-10-03T10:00:00Z BANANA [req-abc12345] [TestService] weird level"


def test_empty_input_returns_empty_frame():
    df = parse_logs("")
    assert len(df) == 0
    assert list(df.columns) == COLUMNS


def test_blank_lines_are_skipped():
    df = parse_logs("\n\n" + "2026-10-04T06:09:02Z ERROR [req-1] [svc] msg" + "\n\n")
    assert len(df) == 1
    assert df.iloc[0]["line_id"] == 0


@needs_generated_data
def test_all_1000_generated_lines_parse():
    df = parse_file(config.RAW_LOG_PATH)
    assert len(df) == config.TARGET_DATASET_SIZE
    assert bool(df["parsed"].all()), "every generated line must parse"
    # Field sanity across the whole file.
    assert set(df["level"].unique()) <= VALID_LEVELS
    assert df["request_id"].str.startswith("req-").all()
    assert df["timestamp"].str.match(ISO_RE).all()


@needs_generated_data
def test_roundtrip_against_ground_truth():
    """Parsed fields must line up with the Day 2 answer key by line_id."""
    parsed = parse_file(config.RAW_LOG_PATH)
    truth = pd.read_csv(config.GROUND_TRUTH_PATH)
    merged = parsed.merge(truth, on="line_id", suffixes=("_p", "_t"))
    assert len(merged) == len(truth)
    assert (merged["timestamp_p"] == merged["timestamp_t"]).all()
    assert (merged["level_p"] == merged["level_t"]).all()
    assert (merged["request_id_p"] == merged["request_id_t"]).all()
    assert (merged["message_p"] == merged["message_t"]).all()
