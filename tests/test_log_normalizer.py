"""Day 4 focused tests for the log normalizer."""

from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.log_normalizer import NORMALIZED_COLUMN, normalize_dataframe, normalize_message
from src.log_parser import COLUMNS, parse_file

FIXTURE = Path(__file__).parent / "fixtures" / "sample.log"

needs_generated_data = pytest.mark.skipif(
    not config.RAW_LOG_PATH.exists(), reason="run `python data_generator.py` first"
)


# --- Individual token rules ---------------------------------------------------
def test_number_variation_collapses():
    a = normalize_message("Client request aborted, connection timed out after 47s")
    b = normalize_message("Client request aborted, connection timed out after 56s")
    assert a == b == "Client request aborted, connection timed out after <NUM>s"


def test_host_port_to_single_host():
    got = normalize_message("Database query timed out after 14s on connection to cache-01.internal:3306")
    assert got == "Database query timed out after <NUM>s on connection to <HOST>"


def test_bare_host_to_host():
    got = normalize_message("unable to acquire connection to db-primary.internal")
    assert got == "unable to acquire connection to <HOST>"


def test_file_line_becomes_loc_not_host():
    # Ordering guard: file:line must become <LOC>, NOT be mistaken for host:port.
    got = normalize_message("NullPointerException at PaymentController.java:57 - null reference for user 8377")
    assert "<LOC>" in got
    assert "<HOST>" not in got
    assert got == "NullPointerException at <LOC> - null reference for user <NUM>"


def test_query_id_hex_to_id():
    got = normalize_message("Timeout waiting for database response from cache-01.internal (query_id=3d23bcdd)")
    assert got == "Timeout waiting for database response from <HOST> (query_id=<ID>)"


def test_path_to_path():
    got = normalize_message("Gateway timeout: request to /api/v1/orders did not complete within 119s")
    assert got == "Gateway timeout: request to <PATH> did not complete within <NUM>s"


def test_quoted_strings_to_str():
    got = normalize_message("Invalid input for field 'email': expected an integer, got '12'")
    assert got == "Invalid input for field <STR>: expected an integer, got <STR>"


def test_ip_to_ip():
    got = normalize_message("Unauthorized access attempt to /api/v1/checkout from ip=10.20.30.40 (401)")
    assert got == "Unauthorized access attempt to <PATH> from ip=<IP> (<NUM>)"


def test_username_to_user():
    got = normalize_message("Authentication failed for user user_4821: invalid credentials")
    assert got == "Authentication failed for user <USER>: invalid credentials"


# --- Safety / quality properties ---------------------------------------------
def test_idempotent():
    once = normalize_message("Database query timed out after 14s on connection to cache-01.internal:3306")
    twice = normalize_message(once)
    assert once == twice


def test_empty_and_none_are_safe():
    assert normalize_message("") == ""
    assert normalize_message("   ") == ""
    assert normalize_message(None) == ""


def test_meaningful_terms_preserved():
    got = normalize_message("NullPointerException at OrderService.java:88 - null reference for user 1234")
    # The distinctive error vocabulary must survive; only the numbers/location go.
    assert "NullPointerException" in got
    assert "null reference" in got


def test_anti_over_normalization_keeps_categories_distinct():
    db = normalize_message("Database query timed out after 14s on connection to cache-01.internal:3306")
    req = normalize_message("Upstream request timed out after 14s waiting for response from api-gateway")
    # Both are "timeouts" but must NOT collapse into the same template.
    assert db != req


# --- DataFrame behaviour ------------------------------------------------------
def test_dataframe_adds_column_and_preserves_original():
    parsed = parse_file(FIXTURE)
    original_message = parsed["message"].copy()
    before = parsed[COLUMNS].copy()

    out = normalize_dataframe(parsed)

    assert NORMALIZED_COLUMN in out.columns
    # New column sits immediately after message.
    assert list(out.columns)[list(out.columns).index("message") + 1] == NORMALIZED_COLUMN
    # Original message column and all metadata are byte-for-byte unchanged.
    assert out["message"].equals(original_message)
    assert out[COLUMNS].equals(before)
    # Input frame was not mutated in place.
    assert NORMALIZED_COLUMN not in parsed.columns


def test_unparsed_rows_pass_through():
    parsed = parse_file(FIXTURE)
    out = normalize_dataframe(parsed)
    unparsed = out[~out["parsed"]]
    assert len(unparsed) == 2  # malformed line + unknown BANANA level
    # For unparsed rows the normalized text equals the preserved raw message.
    assert (unparsed[NORMALIZED_COLUMN] == unparsed["message"]).all()


@needs_generated_data
def test_all_1000_normalized_and_aligned_with_ground_truth():
    parsed = parse_file(config.RAW_LOG_PATH)
    out = normalize_dataframe(parsed)

    assert len(out) == config.TARGET_DATASET_SIZE
    assert out[NORMALIZED_COLUMN].notna().all()

    # Alignment with the Day 2 answer key is preserved after normalization.
    truth = pd.read_csv(config.GROUND_TRUTH_PATH)
    merged = out.merge(truth, on="line_id", suffixes=("_p", "_t"))
    assert len(merged) == len(truth)
    assert (merged["message_p"] == merged["message_t"]).all()

    # Normalization must actually collapse variety.
    raw_distinct = out["message"].nunique()
    norm_distinct = out[NORMALIZED_COLUMN].nunique()
    assert norm_distinct < raw_distinct
