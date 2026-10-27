"""Day 12 focused tests for the SQLite persistence layer.

Every test uses pytest's `tmp_path`, so the real project database (if any) is
never touched. Tests exercise initialization, saving, retrieval, and the
isolation/safety guarantees of `src/database.py`.
"""

import importlib
import sqlite3

import pytest

from src import database
from src.incident_explainer import explain_incident


def _explanation(**overrides):
    """A valid Day 11-shaped incident explanation mapping."""
    base = {
        "cluster_id": 15,
        "rank": 1,
        "summary": "Cluster 15 (rank 1) matched a 'database connection failure' pattern.",
        "evidence": {
            "representative_message": "Could not connect to database server <HOST>",
            "occurrence_count": 76,
            "severity_score": 1.0,
            "frequency_score": 0.95,
            "risk_score": 0.97,
        },
        "suggested_checks": ["Verify database server availability."],
        "limitations": ["Root cause is not confirmed."],
    }
    base.update(overrides)
    return base


def _db(tmp_path, name="test.db"):
    return tmp_path / name


# --- Initialization -----------------------------------------------------------
def test_initialize_creates_file_and_table(tmp_path):
    db = _db(tmp_path)
    database.initialize_database(db)
    assert db.exists()
    conn = sqlite3.connect(str(db))
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (database.TABLE_NAME,),
        ).fetchone()
    finally:
        conn.close()
    assert exists is not None


def test_initialize_is_idempotent(tmp_path):
    db = _db(tmp_path)
    database.initialize_database(db)
    database.initialize_database(db)  # must not raise
    assert db.exists()


def test_records_survive_repeated_initialization(tmp_path):
    db = _db(tmp_path)
    database.initialize_database(db)
    database.save_incidents([_explanation()], db)
    database.initialize_database(db)  # re-init after data exists
    rows = database.get_incidents(db)
    assert len(rows) == 1


def test_initialize_creates_missing_parent_directory(tmp_path):
    db = tmp_path / "nested" / "dir" / "test.db"
    database.initialize_database(db)
    assert db.exists()


def test_initialize_rejects_bad_path_type_and_blank(tmp_path):
    with pytest.raises(TypeError):
        database.initialize_database(123)
    with pytest.raises(ValueError):
        database.initialize_database("   ")


def test_initialize_directory_path_raises_sqlite_error(tmp_path):
    # A directory cannot be opened as a database file.
    with pytest.raises(sqlite3.Error):
        database.initialize_database(tmp_path)


# --- Saving -------------------------------------------------------------------
def test_save_single_incident_returns_count(tmp_path):
    db = _db(tmp_path)
    assert database.save_incidents([_explanation()], db) == 1
    assert len(database.get_incidents(db)) == 1


def test_save_multiple_incidents(tmp_path):
    db = _db(tmp_path)
    batch = [_explanation(cluster_id=i) for i in range(5)]
    assert database.save_incidents(batch, db) == 5
    assert len(database.get_incidents(db)) == 5


def test_all_fields_roundtrip(tmp_path):
    db = _db(tmp_path)
    original = _explanation()
    database.save_incidents([original], db)
    saved = database.get_incidents(db)[0]

    assert saved["cluster_id"] == original["cluster_id"]
    assert saved["rank"] == original["rank"]
    assert saved["summary"] == original["summary"]
    assert saved["evidence"] == original["evidence"]
    assert saved["suggested_checks"] == original["suggested_checks"]
    assert saved["limitations"] == original["limitations"]
    assert isinstance(saved["id"], int)
    assert isinstance(saved["created_at"], str) and saved["created_at"]


def test_json_fields_roundtrip_types(tmp_path):
    db = _db(tmp_path)
    database.save_incidents([_explanation()], db)
    saved = database.get_incidents(db)[0]
    assert isinstance(saved["evidence"], dict)
    assert isinstance(saved["suggested_checks"], list)
    assert isinstance(saved["limitations"], list)


def test_unicode_and_special_characters_roundtrip(tmp_path):
    db = _db(tmp_path)
    text = "数据库 connection failed 🚀 café — naïve <HOST> & 'quotes'"
    original = _explanation(
        summary=text, evidence={"representative_message": text, "occurrence_count": 3}
    )
    database.save_incidents([original], db)
    saved = database.get_incidents(db)[0]
    assert saved["summary"] == text
    assert saved["evidence"]["representative_message"] == text


def test_empty_collection_saves_nothing_and_creates_no_db(tmp_path):
    db = _db(tmp_path)
    assert database.save_incidents([], db) == 0
    assert not db.exists()          # no fabricated records, no file created
    assert database.get_incidents(db) == []


def test_missing_or_null_cluster_id_rejected(tmp_path):
    db = _db(tmp_path)
    with pytest.raises(ValueError):
        database.save_incidents([_explanation(cluster_id=None)], db)
    missing = _explanation()
    del missing["cluster_id"]
    with pytest.raises(ValueError):
        database.save_incidents([missing], db)


def test_invalid_collection_types_rejected(tmp_path):
    db = _db(tmp_path)
    with pytest.raises(TypeError):
        database.save_incidents("not a collection", db)
    with pytest.raises(TypeError):
        database.save_incidents(_explanation(), db)     # a single mapping
    with pytest.raises(TypeError):
        database.save_incidents(123, db)                # non-iterable
    with pytest.raises(TypeError):
        database.save_incidents(["not a mapping"], db)  # item not a mapping


def test_missing_required_field_rejected(tmp_path):
    db = _db(tmp_path)
    incomplete = _explanation()
    del incomplete["summary"]
    with pytest.raises(ValueError):
        database.save_incidents([incomplete], db)


def test_non_json_serializable_value_rejected(tmp_path):
    db = _db(tmp_path)
    bad = _explanation(evidence={"weird": {1, 2, 3}})  # a set is not JSON-serializable
    with pytest.raises(TypeError):
        database.save_incidents([bad], db)
    assert not db.exists()  # failed before any DB write


def test_failed_batch_leaves_no_partial_commit(tmp_path):
    db = _db(tmp_path)
    good = _explanation(cluster_id=1)
    bad = _explanation(cluster_id=None)  # invalid -> whole batch must fail
    with pytest.raises(ValueError):
        database.save_incidents([good, bad], db)
    assert database.get_incidents(db) == []


def test_repeated_same_cluster_id_creates_separate_records(tmp_path):
    db = _db(tmp_path)
    database.save_incidents([_explanation(cluster_id=7)], db)
    database.save_incidents([_explanation(cluster_id=7)], db)
    rows = database.get_incidents(db)
    assert len(rows) == 2
    assert all(r["cluster_id"] == 7 for r in rows)
    assert rows[0]["id"] != rows[1]["id"]


def test_null_rank_is_preserved_as_null(tmp_path):
    db = _db(tmp_path)
    database.save_incidents([_explanation(rank=None)], db)
    assert database.get_incidents(db)[0]["rank"] is None


# --- Retrieval ----------------------------------------------------------------
def test_newest_first_ordering(tmp_path):
    db = _db(tmp_path)
    database.save_incidents([_explanation(cluster_id=1)], db)  # older
    database.save_incidents([_explanation(cluster_id=2)], db)  # newer
    rows = database.get_incidents(db)
    assert [r["cluster_id"] for r in rows] == [2, 1]


def test_empty_database_returns_empty_list(tmp_path):
    db = _db(tmp_path)
    database.initialize_database(db)
    assert database.get_incidents(db) == []


def test_missing_database_file_returns_empty_and_creates_nothing(tmp_path):
    db = _db(tmp_path)
    assert database.get_incidents(db) == []
    assert not db.exists()  # read-only retrieval never creates a database


def test_missing_table_returns_empty(tmp_path):
    db = _db(tmp_path)
    sqlite3.connect(str(db)).close()  # a valid but empty (table-less) db file
    assert database.get_incidents(db) == []


def test_limit_works(tmp_path):
    db = _db(tmp_path)
    database.save_incidents([_explanation(cluster_id=i) for i in range(5)], db)
    assert len(database.get_incidents(db, limit=2)) == 2


def test_limit_none_returns_all(tmp_path):
    db = _db(tmp_path)
    database.save_incidents([_explanation(cluster_id=i) for i in range(4)], db)
    assert len(database.get_incidents(db, limit=None)) == 4


def test_offset_works(tmp_path):
    db = _db(tmp_path)
    database.save_incidents([_explanation(cluster_id=i) for i in range(5)], db)
    # Newest-first ids are 5,4,3,2,1 -> offset 2 skips the two newest.
    rows = database.get_incidents(db, offset=2)
    assert len(rows) == 3
    assert [r["cluster_id"] for r in rows] == [2, 1, 0]


@pytest.mark.parametrize("bad", [0, -1, True, False, 2.5, "3"])
def test_invalid_limit_rejected(bad, tmp_path):
    db = _db(tmp_path)
    database.initialize_database(db)
    with pytest.raises(ValueError):
        database.get_incidents(db, limit=bad)


@pytest.mark.parametrize("bad", [-1, True, False, 1.5, "0", None])
def test_invalid_offset_rejected(bad, tmp_path):
    db = _db(tmp_path)
    database.initialize_database(db)
    with pytest.raises(ValueError):
        database.get_incidents(db, offset=bad)


# --- Isolation and safety -----------------------------------------------------
def test_sql_like_text_is_stored_as_data_not_executed(tmp_path):
    db = _db(tmp_path)
    payload = "'; DROP TABLE incidents; --"
    database.save_incidents([_explanation(summary=payload)], db)
    rows = database.get_incidents(db)
    assert len(rows) == 1
    assert rows[0]["summary"] == payload
    # The table still exists and accepts more writes.
    database.save_incidents([_explanation(cluster_id=99)], db)
    assert len(database.get_incidents(db)) == 2


def test_connections_are_closed_after_operations(tmp_path):
    db = _db(tmp_path)
    database.initialize_database(db)
    database.save_incidents([_explanation()], db)
    database.get_incidents(db)
    # On Windows an open handle would block deletion; success proves closure.
    db.unlink()
    assert not db.exists()


def test_importing_module_has_no_side_effects(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    importlib.reload(database)  # re-executes module top-level code
    assert list(tmp_path.iterdir()) == []


# --- Integration smoke test (read-only, no project data written) --------------
def test_day11_output_persists_without_modifying_explainer(tmp_path):
    db = _db(tmp_path)
    incident = {
        "cluster_id": 15,
        "rank": 1,
        "size": 76,
        "representative_message": "Could not connect to database server <HOST>: connection refused",
        "severity_score": 1.0,
        "frequency_score": 0.95,
        "risk_score": 0.97,
    }
    explanation = explain_incident(incident)          # real Day 11 output shape
    assert database.save_incidents([explanation], db) == 1

    saved = database.get_incidents(db)[0]
    assert saved["cluster_id"] == explanation["cluster_id"]
    assert saved["summary"] == explanation["summary"]
    assert saved["evidence"] == explanation["evidence"]
    assert saved["suggested_checks"] == explanation["suggested_checks"]
    assert saved["limitations"] == explanation["limitations"]
