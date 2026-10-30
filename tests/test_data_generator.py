"""Focused tests for the synthetic log generator.

These verify the generator only (reproducibility, mapping, no label leakage,
skew). The other tests/ modules cover the rest of the pipeline.
"""

import json

import pandas as pd

from data_generator import generate_dataset


def test_reproducible_with_same_seed(tmp_path):
    """Same seed -> byte-identical outputs; different seed -> different data."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"

    df_a = generate_dataset(seed=42, size=200, base_dir=a)
    df_b = generate_dataset(seed=42, size=200, base_dir=b)
    df_c = generate_dataset(seed=99, size=200, base_dir=c)

    # Same seed produces identical raw logs and CSVs.
    assert (a / "raw" / "synthetic_logs.log").read_text() == (b / "raw" / "synthetic_logs.log").read_text()
    assert df_a.equals(df_b)
    # A different seed produces different data.
    assert not df_a.equals(df_c)


def test_outputs_exist_and_counts_match(tmp_path):
    df = generate_dataset(seed=42, size=200, base_dir=tmp_path)

    log_path = tmp_path / "raw" / "synthetic_logs.log"
    csv_path = tmp_path / "processed" / "ground_truth.csv"
    meta_path = tmp_path / "processed" / "generation_meta.json"
    assert log_path.exists() and csv_path.exists() and meta_path.exists()

    log_lines = log_path.read_text().strip().splitlines()
    assert len(log_lines) == 200
    assert len(df) == 200

    meta = json.loads(meta_path.read_text())
    assert sum(meta["category_counts"].values()) == 200


def test_line_to_row_mapping_preserved_after_shuffle(tmp_path):
    """Raw log line N must correspond exactly to CSV row with line_id N."""
    generate_dataset(seed=42, size=200, base_dir=tmp_path)
    log_lines = (tmp_path / "raw" / "synthetic_logs.log").read_text().strip().splitlines()
    df = pd.read_csv(tmp_path / "processed" / "ground_truth.csv")

    for _, row in df.iterrows():
        line = log_lines[int(row["line_id"])]
        # The raw line must contain this row's timestamp, level, request id and message.
        assert line.startswith(f"{row['timestamp']} {row['level']} [{row['request_id']}]")
        assert line.endswith(row["message"])


def test_no_label_leakage_into_messages(tmp_path):
    """The category label must never appear inside the raw log message."""
    df = generate_dataset(seed=42, size=200, base_dir=tmp_path)
    for _, row in df.iterrows():
        assert row["category"] not in row["message"].lower()
        assert row["category"] not in row["message"]


def test_all_categories_present_and_skewed(tmp_path):
    df = generate_dataset(seed=42, size=1000, base_dir=tmp_path)
    counts = df["category"].value_counts()

    # All eight categories appear.
    assert len(counts) == 8
    # Distribution is skewed: the most common category far exceeds the rarest.
    assert counts.max() > 5 * counts.min()
