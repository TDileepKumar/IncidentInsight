"""Day 2 - Synthetic log generator for IncidentInsight.

Produces realistic-looking application error logs PLUS a ground-truth "answer
key" so we can later measure whether clustering groups the right things.

Two output streams, kept deliberately separate:
  * data/raw/synthetic_logs.log   -> what the pipeline will ingest (NO labels)
  * data/processed/ground_truth.csv -> the answer key (line_id -> category)

The label is never written into the log message, otherwise clustering could
"cheat" by reading the answer, and our evaluation would be meaningless.

Run:  python data_generator.py
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src import config

# --- Value pools for the variable tokens -------------------------------------
# These are the "noise" fields the Day-4 normalizer will later strip out and
# replace with placeholders like <ID>, <NUM>, <LOC>, <HOST>.
HOSTS = ["db-primary.internal", "db-replica-2.internal", "cache-01.internal"]
PORTS = [5432, 5433, 3306]
SERVICES_GENERIC = ["api-gateway", "web-frontend"]
PATHS = ["/api/v1/orders", "/api/v1/checkout", "/health", "/api/v1/users/profile"]
USERS = ["jdoe", "alice", "user_4821", "admin", "m.chen"]
CLASSES = ["OrderService", "PaymentController", "CartHandler", "UserSession"]
PROVIDERS = ["Stripe", "PayPal", "Adyen"]
FIELDS = ["email", "username", "card_number", "shipping_zip"]
EXPECTED = ["an integer", "email format", "16 digits", "a non-empty string"]
JUNK = ["abc", "12", "not-an-email", "", "99999999999999999999"]
FILENAMES = ["img_2931.png", "invoice_0042.pdf", "avatar_77.jpg", "report_q3.csv"]

HEX = "0123456789abcdef"


def _hex(rng: random.Random, length: int) -> str:
    """Return a random hex string of the given length (used for IDs)."""
    return "".join(rng.choice(HEX) for _ in range(length))


def _random_values(rng: random.Random) -> dict:
    """Build a fresh dict of random token values for one log message.

    A template can reference any subset of these keys via str.format().
    Generating them all each time keeps templates simple and readable.
    """
    return {
        "host": rng.choice(HOSTS),
        "port": rng.choice(PORTS),
        "svc": rng.choice(SERVICES_GENERIC),
        "path": rng.choice(PATHS),
        "user": rng.choice(USERS),
        "uid": rng.randint(1000, 9999),
        "cls": rng.choice(CLASSES),
        "line": rng.randint(40, 320),
        "txn": _hex(rng, 10),
        "order": _hex(rng, 8),
        "prov": rng.choice(PROVIDERS),
        "field": rng.choice(FIELDS),
        "exp": rng.choice(EXPECTED),
        "val": rng.choice(JUNK),
        "ip": f"{rng.randint(10, 250)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
        "qid": _hex(rng, 8),
        "n": rng.randint(5, 120),   # timeouts in seconds
        "n2": rng.randint(1, 5),     # small counts like retries
        "fname": rng.choice(FILENAMES),
    }


# --- Category specifications -------------------------------------------------
# Each category defines: severity mix, plausible service names, and 2-3 message
# variations. Multiple variations per category force clustering to group by
# meaning rather than by an exact string match.
CATEGORY_SPECS = {
    "db_connection_failure": {
        "levels": [("ERROR", 0.8), ("FATAL", 0.2)],
        "services": ["DatabaseService", "PostgresClient", "ConnectionPool"],
        "templates": [
            "Could not connect to database server {host}:{port}: connection refused",
            "Database connection pool exhausted, unable to acquire connection to {host}",
            "Failed to establish connection to primary database {host}:{port} after {n2} retries",
        ],
    },
    "db_timeout": {
        "levels": [("ERROR", 0.9), ("WARNING", 0.1)],
        "services": ["DatabaseService", "QueryExecutor"],
        "templates": [
            "Database query timed out after {n}s on connection to {host}:{port}",
            "Timeout waiting for database response from {host} (query_id={qid})",
            "Slow query exceeded {n}s threshold on {host}, aborting connection",
        ],
    },
    "request_timeout": {
        "levels": [("ERROR", 0.7), ("WARNING", 0.3)],
        "services": ["api-gateway", "HttpClient", "web-frontend"],
        "templates": [
            "Upstream request timed out after {n}s waiting for response from {svc}",
            "Gateway timeout: request to {path} did not complete within {n}s",
            "Client request aborted, connection timed out after {n}s",
        ],
    },
    "auth_failure": {
        "levels": [("WARNING", 0.6), ("ERROR", 0.4)],
        "services": ["AuthService", "TokenValidator"],
        "templates": [
            "Authentication failed for user {user}: invalid credentials",
            "Unauthorized access attempt to {path} from ip={ip} (401)",
            "Token validation failed for user {user}: expired or malformed JWT",
        ],
    },
    "null_pointer": {
        "levels": [("ERROR", 1.0)],
        "services": ["OrderService", "PaymentController", "CartHandler"],
        "templates": [
            "NullPointerException at {cls}.java:{line} - null reference for user {uid}",
            "Unhandled null pointer: cannot invoke method on null object in {cls}",
            "NullPointerException in {cls} handler at line {line}",
        ],
    },
    "validation_error": {
        "levels": [("WARNING", 1.0)],
        "services": ["RequestValidator", "web-frontend"],
        "templates": [
            "Invalid input for field '{field}': expected {exp}, got '{val}'",
            "Request validation failed: field '{field}' is required",
            "Malformed payload for field '{field}' (user {uid})",
        ],
    },
    "payment_gateway_error": {
        "levels": [("FATAL", 0.5), ("ERROR", 0.5)],
        "services": ["PaymentController", "BillingService"],
        "templates": [
            "Payment gateway returned 502 for transaction {txn}, provider unavailable",
            "Failed to process payment {txn}: gateway timeout from provider {prov}",
            "Payment declined by provider {prov} for order {order}, code=502",
        ],
    },
    "file_not_found": {
        "levels": [("ERROR", 0.7), ("WARNING", 0.3)],
        "services": ["StaticAssets", "FileStore"],
        "templates": [
            "FileNotFoundError: no such file or directory: /uploads/{fname}",
            "Missing resource: could not read /uploads/{fname}",
            "Static asset not found at /uploads/{fname} (404)",
        ],
    },
}


def _choose_level(rng: random.Random, levels: list[tuple[str, float]]) -> str:
    """Pick a severity level according to its weight (e.g. ERROR 80% of time)."""
    r = rng.random()
    cumulative = 0.0
    for level, weight in levels:
        cumulative += weight
        if r <= cumulative:
            return level
    return levels[-1][0]


def _allocate_counts(weights: dict[str, int], total: int) -> dict[str, int]:
    """Scale category weights so the counts sum exactly to `total`.

    Uses the largest-remainder method: give each category its integer share,
    then hand the leftover units to whichever categories had the biggest
    fractional parts. This guarantees the dataset is exactly `total` lines for
    any size, not just when the weights happen to divide evenly.
    """
    weight_sum = sum(weights.values())
    exact = {cat: total * w / weight_sum for cat, w in weights.items()}
    counts = {cat: int(value) for cat, value in exact.items()}

    remainder = total - sum(counts.values())
    # Order categories by the size of their fractional part, largest first.
    by_fraction = sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)
    i = 0
    while remainder > 0:
        counts[by_fraction[i % len(by_fraction)][0]] += 1
        remainder -= 1
        i += 1
    return counts


def _make_timestamp(rng: random.Random, start: datetime, window_seconds: int) -> str:
    """Return an ISO-8601 UTC timestamp somewhere inside the log window."""
    offset = rng.randint(0, window_seconds)
    return (start + timedelta(seconds=offset)).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_records(rng: random.Random, size: int) -> list[dict]:
    """Create `size` log records across all categories, then shuffle them.

    Returns a list of dicts. The shuffle interleaves categories so the log does
    not come pre-sorted by type (which would make clustering trivially easy).
    line_id is assigned AFTER shuffling so the log file and CSV always agree.
    """
    counts = _allocate_counts(config.CATEGORY_WEIGHTS, size)
    start = datetime.strptime(config.LOG_START, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    window_seconds = config.LOG_WINDOW_DAYS * 24 * 3600

    records: list[dict] = []
    for category, count in counts.items():
        spec = CATEGORY_SPECS[category]
        for _ in range(count):
            values = _random_values(rng)
            records.append(
                {
                    "timestamp": _make_timestamp(rng, start, window_seconds),
                    "level": _choose_level(rng, spec["levels"]),
                    "service": rng.choice(spec["services"]),
                    "request_id": "req-" + _hex(rng, 8),
                    "category": category,
                    "message": rng.choice(spec["templates"]).format(**values),
                }
            )

    rng.shuffle(records)
    for line_id, record in enumerate(records):
        record["line_id"] = line_id
    return records


def _format_log_line(record: dict) -> str:
    """Render one record as a raw log line. Contains NO category label."""
    return (
        f"{record['timestamp']} {record['level']} "
        f"[{record['request_id']}] [{record['service']}] {record['message']}"
    )


def generate_dataset(
    seed: int = config.RANDOM_SEED,
    size: int = config.TARGET_DATASET_SIZE,
    base_dir: "str | Path | None" = None,
) -> pd.DataFrame:
    """Generate the full dataset, write the three output files, return the CSV data.

    base_dir lets tests write into a temporary folder instead of the real data/.
    """
    rng = random.Random(seed)
    records = generate_records(rng, size)

    base = config.DATA_DIR if base_dir is None else Path(base_dir)
    raw_dir = base / "raw"
    processed_dir = base / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1) Raw log file (ingestion input) - no labels.
    log_lines = [_format_log_line(r) for r in records]
    (raw_dir / "synthetic_logs.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # 2) Ground-truth CSV (answer key).
    df = pd.DataFrame(
        records,
        columns=["line_id", "timestamp", "level", "service", "request_id", "category", "message"],
    )
    df.to_csv(processed_dir / "ground_truth.csv", index=False)

    # 3) Generation metadata (for reproducibility / documentation).
    category_counts = df["category"].value_counts().to_dict()
    level_counts = df["level"].value_counts().to_dict()
    meta = {
        "seed": seed,
        "size": size,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "log_start": config.LOG_START,
        "log_window_days": config.LOG_WINDOW_DAYS,
        "category_counts": category_counts,
        "level_counts": level_counts,
        "note": "Synthetic data. ground_truth.csv is the answer key for evaluating clustering; "
        "labels are intentionally absent from synthetic_logs.log.",
    }
    (processed_dir / "generation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return df


def main() -> None:
    df = generate_dataset()
    print(f"Generated {len(df)} log lines (seed={config.RANDOM_SEED}).")
    print(f"  Raw log:      {config.RAW_LOG_PATH}")
    print(f"  Ground truth: {config.GROUND_TRUTH_PATH}")
    print(f"  Metadata:     {config.GENERATION_META_PATH}")
    print("\nCategory distribution:")
    for category, count in df["category"].value_counts().items():
        print(f"  {category:<24} {count}")


if __name__ == "__main__":
    main()
