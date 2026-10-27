"""Central configuration for IncidentInsight.

Everything tunable lives here so no other module hardcodes magic numbers.
For Day 2 this holds the synthetic-data generator settings; later days add
clustering thresholds, DB path, and LLM provider settings.
"""

from pathlib import Path

# --- Project paths -----------------------------------------------------------
# PROJECT_ROOT is the repo root (the folder that contains src/, data/, app.py).
# __file__ is src/config.py -> .parent is src/ -> .parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Generator outputs (Day 2)
RAW_LOG_PATH = RAW_DIR / "synthetic_logs.log"
GROUND_TRUTH_PATH = PROCESSED_DIR / "ground_truth.csv"
GENERATION_META_PATH = PROCESSED_DIR / "generation_meta.json"

# --- Synthetic log generator settings (Day 2) --------------------------------
# A fixed seed makes every run produce the exact same data. This is what lets us
# measure clustering quality reproducibly and lets tests be deterministic.
RANDOM_SEED = 42

# ~1,000 lines: big enough for stable TF-IDF/clustering and believable metrics,
# small enough to regenerate and inspect in seconds while we tune the pipeline.
TARGET_DATASET_SIZE = 1000

# Timestamps are spread across this many days starting from LOG_START.
LOG_START = "2026-10-01T00:00:00Z"
LOG_WINDOW_DAYS = 7

# Relative frequency of each incident category. These are weights, not counts:
# the generator scales them to sum to TARGET_DATASET_SIZE. The distribution is
# intentionally skewed (a few common errors, a rare tail) because real logs are
# skewed, and skew is what makes the frequency+severity ranking meaningful.
#
# NOTE the deliberate overlap: db_connection_failure, db_timeout and
# request_timeout share vocabulary ("connection", "timeout", "database"). These
# are the hard cases that prove clustering groups by meaning, not by luck.
CATEGORY_WEIGHTS = {
    "db_connection_failure": 22,
    "request_timeout": 18,
    "db_timeout": 15,
    "auth_failure": 13,
    "null_pointer": 12,
    "validation_error": 11,
    "payment_gateway_error": 6,
    "file_not_found": 3,
}

# --- Clustering settings (Day 6) ---------------------------------------------
# Agglomerative clustering on the TF-IDF vectors. We use a DISTANCE THRESHOLD
# (not a fixed n_clusters) so the number of incident groups is discovered from
# the data instead of hard-coded.
#
# cosine + average is the right pairing for L2-normalized TF-IDF: cosine treats
# two messages as close when they share vocabulary regardless of length, and
# average linkage is robust to the skewed cluster sizes our logs produce.
#
# CLUSTER_DISTANCE_THRESHOLD = 0.5 is an INITIAL EXPERIMENTAL value, NOT a proven
# optimum. Cosine distance runs 0 (identical) to 2 (opposite); 0.5 means "merge
# messages that are at least ~50% similar". Day 7 sweeps this to tune quality.
CLUSTER_DISTANCE_THRESHOLD = 0.5
CLUSTER_METRIC = "cosine"
CLUSTER_LINKAGE = "average"
# Column appended by the clustering stage; original columns are never modified.
CLUSTER_COLUMN = "cluster_id"
