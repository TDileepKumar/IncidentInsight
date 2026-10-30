"""Command-line entry point for IncidentInsight.

Runs the log-analysis pipeline and prints the highest-ranked incident clusters.

Usage:
    python run.py                          # use the generated synthetic logs
    python run.py path/to/logs.log         # analyze a specific log file
    python run.py --threshold 0.4 --top 15 # tune clustering / how many to show

Exits 0 on success and nonzero on any failure (missing file, unusable input,
invalid threshold). No web server, no dashboard - just run and report.
"""

from __future__ import annotations

import argparse
import sys

from src import config
from src.pipeline import run_pipeline

# Columns shown in the ranked table, in display order.
_DISPLAY_COLUMNS = [
    "rank",
    "cluster_id",
    "size",
    "frequency_score",
    "severity_score",
    "risk_score",
]
_SCORE_COLUMNS = ["frequency_score", "severity_score", "risk_score"]
_MESSAGE_LIMIT = 60


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (exposed for testing)."""
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run the IncidentInsight pipeline and print ranked incident clusters.",
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default=str(config.RAW_LOG_PATH),
        help="Path to the log file to analyze (default: the generated synthetic logs).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.CLUSTER_DISTANCE_THRESHOLD,
        help="Clustering distance threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top-ranked clusters to display (default: %(default)s).",
    )
    return parser


def _format_table(ranking, top: int):
    """Return a printable, rounded copy of the top-ranked clusters."""
    table = ranking.head(top)[_DISPLAY_COLUMNS].copy()
    for col in _SCORE_COLUMNS:
        table[col] = table[col].round(3)
    table["representative_message"] = (
        ranking.head(top)["representative_message"].astype(str).str.slice(0, _MESSAGE_LIMIT)
    )
    return table.to_string(index=False)


def main(argv=None) -> int:
    """Run the pipeline and print results. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    if args.top < 1:
        print("Error: --top must be a positive integer.", file=sys.stderr)
        return 2

    try:
        result = run_pipeline(log_path=args.log_path, distance_threshold=args.threshold)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Processed {result.n_logs} log entries -> {result.n_clusters} clusters "
          f"(threshold={args.threshold})")

    if result.ranking.empty:
        print("No clusters to display.")
        return 0

    print(f"\nTop {min(args.top, len(result.ranking))} incident clusters by risk:")
    print(_format_table(result.ranking, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
