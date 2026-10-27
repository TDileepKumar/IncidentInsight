"""Day 5 - TF-IDF feature extraction for IncidentInsight.

Stage 3 of the pipeline. Converts normalized message text into numeric vectors
so that cosine similarity (Day 6) and agglomerative clustering (Day 6) can
operate on them.

Approved Day 5 decisions:
  * vectorize ALL rows in their original DataFrame order (row i <-> line_id i)
  * unigrams, stop_words=None, L2 normalization
  * return BOTH the sparse matrix and the FITTED vectorizer
  * support transforming unseen messages with an already-fitted vectorizer
  * provide a top-terms inspection helper
  * handle empty input safely (see LIMITATIONS below)

This module is PURE: no DB, no UI, no network, no clustering.

LIMITATIONS (empty / degenerate input):
  * vectorize_messages([]) cannot fit a vocabulary, so it returns an empty
    (0 x 0) sparse matrix and an UNFITTED vectorizer. Calling transform_messages
    on that unfitted vectorizer will raise NotFittedError - the caller must
    guard against empty input before transforming new data.
  * If every message is blank, TF-IDF has no vocabulary either; we return a
    (n x 0) matrix of zeros rather than raising, and the vectorizer stays
    unfitted. No meaningful similarity is possible in that case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.log_normalizer import NORMALIZED_COLUMN

# Single source of truth for vectorizer settings, so Day 6 tuning is one place.
DEFAULT_PARAMS = {
    "lowercase": True,        # <HOST> -> host, tokens are case-insensitive
    "norm": "l2",             # unit-length rows -> cosine == dot product
    "ngram_range": (1, 1),    # unigrams to start; bigrams are a Day-6 lever
    "stop_words": None,       # IDF already suppresses filler; keep log words
}


def build_vectorizer(**overrides) -> TfidfVectorizer:
    """Create a TfidfVectorizer with DEFAULT_PARAMS, applying any overrides."""
    params = {**DEFAULT_PARAMS, **overrides}
    return TfidfVectorizer(**params)


def vectorize_messages(messages, **overrides):
    """Fit TF-IDF on `messages` and return (sparse_matrix, fitted_vectorizer).

    Row order matches the input order exactly - nothing is sorted or dropped.
    See module LIMITATIONS for empty / all-blank input behaviour.
    """
    messages = list(messages)
    vectorizer = build_vectorizer(**overrides)

    if len(messages) == 0:
        return csr_matrix((0, 0)), vectorizer  # unfitted

    try:
        matrix = vectorizer.fit_transform(messages)
    except ValueError as exc:
        # sklearn raises this SPECIFIC ValueError when there is nothing to
        # tokenize (all-blank or punctuation-only input -> empty vocabulary).
        # Treat only that as the documented degenerate case; re-raise any OTHER
        # ValueError (e.g. an invalid ngram_range) so real configuration errors
        # are never silently hidden.
        if "empty vocabulary" not in str(exc).lower():
            raise
        return csr_matrix((len(messages), 0)), vectorizer  # unfitted
    return matrix, vectorizer


def vectorize_dataframe(df: pd.DataFrame, column: str = NORMALIZED_COLUMN, **overrides):
    """Vectorize a DataFrame column IN ROW ORDER. Returns (matrix, vectorizer).

    Alignment contract: matrix row i corresponds to df row i (and its line_id).
    """
    if column not in df.columns:
        raise KeyError(f"column '{column}' not found in DataFrame")
    return vectorize_messages(df[column].tolist(), **overrides)


def transform_messages(vectorizer: TfidfVectorizer, messages):
    """Transform unseen messages with an ALREADY-FITTED vectorizer.

    Unknown tokens are ignored; a message made only of unknown tokens becomes an
    all-zero row. The vectorizer must be fitted (see LIMITATIONS).
    """
    messages = list(messages)
    if len(messages) == 0:
        n_features = len(vectorizer.vocabulary_) if hasattr(vectorizer, "vocabulary_") else 0
        return csr_matrix((0, n_features))
    return vectorizer.transform(messages)


def top_terms(vectorizer: TfidfVectorizer, vector, k: int = 5) -> list[tuple[str, float]]:
    """Return the top-k (term, weight) pairs of one vector, highest first.

    `vector` may be a sparse row (1 x V) or a dense 1-D array. Zero-weight and
    out-of-vocabulary slots are skipped. Useful for dashboards and debugging.
    """
    row = vector.toarray().ravel() if issparse(vector) else np.asarray(vector).ravel()
    names = vectorizer.get_feature_names_out()

    ranked: list[tuple[str, float]] = []
    for idx in np.argsort(row)[::-1]:
        weight = float(row[idx])
        if weight <= 0.0:
            break
        ranked.append((str(names[idx]), weight))
        if len(ranked) >= k:
            break
    return ranked
