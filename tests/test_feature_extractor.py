"""Day 5 focused tests for TF-IDF feature extraction."""

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.feature_extractor import (
    build_vectorizer,
    top_terms,
    transform_messages,
    vectorize_dataframe,
    vectorize_messages,
)
from src.log_normalizer import NORMALIZED_COLUMN, normalize_dataframe
from src.log_parser import parse_file

needs_generated_data = pytest.mark.skipif(
    not config.RAW_LOG_PATH.exists(), reason="run `python data_generator.py` first"
)

# Two real, clearly-distinct templates from our data (NPE vs payment gateway).
NPE = "NullPointerException at <LOC> - null reference for user <NUM>"
PAYMENT = "Payment declined by provider PayPal for order <ID>, code=<NUM>"


def _norms(matrix) -> np.ndarray:
    return np.linalg.norm(matrix.toarray(), axis=1)


# --- Basic contract -----------------------------------------------------------
def test_shape_and_sparse_format():
    docs = [NPE, PAYMENT, "Database query timed out after <NUM>s on connection to <HOST>"]
    matrix, vectorizer = vectorize_messages(docs)
    assert issparse(matrix)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == len(vectorizer.vocabulary_)


def test_returns_fitted_vectorizer():
    _, vectorizer = vectorize_messages([NPE, PAYMENT])
    assert hasattr(vectorizer, "vocabulary_")        # fitted
    assert "nullpointerexception" in vectorizer.vocabulary_


def test_uses_approved_settings():
    v = build_vectorizer()
    assert v.norm == "l2"
    assert v.ngram_range == (1, 1)
    assert v.stop_words is None
    assert v.lowercase is True


def test_placeholders_lose_brackets_in_vocab():
    _, vectorizer = vectorize_messages([NPE, PAYMENT])
    vocab = set(vectorizer.vocabulary_)
    # The default tokenizer strips punctuation, so <LOC> becomes the token 'loc'
    # and the literal bracketed form never enters the vocabulary.
    assert "<loc>" not in vocab
    assert "loc" in vocab
    assert "num" in vocab


# --- Similarity properties ----------------------------------------------------
def test_identical_messages_have_cosine_one():
    docs = [NPE, NPE]  # identical, non-empty
    matrix, _ = vectorize_messages(docs)
    sim = cosine_similarity(matrix)[0, 1]
    assert sim == pytest.approx(1.0, abs=1e-9)
    # Identical text -> identical vectors.
    assert np.allclose(matrix.toarray()[0], matrix.toarray()[1])


def test_selected_cross_category_pair_is_low():
    # A carefully chosen, clearly-distinct pair (not a universal claim about all
    # cross-category pairs - some genuinely overlap, e.g. the timeout families).
    matrix, _ = vectorize_messages([NPE, PAYMENT])
    sim = cosine_similarity(matrix)[0, 1]
    assert sim < 0.3


def test_row_norms_are_unit_l2():
    matrix, _ = vectorize_messages([NPE, PAYMENT, "Authentication failed for user <USER>: invalid credentials"])
    assert np.allclose(_norms(matrix), 1.0, atol=1e-9)


def test_determinism():
    docs = [NPE, PAYMENT, "Database query timed out after <NUM>s on connection to <HOST>"]
    m1, _ = vectorize_messages(docs)
    m2, _ = vectorize_messages(docs)
    assert np.array_equal(m1.toarray(), m2.toarray())


# --- Unseen-message transformation -------------------------------------------
def test_transform_unseen_message_with_fitted_vectorizer():
    train = [NPE, PAYMENT, "Database query timed out after <NUM>s on connection to <HOST>"]
    _, vectorizer = vectorize_messages(train)

    # A new message: some known tokens, one brand-new token ("zebra").
    new = transform_messages(vectorizer, ["NullPointerException zebra at <LOC>"])
    assert new.shape == (1, len(vectorizer.vocabulary_))
    # Unknown token 'zebra' is ignored -> not in the vocabulary.
    assert "zebra" not in vectorizer.vocabulary_
    # The known part still produces a non-zero vector.
    assert new.nnz > 0


def test_transform_all_unknown_message_is_zero_row():
    _, vectorizer = vectorize_messages([NPE, PAYMENT])
    row = transform_messages(vectorizer, ["qqq www eee"])
    assert row.shape[0] == 1
    assert row.nnz == 0  # nothing matched the fitted vocabulary


def test_transform_empty_list_is_safe():
    _, vectorizer = vectorize_messages([NPE, PAYMENT])
    empty = transform_messages(vectorizer, [])
    assert empty.shape == (0, len(vectorizer.vocabulary_))


# --- top_terms helper ---------------------------------------------------------
def test_top_terms_helper():
    matrix, vectorizer = vectorize_messages([NPE, PAYMENT])
    terms = top_terms(vectorizer, matrix[0], k=3)
    assert isinstance(terms, list)
    assert len(terms) <= 3
    assert all(isinstance(t, str) and isinstance(w, float) for t, w in terms)
    # Weights are returned in descending order.
    weights = [w for _, w in terms]
    assert weights == sorted(weights, reverse=True)
    # The distinctive token for this message is among the top terms.
    assert "nullpointerexception" in [t for t, _ in terms]


# --- Empty / degenerate input -------------------------------------------------
def test_empty_input_is_safe():
    matrix, vectorizer = vectorize_messages([])
    assert issparse(matrix)
    assert matrix.shape[0] == 0
    assert not hasattr(vectorizer, "vocabulary_")  # documented: stays unfitted


def test_all_blank_input_is_safe():
    matrix, vectorizer = vectorize_messages(["", "   "])
    assert matrix.shape[0] == 2
    assert matrix.shape[1] == 0                    # no vocabulary
    assert not hasattr(vectorizer, "vocabulary_")  # documented: stays unfitted


def test_punctuation_only_input_is_safe():
    # Non-blank but nothing to tokenize -> empty vocabulary, handled gracefully.
    matrix, vectorizer = vectorize_messages(["***", "---"])
    assert matrix.shape == (2, 0)
    assert not hasattr(vectorizer, "vocabulary_")


def test_unrelated_value_error_is_not_swallowed():
    # A real configuration error (invalid ngram_range) must propagate, not be
    # mistaken for the empty-vocabulary case.
    with pytest.raises(ValueError):
        vectorize_messages([NPE, PAYMENT], ngram_range=(2, 1))


# --- Full-dataset: alignment, norms, IDF sanity -------------------------------
@needs_generated_data
def test_full_1000_rows_aligned_and_normalized():
    df = normalize_dataframe(parse_file(config.RAW_LOG_PATH))
    matrix, vectorizer = vectorize_dataframe(df)

    # Row count and alignment: matrix row i <-> df row i <-> line_id i.
    assert matrix.shape[0] == 1000
    assert len(df) == 1000
    assert list(df["line_id"]) == list(range(1000))
    assert issparse(matrix)

    # Every row is L2-normalized (rows with at least one token).
    norms = _norms(matrix)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-9)

    # Row-order alignment proof: line_id 0's top terms come from its own template.
    row0_template = df.loc[df["line_id"] == 0, NORMALIZED_COLUMN].iloc[0]
    top0 = [t for t, _ in top_terms(vectorizer, matrix[0], k=5)]
    # At least one distinctive token of that template appears in its top terms.
    assert any(tok in row0_template.lower() for tok in top0)


@needs_generated_data
def test_ground_truth_alignment_through_line_id():
    import pandas as pd

    df = normalize_dataframe(parse_file(config.RAW_LOG_PATH))
    matrix, _ = vectorize_dataframe(df)
    truth = pd.read_csv(config.GROUND_TRUTH_PATH)

    # The DataFrame we vectorized still lines up 1:1 with the answer key.
    merged = df.merge(truth, on="line_id", suffixes=("_p", "_t"))
    assert len(merged) == len(truth) == matrix.shape[0]
    assert (merged["message_p"] == merged["message_t"]).all()


def test_idf_downweights_ubiquitous_tokens():
    # Deterministic, self-contained: 'alpha' appears in every document, while
    # 'beta'/'gamma'/'delta' each appear in only one. A token present everywhere
    # carries no discriminating signal, so its IDF must be the lowest.
    docs = ["alpha beta", "alpha gamma", "alpha delta"]
    _, vectorizer = vectorize_messages(docs)
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_

    assert idf[vocab["alpha"]] < idf[vocab["beta"]]
    assert idf[vocab["alpha"]] < idf[vocab["gamma"]]
    # The three single-document tokens are equally rare -> equal IDF.
    assert idf[vocab["beta"]] == idf[vocab["delta"]]
