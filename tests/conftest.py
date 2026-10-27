"""Shared pytest configuration for the IncidentInsight test suite.

The Day 13 explanation cache in `src/llm_explainer.py` is module-level state that
survives between tests in one process. It is emptied before and after every test
so a successful mocked explanation from one test can never be replayed in another
(which would silently skip the HTTP call the other test is asserting on). This is
test isolation only - no assertion is relaxed and no test is skipped.
"""

import pytest

from src import llm_explainer


@pytest.fixture(autouse=True)
def _empty_llm_cache():
    llm_explainer.clear_llm_cache()
    yield
    llm_explainer.clear_llm_cache()
