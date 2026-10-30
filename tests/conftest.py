"""Shared pytest configuration for the IncidentInsight test suite.

The explanation cache in `src/llm_explainer.py` is module-level state that
survives between tests in one process. It is emptied before and after every test
so a successful mocked explanation from one test can never be replayed in another
(which would silently skip the HTTP call the other test is asserting on). This is
test isolation only - no assertion is relaxed and no test is skipped.

The two dataset gates below are defined here once and imported by every test
module that needs them. Their skip reasons point at the preserved baseline rather
than at the generator: the canonical files hold a historical dataset that the
current generator cannot reproduce, so regenerating is never the way to provision
them.
"""

import pytest

from src import config, llm_explainer

# Repo-relative forms, so a skip reason never prints a machine-specific path.
_RAW = config.RAW_LOG_PATH.relative_to(config.PROJECT_ROOT).as_posix()
_TRUTH = config.GROUND_TRUTH_PATH.relative_to(config.PROJECT_ROOT).as_posix()
_BASELINE = "data/baseline/7126c46"
_HOWTO = "see README, 'Using the preserved baseline as input'"

needs_generated_data = pytest.mark.skipif(
    not config.RAW_LOG_PATH.exists(),
    reason=f"{_RAW} is absent; copy {_BASELINE}/synthetic_logs.log to that path ({_HOWTO})",
)

# Group B tests read the answer key as well as the raw log, so they need BOTH.
# Gating on the raw log alone made them fail instead of skip when only it existed.
needs_ground_truth = pytest.mark.skipif(
    not (config.RAW_LOG_PATH.exists() and config.GROUND_TRUTH_PATH.exists()),
    reason=(
        f"{_RAW} and {_TRUTH} are both required and one or both is absent; copy "
        f"{_BASELINE}/synthetic_logs.log and {_BASELINE}/ground_truth.csv to those "
        f"paths ({_HOWTO})"
    ),
)


@pytest.fixture(autouse=True)
def _empty_llm_cache():
    llm_explainer.clear_llm_cache()
    yield
    llm_explainer.clear_llm_cache()
