"""Phase 8: end-to-end test that the full pipeline reproduces data/ground_truth.json.

This is the "v1 done" test: it runs the real Phase 2-6 chain (`pipeline.run_pipeline`)
over the actual synthetic PDFs - including real Phase 4 calls to the Groq API,
nothing mocked - and checks the result against the hand-verified answer in
data/ground_truth.json. If this test passes, every deterministic phase from raw
PDF to final number is proven correct end to end, not just phase-by-phase in
isolation.

Written with plain pytest functions and `assert` (not unittest.TestCase like
test_fund_filter.py/test_calculator.py) per Phase 8's instruction - the two
styles are otherwise equivalent and pytest runs both.
"""

import json
import os
from pathlib import Path

import pytest

from src.fund_filter import FilterSpec
from src.pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, if present.

    Duplicated from src/pipeline.py rather than imported: it's a private
    helper there, and this needs to run at module import time (before
    pytest even evaluates the skipif condition below), not inside a test
    function. Existing environment variables are never overwritten.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))



# Runs at import time, not inside the test, because @pytest.mark.skipif's
# condition is evaluated at collection - if this ran later, a .env-only key
# (nothing exported in the shell) would cause a false skip.
_load_dotenv(PROJECT_ROOT / ".env")


@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY is not set (env or .env) - Phase 4 needs a real Groq call.",
)
def test_pipeline_matches_ground_truth(tmp_path: Path) -> None:
    """A full pipeline run over the synthetic dataset must match ground_truth.json exactly."""
    ground_truth = json.loads((PROJECT_ROOT / "data" / "ground_truth.json").read_text())

    # run_log_dir points at pytest's tmp_path rather than the real
    # outputs/run_logs/: this still exercises Phase 7's real "write the
    # audit record" code path (checked below), it just keeps test-generated
    # log files from piling up next to genuine pipeline runs.
    result = run_pipeline(
        ground_truth["query"],
        FilterSpec(is_esg=True, status="active"),
        run_log_dir=tmp_path,
    )

    # If this fails, the real problem is almost always a Phase 4 extraction
    # failure for one fund (a bad LLM call) - check result.verification_error
    # and the per-fund ERROR lines logged during the run before assuming the
    # deterministic logic broke.
    assert result.verification_error is None, result.verification_error
    assert result.counts_match is True
    assert result.expected_count == result.collected_count == len(ground_truth["qualifying_funds"])

    # Included funds: exact list equality (not just set equality) on
    # purpose. Phase 3 preserves source order (sorted PDF filenames), and
    # ground_truth.json's qualifying_funds happens to be listed in that same
    # order, so an exact match also catches an accidental reordering, not
    # just a wrong membership.
    included_names = [fund.name for fund in result.included_funds]
    assert included_names == ground_truth["qualifying_funds"]

    # Excluded funds: set equality on names only. ground_truth.json's
    # excluded_funds values are hand-written prose explanations (e.g.
    # "inactive (closed in Q4)") for a human reading the fixture, not a
    # second source of truth for pipeline.py's generated reason strings -
    # asserting on exact wording here would make the test brittle to
    # phrasing, not logic.
    excluded_names = {fund.name for fund in result.excluded_funds}
    assert excluded_names == set(ground_truth["excluded_funds"].keys())

    # pytest.approx rather than exact equality: the final number is a chain
    # of float multiplications and one division (Phase 6), so it can differ
    # from ground_truth.json's value in the last decimal place or two even
    # when the computation is correct.
    assert result.final_answer == pytest.approx(ground_truth["correct_answer"])

    # Phase 7's contract is "always write the audit record" - confirm that
    # actually happened, since every assertion above only checks the
    # in-memory PipelineResult, not what Phase 7 is supposed to persist.
    written_logs = list(tmp_path.glob("run_*.json"))
    assert len(written_logs) == 1
