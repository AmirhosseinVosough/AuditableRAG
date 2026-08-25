"""Phase 15: regression tests for the agentic orchestrator's *decisions*, not the deterministic math.

Phase 3 (scoping), Phase 6 (the weighted-average arithmetic), and Phase 13's
`is_borderline_quality_failure` already have their own correctness tests
(test_fund_filter.py, test_calculator.py) or are exercised end-to-end,
nothing mocked, by test_pipeline.py. This file is different on purpose: it
tests the REASON -> ACT -> EVALUATE -> RETRY -> (EXCLUDE | ESCALATE) ->
HARD-STOP control flow in `agent_orchestrator.run_agentic_pipeline` -
whether it retries the right number of times, whether it excludes instead
of crashing, whether the hard iteration cap actually stops execution, and
whether a borderline failure lands in the right bucket. None of that
depends on whether the LLM's *judgment* was correct, only on how the
orchestrator *reacts* to a given extraction outcome - so `parse_query` and
`extract_fund_fields` are faked here rather than called for real (contrast
with test_pipeline.py's "nothing mocked" end-to-end style, which is testing
a different thing: that the whole real chain produces the right number).

Faking those two call sites also means every test below runs fast,
deterministically, and without a GROQ_API_KEY - important for a suite meant
to prove a hard cap actually terminates a loop: that has to be provable on
every CI run, not just skipped when no key is configured.

What's real and unmocked: Phase 2 (`extract_pdf_content`) and Phase 3
(`scope_documents`/`parse_fund_metadata`) run against the actual synthetic
PDFs in data/raw_pdfs/ - both are deterministic, free, and fast, so faking
them would only make these tests less honest about what they're checking
for no benefit. Only the two genuinely LLM-judged calls are faked, and
`extract_fund_fields` is faked per-fund: every candidate not named in a
test's `overrides` gets a plausible, always-passing response, so each test
only has to describe the one fund whose behavior is actually under test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch

from src.agentic.agent_orchestrator import MAX_EXCLUDED_FRACTION, MAX_RETRY_ATTEMPTS, run_agentic_pipeline
from src.agentic.query_parser import QuerySpec
from src.extraction.field_extraction import ExtractedFields
from src.fund_filter import parse_fund_metadata


# Matches every other phase's canonical example question - scopes to the
# same 6 real candidates (data/ground_truth.json's qualifying_funds) every
# time, so tests can name a specific fund by its real name.
_QUESTION = "weighted average expense ratio for ESG funds, excluding funds closed in Q3"
_QUERY_SPEC = QuerySpec(
    is_esg=True,
    status="active",
    closed_quarter_exclusions=("Q3",),
    requested_metric="weighted_average_expense_ratio",
)


def _clean_response_for(text: str) -> ExtractedFields:
    """A response that always passes check_extraction_quality, for any real candidate's text.

    Values are plausible-but-arbitrary - these tests assert on the
    orchestrator's *decisions* (trace entries, bucket membership, call
    counts), never on the exact computed final_answer, so there is no need
    to reproduce each fund's real expense_ratio/aum here.
    """
    name = parse_fund_metadata(text).name
    return ExtractedFields(fund_name=name, expense_ratio=0.50, aum=100.0)


def _run_with_overrides(
    overrides: dict[str, Callable[[int], ExtractedFields]],
    run_log_dir: Path,
) -> tuple[dict, dict[str, int]]:
    """Run the orchestrator with parse_query fixed and extract_fund_fields faked per-fund.

    *overrides* maps a real fund name to a callable invoked with that fund's
    1-indexed call number (1 on the first attempt, 2 on the retry, etc.),
    which may return an `ExtractedFields` or raise - whatever the test needs
    that specific fund's Nth attempt to do. Every candidate not named in
    *overrides* gets `_clean_response_for`'s always-passing response.

    A plain `MagicMock()` (no `spec=Groq`) stands in for the Groq client -
    `_LLMCallRecorder` only needs `.chat.completions.create` to be readable
    and reassignable, which any MagicMock supports; since `extract_fund_fields`
    and `parse_query` are both faked below, the client is never actually
    used to make a network call, so it does not need to be Groq-shaped.

    Returns:
        (result, call_counts) - `result` is `run_agentic_pipeline`'s return
        dict; `call_counts` maps each overridden fund's name to how many
        times its callable was actually invoked, for asserting exact retry
        counts without relying on `Mock.call_count` across the whole batch.
    """
    call_counts: dict[str, int] = {}

    def side_effect(text: str, *, client=None, model=None) -> ExtractedFields:
        name = parse_fund_metadata(text).name
        if name in overrides:
            call_counts[name] = call_counts.get(name, 0) + 1
            return overrides[name](call_counts[name])
        return _clean_response_for(text)

    with (
        patch("src.agentic.agent_orchestrator.parse_query", return_value=_QUERY_SPEC),
        patch("src.agentic.agent_orchestrator.extract_fund_fields", side_effect=side_effect),
    ):
        result = run_agentic_pipeline(_QUESTION, client=MagicMock(), run_log_dir=run_log_dir)

    return result, call_counts


def test_clean_run_has_zero_retries(tmp_path: Path) -> None:
    """No fund misbehaves -> every candidate passes on attempt 1, zero RETRY entries, a real answer."""
    result, _call_counts = _run_with_overrides({}, tmp_path)

    assert len(result["included_funds"]) == 6
    assert result["excluded_funds"] == []
    assert result["needs_human_review"] == []
    assert result["hard_stopped"] is False
    assert result["final_answer"] is not None
    assert [e for e in result["decision_trace"] if e["step"] == "RETRY"] == []


def test_transient_failure_is_retried_then_recovers(tmp_path: Path) -> None:
    """A fund whose extraction call fails twice, then succeeds, must end up included - not excluded.

    This is the retry mechanism's actual job: recover from a transient
    failure (a dropped call, a rate limit) instead of giving up on the
    first hiccup.
    """
    target = "Evergreen ESG Equity Fund"

    def flaky(call_number: int) -> ExtractedFields:
        if call_number < 3:
            raise RuntimeError(f"simulated transient network failure (attempt {call_number})")
        return ExtractedFields(fund_name=target, expense_ratio=0.45, aum=120.0)

    result, call_counts = _run_with_overrides({target: flaky}, tmp_path)

    assert call_counts[target] == 3, "expected 2 failures + 1 success - retry did not actually happen"
    included_names = [f["name"] for f in result["included_funds"]]
    excluded_names = [f["name"] for f in result["excluded_funds"]]
    assert target in included_names
    assert target not in excluded_names
    retry_entries = [e for e in result["decision_trace"] if e["step"] == "RETRY" and e["detail"].startswith(target)]
    assert len(retry_entries) == 2


def test_exhausting_retries_excludes_rather_than_crashes(tmp_path: Path) -> None:
    """A fund whose extraction call fails on every attempt must be excluded, not raise out of the pipeline.

    The bar here is as much "run_agentic_pipeline returns normally at all"
    as it is the specific bucket - a persistent per-fund failure must not
    take down the whole batch.
    """
    target = "Evergreen ESG Equity Fund"

    def always_fails(call_number: int) -> ExtractedFields:
        raise RuntimeError(f"simulated persistent failure (attempt {call_number})")

    result, call_counts = _run_with_overrides({target: always_fails}, tmp_path)  # must not raise

    assert call_counts[target] == MAX_RETRY_ATTEMPTS
    excluded = {f["name"]: f["reason"] for f in result["excluded_funds"]}
    assert target in excluded
    assert "every extraction call failed outright" in excluded[target]
    # the other 5 real candidates were never touched by the failure - one
    # bad fund does not take the rest of the batch down with it
    assert len(result["included_funds"]) == 5


def test_iteration_cap_halts_on_a_fund_engineered_to_always_fail_evaluate(tmp_path: Path) -> None:
    """The infinite-loop guard: a fund that can NEVER pass EVALUATE must still stop at exactly the cap.

    Stopping-condition test for Phase 15: `expense_ratio=999.0` is wildly
    outside check_extraction_quality's plausible range on every single
    attempt - nothing about retrying could ever make this fund pass, which
    is exactly the pathological case the hard-coded
    `while attempt < MAX_RETRY_ATTEMPTS` bound exists for. Asserting the
    call count is exactly `MAX_RETRY_ATTEMPTS` (not "at least", not "the
    test finished") is what actually proves the loop terminated at the cap
    rather than running away - if the bound were ever broken, this fund's
    mock would be invoked far more than MAX_RETRY_ATTEMPTS times, or the
    test would hang outright, either of which fails this assertion (or the
    whole test run) loudly instead of silently.
    """
    target = "Evergreen ESG Equity Fund"

    def always_garbage(_call_number: int) -> ExtractedFields:
        return ExtractedFields(fund_name=target, expense_ratio=999.0, aum=120.0)

    result, call_counts = _run_with_overrides({target: always_garbage}, tmp_path)

    assert call_counts[target] == MAX_RETRY_ATTEMPTS, (
        f"expected exactly {MAX_RETRY_ATTEMPTS} calls (the hard cap), got {call_counts[target]} - "
        "the iteration cap did not stop the loop where it should have"
    )
    excluded_names = [f["name"] for f in result["excluded_funds"]]
    assert target in excluded_names
    retry_entries = [e for e in result["decision_trace"] if e["step"] == "RETRY" and e["detail"].startswith(target)]
    assert len(retry_entries) == MAX_RETRY_ATTEMPTS


def test_needs_human_review_is_populated_for_a_borderline_failure(tmp_path: Path) -> None:
    """A fund that fails EVALUATE every time, but only ever by a small margin, must be escalated.

    10.4% is just past check_extraction_quality's 10.0 bound and well
    inside is_borderline_quality_failure's margin - a plausible near-miss,
    not obvious garbage - so it belongs in needs_human_review, never
    silently in excluded_funds or, worse, included_funds.
    """
    target = "Evergreen ESG Equity Fund"

    def always_borderline(_call_number: int) -> ExtractedFields:
        return ExtractedFields(fund_name=target, expense_ratio=10.4, aum=120.0)

    result, call_counts = _run_with_overrides({target: always_borderline}, tmp_path)

    assert call_counts[target] == MAX_RETRY_ATTEMPTS
    review_names = [f["name"] for f in result["needs_human_review"]]
    assert target in review_names
    assert target not in [f["name"] for f in result["excluded_funds"]]
    assert target not in [f["name"] for f in result["included_funds"]]
    escalate_entries = [
        e for e in result["decision_trace"] if e["step"] == "ESCALATE" and e["detail"].startswith(target)
    ]
    assert len(escalate_entries) == 1


def test_hard_stop_when_too_many_funds_are_unresolved(tmp_path: Path) -> None:
    """More than MAX_EXCLUDED_FRACTION of the candidate set failing must hard-stop, not compute a partial answer.

    4 of the 6 real candidates for the canonical question (67%) fail
    persistently, exceeding the 50% threshold - the run must refuse to
    compute an answer over so incomplete a set, mirroring v1's Phase 5 gate.
    """
    failing_targets = [
        "Evergreen ESG Equity Fund",
        "Horizon Sustainable Bond Fund",
        "Climate Transition Leaders Fund",
        "Green Infrastructure Fund",
    ]
    assert len(failing_targets) / 6 > MAX_EXCLUDED_FRACTION  # the test's own premise, checked

    def always_fails(_call_number: int) -> ExtractedFields:
        raise RuntimeError("simulated persistent failure")

    result, _call_counts = _run_with_overrides({name: always_fails for name in failing_targets}, tmp_path)

    assert result["hard_stopped"] is True
    assert result["final_answer"] is None
    assert result["hard_stop_reason"]
    assert len([e for e in result["decision_trace"] if e["step"] == "HARD-STOP"]) == 1
