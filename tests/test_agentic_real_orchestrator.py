"""Phase 16: regression tests for the real-data agentic loop's *decisions* - same style as Phase 15.

Mirrors `test_agentic_orchestrator.py` exactly: `real_classifier.classify_esg_status` and
`cascade.extraction_cascade.extract_with_cascade` are faked (per-document, via a `side_effect`
keyed to the PDF's filename/stem - real-data candidate identity, per Phase 16's design), plus
`parse_query`, for the same reason Phase 15's tests faked `extract_fund_fields`/`parse_query`: these
tests are about how `_run_agentic_real` *reacts* to a given classify/extract outcome, not whether
the LLM's judgment was correct. Fast, deterministic, no `GROQ_API_KEY` required.

What's real and unmocked: `real_data_loader.load_real_pdfs`, `extract_pdf_content`, and
`real_classifier.filter_real_funds` all run against the actual fixtures in `data/user_uploads/` - all
three are deterministic and free, so faking them would only make these tests less honest for no
benefit. That directory holds 6 files, one of them (`corrupted_fact_sheet.pdf`, 71 bytes) genuinely
broken - every test here sees it pre-excluded at the `extract_pdf_content` step, before
`classify_esg_status` is ever called for it, so only 5 documents reach the classify stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch

from src.agentic.agent_orchestrator import (
    MAX_EXCLUDED_FRACTION,
    MAX_REAL_INFRA_RETRY_ATTEMPTS,
    run_agentic_pipeline,
)
from src.agentic.query_parser import QuerySpec
from src.agentic.real_classifier import RealFundMetadata
from src.cascade.extraction_cascade import CascadeResult, FieldResolution
from src.extraction.field_extraction import RealExtractedFields
from src.shared.source_location import SourceLocation


_QUESTION = "weighted average expense ratio for ESG active funds"
_QUERY_SPEC = QuerySpec(
    is_esg=True,
    status="active",
    closed_quarter_exclusions=(),
    requested_metric="weighted_average_expense_ratio",
)

# The 5 real (non-corrupted) fixtures in data/user_uploads/, by filename stem -
# every test's default ("not under test") behavior is defined for these.
_REAL_STEMS = ("esgu_fact_sheet", "esgv_fact_sheet", "ivv_fact_sheet", "spy_fact_sheet", "ussg_fact_sheet")


def _clean_metadata_for(identity: str) -> RealFundMetadata:
    """A classification that always survives filter_real_funds's is_esg=True/status='active' check."""
    return RealFundMetadata(name=identity, is_esg=True, status="active", flags=())


def _clean_cascade_result_for(identity: str) -> CascadeResult:
    """A cascade result that always passes EVALUATE (needs_human_review=False)."""
    fields = RealExtractedFields(
        fund_name=f"{identity} Fund", is_esg=True, status="active", expense_ratio=0.50, aum=100.0, flags=()
    )
    resolutions = {
        "expense_ratio": FieldResolution(value=0.50, resolved_by="llm", source=None),
        "aum": FieldResolution(value=100.0, resolved_by="llm", source=None),
    }
    return CascadeResult(fields=fields, resolutions=resolutions, needs_human_review=False, review_reasons=())


def _run_real_with_overrides(
    classify_overrides: dict[str, Callable[[int], RealFundMetadata]],
    cascade_overrides: dict[str, Callable[[int], CascadeResult]],
    run_log_dir: Path,
) -> tuple[dict, dict[str, int], dict[str, int]]:
    """Run `_run_agentic_real` (via `run_agentic_pipeline(..., source="real")`) with both LLM-boundary calls faked.

    *classify_overrides*/*cascade_overrides* map a filename stem to a
    callable invoked with that document's 1-indexed call number, which may
    return a result or raise. Every stem not named in either override dict
    gets the default always-passing response.

    Real, unmocked: `load_real_pdfs`, `extract_pdf_content`,
    `filter_real_funds` - all run against `data/user_uploads/`'s actual files.

    Returns:
        (result, classify_call_counts, cascade_call_counts) - the latter two
        map each overridden stem to how many times its callable actually ran.
    """
    classify_call_counts: dict[str, int] = {}
    cascade_call_counts: dict[str, int] = {}

    def classify_side_effect(identity: str, pages: list[str], question: str, *, client=None) -> RealFundMetadata:
        if identity in classify_overrides:
            classify_call_counts[identity] = classify_call_counts.get(identity, 0) + 1
            return classify_overrides[identity](classify_call_counts[identity])
        return _clean_metadata_for(identity)

    def cascade_side_effect(pdf_path: Path, question: str, *, client=None, model=None) -> CascadeResult:
        identity = pdf_path.stem
        if identity in cascade_overrides:
            cascade_call_counts[identity] = cascade_call_counts.get(identity, 0) + 1
            return cascade_overrides[identity](cascade_call_counts[identity])
        return _clean_cascade_result_for(identity)

    with (
        patch("src.agentic.agent_orchestrator.parse_query", return_value=_QUERY_SPEC),
        patch("src.agentic.agent_orchestrator.classify_esg_status", side_effect=classify_side_effect),
        patch("src.agentic.agent_orchestrator.extract_with_cascade", side_effect=cascade_side_effect),
    ):
        result = run_agentic_pipeline(_QUESTION, source="real", client=MagicMock(), run_log_dir=run_log_dir)

    return result, classify_call_counts, cascade_call_counts


def test_clean_real_run_has_zero_retries(tmp_path: Path) -> None:
    """No document misbehaves -> every real fund classifies+extracts cleanly, corrupted fixture excluded."""
    result, _classify_counts, _cascade_counts = _run_real_with_overrides({}, {}, tmp_path)

    assert len(result["included_funds"]) == 5
    assert result["needs_human_review"] == []
    assert result["hard_stopped"] is False
    assert result["final_answer"] is not None
    assert [e for e in result["decision_trace"] if e["step"] == "RETRY"] == []

    excluded_names = [f["name"] for f in result["excluded_funds"]]
    assert excluded_names == ["corrupted_fact_sheet"]
    assert "could not be read" in result["excluded_funds"][0]["reason"]


def test_cheap_stage_exclusion_never_reaches_the_expensive_cascade(tmp_path: Path) -> None:
    """A document classified as not-ESG must never trigger extract_with_cascade at all."""
    target = "spy_fact_sheet"

    def not_esg(_call_number: int) -> RealFundMetadata:
        return RealFundMetadata(name=target, is_esg=False, status="active", flags=())

    result, classify_counts, cascade_counts = _run_real_with_overrides({target: not_esg}, {}, tmp_path)

    assert classify_counts[target] == 1
    assert target not in cascade_counts  # the expensive cascade was never called for this document
    excluded_names = [f["name"] for f in result["excluded_funds"]]
    assert target in excluded_names
    assert len(result["included_funds"]) == 4


def test_classify_transient_failure_recovers_within_infra_retry_cap(tmp_path: Path) -> None:
    """A classify_esg_status call that fails once, then succeeds, must still let the document become a candidate."""
    target = "spy_fact_sheet"

    def flaky(call_number: int) -> RealFundMetadata:
        if call_number < 2:
            raise RuntimeError(f"simulated transient failure (attempt {call_number})")
        return _clean_metadata_for(target)

    result, classify_counts, _cascade_counts = _run_real_with_overrides({target: flaky}, {}, tmp_path)

    assert classify_counts[target] == 2
    included_names = [f["name"] for f in result["included_funds"]]
    assert target in included_names
    retry_entries = [
        e for e in result["decision_trace"] if e["step"] == "RETRY" and e["detail"].startswith(target)
    ]
    assert len(retry_entries) == 1


def test_classify_persistent_failure_excludes_without_crashing_the_batch(tmp_path: Path) -> None:
    """A classify_esg_status call that always fails must exclude that one document, not crash the run."""
    target = "spy_fact_sheet"

    def always_fails(_call_number: int) -> RealFundMetadata:
        raise RuntimeError("simulated persistent failure")

    result, classify_counts, _cascade_counts = _run_real_with_overrides({target: always_fails}, {}, tmp_path)

    assert classify_counts[target] == MAX_REAL_INFRA_RETRY_ATTEMPTS
    excluded = {f["name"]: f["reason"] for f in result["excluded_funds"]}
    assert target in excluded
    assert "couldn't determine ESG/active status" in excluded[target]
    assert len(result["included_funds"]) == 4  # the other 4 real, uncorrupted documents unaffected


def test_cascade_transient_failure_recovers_within_infra_retry_cap(tmp_path: Path) -> None:
    """An extract_with_cascade call that fails once, then succeeds, must still include the fund."""
    target = "spy_fact_sheet"

    def flaky(call_number: int) -> CascadeResult:
        if call_number < 2:
            raise RuntimeError(f"simulated transient failure (attempt {call_number})")
        return _clean_cascade_result_for(target)

    result, _classify_counts, cascade_counts = _run_real_with_overrides({}, {target: flaky}, tmp_path)

    assert cascade_counts[target] == 2
    included_names = [f["name"] for f in result["included_funds"]]
    assert target in included_names
    retry_entries = [
        e for e in result["decision_trace"] if e["step"] == "RETRY" and e["detail"].startswith(target)
    ]
    assert len(retry_entries) == 1


def test_cascade_persistent_failure_excludes_without_crashing_the_batch(tmp_path: Path) -> None:
    """An extract_with_cascade call that always fails must exclude that fund after exactly the infra-retry cap."""
    target = "spy_fact_sheet"

    def always_fails(_call_number: int) -> CascadeResult:
        raise RuntimeError("simulated persistent failure")

    result, _classify_counts, cascade_counts = _run_real_with_overrides({}, {target: always_fails}, tmp_path)

    assert cascade_counts[target] == MAX_REAL_INFRA_RETRY_ATTEMPTS
    excluded = {f["name"]: f["reason"] for f in result["excluded_funds"]}
    assert target in excluded
    assert "extract_with_cascade failed outright" in excluded[target]


def test_needs_human_review_true_escalates_without_a_wasted_retry(tmp_path: Path) -> None:
    """A clean-but-uncertain cascade result (needs_human_review=True) must escalate immediately - no retry.

    This is Phase 16's central design point: unlike a genuine exception, a
    clean result the cascade itself flagged uncertain is never retried,
    since the cascade already exhausted its own internal fallback tiers -
    retrying would almost certainly reproduce the identical answer.
    """
    target = "spy_fact_sheet"

    def uncertain(_call_number: int) -> CascadeResult:
        fields = RealExtractedFields(
            fund_name=None, is_esg=True, status="active", expense_ratio=None, aum=100.0, flags=("expense_ratio: not found",)
        )
        resolutions = {
            "aum": FieldResolution(
                value=100.0, resolved_by="llm", source=SourceLocation(file="spy_fact_sheet.pdf", page=2, snippet="Net Assets: $100M")
            ),
        }
        return CascadeResult(fields=fields, resolutions=resolutions, needs_human_review=True, review_reasons=("expense_ratio: not found by any tier",))

    result, _classify_counts, cascade_counts = _run_real_with_overrides({}, {target: uncertain}, tmp_path)

    assert cascade_counts[target] == 1  # no retry on a clean-but-uncertain result
    review_names = [f["name"] for f in result["needs_human_review"]]
    assert target in review_names
    assert target not in [f["name"] for f in result["excluded_funds"]]
    assert target not in [f["name"] for f in result["included_funds"]]

    review_entry = next(f for f in result["needs_human_review"] if f["name"] == target)
    assert "expense_ratio: not found by any tier" in review_entry["reason"]
    assert "aum" in review_entry["citations"]
    assert review_entry["citations"]["aum"]["page"] == 2

    escalate_entries = [
        e for e in result["decision_trace"] if e["step"] == "ESCALATE" and e["detail"].startswith(target)
    ]
    assert len(escalate_entries) == 1


def test_hard_stop_excludes_cheap_stage_from_the_denominator(tmp_path: Path) -> None:
    """The hard-stop fraction must count only post-scope failures, not cheap-stage exclusions.

    3 of the 5 real documents are excluded at the cheap classify stage
    (not-ESG) - a large fraction of the *total* corpus, but they never
    became candidates, so they must not count toward the hard-stop
    denominator/numerator at all. Only 2 documents actually reach ACT-extract
    as candidates, and both are forced to fail there - 2/2 = 100%, which
    correctly exceeds MAX_EXCLUDED_FRACTION and hard-stops; if the cheap
    exclusions were wrongly folded into the fraction, this would either
    hard-stop for the wrong reason or fail to hard-stop at all.
    """
    not_esg_stems = ["esgv_fact_sheet", "ivv_fact_sheet", "ussg_fact_sheet"]
    failing_candidates = ["esgu_fact_sheet", "spy_fact_sheet"]
    candidate_count = len(_REAL_STEMS) - len(not_esg_stems)  # 5 real docs minus the 3 not-ESG -> 2 candidates
    assert len(failing_candidates) / candidate_count > MAX_EXCLUDED_FRACTION  # the test's own premise, checked

    def not_esg(identity: str) -> Callable[[int], RealFundMetadata]:
        return lambda _call_number: RealFundMetadata(name=identity, is_esg=False, status="active", flags=())

    def always_fails(_call_number: int) -> CascadeResult:
        raise RuntimeError("simulated persistent failure")

    classify_overrides = {stem: not_esg(stem) for stem in not_esg_stems}
    cascade_overrides = {stem: always_fails for stem in failing_candidates}

    result, _classify_counts, _cascade_counts = _run_real_with_overrides(
        classify_overrides, cascade_overrides, tmp_path
    )

    assert result["hard_stopped"] is True
    assert result["final_answer"] is None
    assert result["hard_stop_reason"]
    # the stated fraction must reflect 2 candidates, not 5 documents
    assert "2/2" in result["hard_stop_reason"]
    assert len([e for e in result["decision_trace"] if e["step"] == "HARD-STOP"]) == 1
    # all 3 not-ESG documents still land in excluded_funds (reported, per
    # Phase 16's design), just not counted toward the hard-stop fraction
    excluded_names = {f["name"] for f in result["excluded_funds"]}
    assert set(not_esg_stems) <= excluded_names
    assert set(failing_candidates) <= excluded_names
