"""Phase 7/9: full pipeline orchestration, in two clearly separated modes.

This module is the thing everything else has been building toward: given a
natural-language question and the (explicit, hand-supplied) filter criteria
that answer it, run the whole chain -

    Phase 2 (pdf_extraction)   -> raw text per PDF
    Phase 3-ish                -> which funds qualify, and why the rest don't
    Phase 4-ish                -> expense_ratio / aum per qualifying fund (LLM, forced tool-call)
    Phase 5 (verification)     -> expected-vs-collected completeness gate
    Phase 6 (calculator)       -> the final weighted-average number (plain arithmetic)

- and produce one JSON audit record: the answer, every included/excluded
fund with a reason, and the expected/collected counts Phase 5 checked.

Phase 9 adds a second *source* for that chain, alongside the original
synthetic one, and `run_pipeline`'s `source` parameter is a thin dispatcher
between the two - each has its own private function below so it's obvious
at a glance which mode is running and why:

    source="synthetic" -> `_run_synthetic_pipeline` (Phases 1-8, byte-for-byte
        unchanged): reads data/raw_pdfs/, filters via fund_filter.py's
        regex-based `parse_fund_metadata` (built for that fixture's exact
        field-label format), extracts expense_ratio/aum via
        `field_extraction.extract_fund_fields`. This is the project's
        dev/testing mode - it is what test_pipeline.py checks against
        data/ground_truth.json - and Phase 9 does not touch it.

    source="real" -> `_run_real_pipeline` (new): reads an arbitrary folder
        via `real_data_loader.load_real_pdfs`. `parse_fund_metadata`'s
        regexes are keyed to the synthetic fixture's exact label text and
        essentially never match a real document, so this path never calls
        it - each document goes through `extraction_cascade.extract_with_cascade`
        (regex pre-pass -> BM25 page-narrowing -> semantic-stub ->
        `field_extraction.extract_real_fund_fields` -> table-data retry ->
        OCR-stub -> flag), with every field allowed to come back null rather
        than guessed. See that module's docstring for the full tier
        breakdown, and the "expected vs. collected" note below for how
        Phase 5 still applies.

Both modes produce identically-shaped output (`PipelineResult`) and share
Phase 2, Phase 5, and Phase 6 verbatim - only how a document's identity/text
is found and how its fields are determined differs between them.

Phase 5, reused unchanged in real mode: real-data mode has no independent
pre-filter step the way synthetic mode's Phase 3 is - a document's is_esg/
status are THE fields being extracted, not known in advance. So real mode's
"expected" set (for `verify_extraction_completeness`) is built up as
documents are processed: a fund becomes "expected" once we can confidently
tell it qualifies (is_esg and status were both determined, and they satisfy
filter_spec) - exactly Phase 5's existing meaning, just fed by an LLM
determination instead of a regex one. A document we can't even tell
qualifies (unreadable, or is_esg/status came back null) is never "expected"
in the first place, the same way a non-ESG fund never was in synthetic
mode - so one bad document doesn't halt the whole batch. But a fund we ARE
confident qualifies and are then missing expense_ratio/aum for IS exactly
the gap Phase 5 already catches for synthetic mode's Phase 4 failures, and
it halts the run the same way, for the same reason.

The record is written to outputs/run_logs/ *before* re-raising a Phase 5
verification failure, so a failed run leaves the same kind of audit trail a
successful one does - the point of an audit trail is to show what happened
when something went wrong, not just when everything worked. This is shared
by both modes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from groq import Groq

from src.calculator import weighted_average_expense_ratio
from src.cascade.extraction_cascade import extract_with_cascade
from src.extraction.field_extraction import DEFAULT_MODEL, ExtractedFields, RealExtractedFields, extract_fund_fields
from src.fund_filter import FilterSpec, FundMetadata, filter_funds, parse_fund_metadata
from src.extraction.pdf_extraction import extract_pdf_content
from src.extraction.real_data_loader import load_real_pdfs
from src.verification import ExtractionVerificationError, verify_extraction_completeness


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
DEFAULT_REAL_DATA_DIR = PROJECT_ROOT / "data" / "user_uploads"
DEFAULT_RUN_LOG_DIR = PROJECT_ROOT / "outputs" / "run_logs"


@dataclass(frozen=True)
class IncludedFund:
    """A fund that qualified and was successfully extracted, in either mode."""

    name: str
    expense_ratio: float
    aum: float


@dataclass(frozen=True)
class ExcludedFund:
    """A fund left out of the final answer, and specifically why.

    In synthetic mode, `reason` distinguishes a Phase 3 filter exclusion
    (e.g. "not ESG") from a Phase 4 extraction failure (e.g. "extraction
    failed: ..."). In real mode it also covers two more cases Phase 9c
    requires: "could not be read: ..." (no extractable text at all) and
    "could not determine ESG/active status: ..." (is_esg or status came back
    null). Collapsing all of these into one silent "not included" would hide
    which phase, or which specific field, is responsible when someone has
    to debug a run.
    """

    name: str
    reason: str


@dataclass(frozen=True)
class PipelineResult:
    """The full audit record for one end-to-end pipeline run, either mode."""

    timestamp: str
    source: Literal["synthetic", "real"]
    question: str
    filter_spec: dict[str, object]
    final_answer: float | None
    included_funds: tuple[IncludedFund, ...]
    excluded_funds: tuple[ExcludedFund, ...]
    expected_count: int
    collected_count: int
    counts_match: bool
    verification_error: str | None

    def to_json_dict(self) -> dict[str, object]:
        """Plain-dict form suitable for `json.dump` - dataclasses aren't serializable directly."""
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "question": self.question,
            "filter_spec": self.filter_spec,
            "final_answer": self.final_answer,
            "included_funds": [asdict(fund) for fund in self.included_funds],
            "excluded_funds": [asdict(fund) for fund in self.excluded_funds],
            "expected_count": self.expected_count,
            "collected_count": self.collected_count,
            "counts_match": self.counts_match,
            "verification_error": self.verification_error,
        }


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, if present.

    Duplicated from field_extraction.py / verification.py rather than
    imported: it's a private helper in both, and each phase's entry point is
    meant to be runnable standalone. Existing environment variables are
    never overwritten.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ============================================================================
# source="synthetic" (Phases 1-8, unchanged)
# ============================================================================


def _synthetic_filter_reason(metadata: FundMetadata, filter_spec: FilterSpec) -> str | None:
    """Human-readable reason a Phase-3-parsed fund fails *filter_spec*, or None if it qualifies.

    `fund_filter.filter_funds` already makes the qualify/exclude decision -
    this only re-derives *why* for a fund it excluded, from the same parsed
    `FundMetadata`, so the audit record can say something more useful than
    "not included".
    """
    reasons: list[str] = []
    if filter_spec.is_esg is not None and metadata.is_esg != filter_spec.is_esg:
        reasons.append("is ESG" if metadata.is_esg else "not ESG")
    if filter_spec.status is not None and metadata.status != filter_spec.status.lower():
        if metadata.status == "closed" and metadata.closed_quarter:
            reasons.append(f"closed in {metadata.closed_quarter}")
        else:
            reasons.append(f"status is {metadata.status}")
    return " and ".join(reasons) if reasons else None


def _run_synthetic_pipeline(
    question: str,
    filter_spec: FilterSpec,
    *,
    raw_pdf_dir: Path,
    client: Groq | None,
    model: str,
) -> tuple[PipelineResult, ExtractionVerificationError | None]:
    """Phases 2-6 over the synthetic fixture set. Unmodified by Phase 9.

    Returns the result plus the verification exception to re-raise (or None
    on success) rather than raising directly, so `run_pipeline` can write
    the audit log before re-raising - see the module docstring.
    """
    active_client = client or Groq()

    # --- Phase 2: raw text per PDF ---
    pdf_paths = sorted(Path(raw_pdf_dir).glob("*.pdf"))
    extracted_funds = [extract_pdf_content(pdf_path) for pdf_path in pdf_paths]

    # Every fund's parsed metadata, keyed by name, so Phase 3's exclusion
    # reasons and Phase 4's inputs both come from the same parse instead of
    # re-parsing the text twice.
    name_to_text: dict[str, str] = {}
    name_to_metadata: dict[str, FundMetadata] = {}
    for fund in extracted_funds:
        text = fund["text"]
        if not isinstance(text, str):
            raise ValueError("Each extracted fund must contain a string 'text' field")
        metadata = parse_fund_metadata(text)
        name_to_text[metadata.name] = text
        name_to_metadata[metadata.name] = metadata

    # --- Phase 3: which funds qualify ---
    qualifying_names = filter_funds(extracted_funds, filter_spec)
    qualifying_set = set(qualifying_names)

    excluded_by_filter = [
        ExcludedFund(name=name, reason=reason)
        for name, metadata in name_to_metadata.items()
        if name not in qualifying_set
        and (reason := _synthetic_filter_reason(metadata, filter_spec)) is not None
    ]

    # --- Phase 4: extract numeric fields for each qualifying fund ---
    # Isolated per fund (rather than one try/except around the whole loop)
    # so one bad fund can't take down extraction for the rest of the batch -
    # the same isolation verification.py's demo already relies on to
    # simulate a single-fund failure.
    successful: list[tuple[str, ExtractedFields]] = []
    failed_extractions: list[ExcludedFund] = []
    for name in qualifying_names:
        try:
            extracted = extract_fund_fields(name_to_text[name], client=active_client, model=model)
        except Exception as exc:  # noqa: BLE001 - isolate one bad fund from the rest of the batch
            logger.error("Phase 4 extraction failed for %r: %s", name, exc)
            failed_extractions.append(ExcludedFund(name=name, reason=f"extraction failed: {exc}"))
            continue
        successful.append((name, extracted))

    included_funds = tuple(
        IncludedFund(name=name, expense_ratio=fields.expense_ratio, aum=fields.aum)
        for name, fields in successful
    )
    excluded_funds = tuple(excluded_by_filter) + tuple(failed_extractions)

    # --- Phase 5: verify nothing was silently dropped between Phase 3 and Phase 4 ---
    expected_count = len(qualifying_names)
    collected_count = len(successful)
    verification_error: str | None = None
    verification_exc: ExtractionVerificationError | None = None
    try:
        verify_extraction_completeness(
            expected_fund_names=qualifying_names,
            extracted_fund_names=[name for name, _ in successful],
        )
    except ExtractionVerificationError as exc:
        verification_error = str(exc)
        verification_exc = exc

    # --- Phase 6: the final number, only if verification passed ---
    final_answer: float | None = None
    if verification_error is None:
        final_answer = weighted_average_expense_ratio(
            {"expense_ratio": fund.expense_ratio, "aum": fund.aum} for fund in included_funds
        )

    result = PipelineResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="synthetic",
        question=question,
        filter_spec={"is_esg": filter_spec.is_esg, "status": filter_spec.status},
        final_answer=final_answer,
        included_funds=included_funds,
        excluded_funds=excluded_funds,
        expected_count=expected_count,
        collected_count=collected_count,
        counts_match=expected_count == collected_count,
        verification_error=verification_error,
    )
    return result, verification_exc


# ============================================================================
# source="real" (Phase 9)
# ============================================================================


def _real_filter_reason(fields: RealExtractedFields, filter_spec: FilterSpec) -> str | None:
    """Human-readable reason a real-mode fund fails *filter_spec*, or None if it qualifies.

    Mirrors `_synthetic_filter_reason` above, but reads from
    `RealExtractedFields` (LLM-determined) instead of `FundMetadata`
    (regex-parsed) - real mode never calls `parse_fund_metadata`, so there
    is no `FundMetadata` to reuse here. Only called once the caller has
    already confirmed both `is_esg` and `status` are non-null.
    """
    reasons: list[str] = []
    if filter_spec.is_esg is not None and fields.is_esg != filter_spec.is_esg:
        reasons.append("is ESG" if fields.is_esg else "not ESG")
    if filter_spec.status is not None and fields.status != filter_spec.status.lower():
        reasons.append(f"status is {fields.status}")
    return " and ".join(reasons) if reasons else None


def _run_real_pipeline(
    question: str,
    filter_spec: FilterSpec,
    *,
    raw_pdf_dir: Path,
    client: Groq | None,
    model: str,
) -> tuple[PipelineResult, ExtractionVerificationError | None]:
    """Phase 9: filter -> extract -> verify -> calculate over an arbitrary folder of real PDFs.

    See the module docstring for how this differs from
    `_run_synthetic_pipeline` and how Phase 5 still applies with no
    independent pre-filter step. Returns (result, verification_exc) for the
    same reason `_run_synthetic_pipeline` does.
    """
    active_client = client or Groq()

    pdf_paths = load_real_pdfs(raw_pdf_dir)

    included_funds: list[IncludedFund] = []
    excluded_funds: list[ExcludedFund] = []
    # "Expected" (Phase 5's exact existing meaning): funds we could
    # confidently determine DO qualify, regardless of whether we also got
    # clean numbers for them. Deliberately not "every PDF we found" - see
    # the module docstring for why that would make Phase 5 halt the whole
    # run over a single unreadable document.
    expected_names: list[str] = []
    collected: list[tuple[str, RealExtractedFields]] = []

    for pdf_path in pdf_paths:
        identity = pdf_path.stem  # canonical identity for real mode - see module docstring

        # runs extraction_cascade.py: Phase 2 read, then regex -> BM25 -> semantic -> LLM -> table-data -> OCR -> flag
        # question drives BM25/semantic narrowing - passed straight through, isolated per document so one bad document can't take down the batch
        try:
            cascade_result = extract_with_cascade(pdf_path, question, client=active_client, model=model)
        except Exception as exc:  # noqa: BLE001 - isolate one bad document from the rest of the batch
            logger.error("Extraction cascade failed for %r: %s", pdf_path.name, exc)
            excluded_funds.append(ExcludedFund(name=identity, reason=f"extraction failed: {exc}"))
            continue

        fields = cascade_result.fields
        display_name = fields.fund_name or identity

        # --- bucket 1: can't tell if it qualifies at all ---
        # Covers both "could not be read" (the cascade reports that with
        # is_esg/status both null too, so it lands here without a separate
        # check) and "is_esg/status genuinely not found". Uses the
        # cascade's own review_reasons rather than fields.flags - it's a
        # strict superset (also covers regex/LLM cross-check disagreements
        # and which tier, if any, resolved each field).
        if fields.is_esg is None or fields.status is None:
            reason = "; ".join(cascade_result.review_reasons) or "could not determine ESG/active status"
            excluded_funds.append(ExcludedFund(name=display_name, reason=reason))
            continue

        # --- bucket 2: we can tell, and it doesn't qualify ---
        reason = _real_filter_reason(fields, filter_spec)
        if reason is not None:
            excluded_funds.append(ExcludedFund(name=display_name, reason=reason))
            continue

        # --- bucket 3: qualifies - this is "expected" for Phase 5's purposes ---
        expected_names.append(display_name)
        if fields.expense_ratio is not None and fields.aum is not None:
            collected.append((display_name, fields))
            included_funds.append(
                IncludedFund(name=display_name, expense_ratio=fields.expense_ratio, aum=fields.aum)
            )
        # else: qualifies, but expense_ratio and/or aum is missing/unparseable.
        # Deliberately NOT reported here as its own excluded-fund entry -
        # it stays "expected but not collected", and
        # verify_extraction_completeness below is what reports it, exactly
        # like a failed Phase 4 call does in synthetic mode. Its specific
        # flags (which field, and why) are still in `fields.flags` and are
        # visible in the ERROR log line verify_extraction_completeness emits.

    # --- Phase 5: verify every fund we're confident qualifies also has clean numbers ---
    verification_error: str | None = None
    verification_exc: ExtractionVerificationError | None = None
    try:
        verify_extraction_completeness(
            expected_fund_names=expected_names,
            extracted_fund_names=[name for name, _ in collected],
        )
    except ExtractionVerificationError as exc:
        verification_error = str(exc)
        verification_exc = exc

    # --- Phase 6: the final number, only if verification passed and something qualified ---
    final_answer: float | None = None
    if verification_error is None and included_funds:
        final_answer = weighted_average_expense_ratio(
            {"expense_ratio": fund.expense_ratio, "aum": fund.aum} for fund in included_funds
        )

    result = PipelineResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="real",
        question=question,
        filter_spec={"is_esg": filter_spec.is_esg, "status": filter_spec.status},
        final_answer=final_answer,
        included_funds=tuple(included_funds),
        excluded_funds=tuple(excluded_funds),
        expected_count=len(expected_names),
        collected_count=len(collected),
        counts_match=len(expected_names) == len(collected),
        verification_error=verification_error,
    )
    return result, verification_exc


# ============================================================================
# Shared entry point
# ============================================================================


def run_pipeline(
    question: str,
    filter_spec: FilterSpec,
    *,
    source: Literal["synthetic", "real"] = "synthetic",
    raw_pdf_dir: Path | None = None,
    run_log_dir: Path = DEFAULT_RUN_LOG_DIR,
    client: Groq | None = None,
    model: str = DEFAULT_MODEL,
) -> PipelineResult:
    """Run the full pipeline for *question* / *filter_spec*, and write the audit record.

    This function itself contains no filter/extract/verify/calculate logic -
    it only picks which of the two fully-separate implementations to run
    (see the module docstring) and handles what both share: writing the
    audit record, then re-raising a Phase 5 failure if there was one.

    Args:
        question: The natural-language question this run answers. Carried
            through into the audit record as a label only - Phase 10 (query
            parsing) doesn't exist yet, so it is never parsed into
            `filter_spec`; the caller supplies that separately.
        filter_spec: The deterministic filter criteria to apply.
        source: "synthetic" (default) reads `raw_pdf_dir` (default
            data/raw_pdfs/) and filters via the regex-based
            `fund_filter.parse_fund_metadata` - Phases 1-8, unchanged.
            "real" reads `raw_pdf_dir` (default data/user_uploads/) via
            `real_data_loader.load_real_pdfs` and determines every field,
            including is_esg/status, via one LLM pass per document - Phase 9.
        raw_pdf_dir: Directory of PDFs to read. Defaults to
            `DEFAULT_RAW_PDF_DIR` for source="synthetic" or
            `DEFAULT_REAL_DATA_DIR` for source="real" - pass explicitly to
            override either.
        run_log_dir: Directory the timestamped JSON audit record is written to.
        client: Groq client to reuse across all LLM calls. Constructed
            fresh (`Groq()`) if omitted.
        model: Model passed through to every LLM call.

    Returns:
        The `PipelineResult` that was also written to `run_log_dir`.

    Raises:
        ExtractionVerificationError: If Phase 5 finds a fund the run was
            confident should be included is missing clean numeric fields.
            The audit record is written to `run_log_dir` *before* this is
            raised, so the run is still diagnosable from disk even though
            the pipeline did not produce a final answer.
        ValueError: If `source` is neither "synthetic" nor "real".
    """
    if source == "synthetic":
        result, verification_exc = _run_synthetic_pipeline(
            question,
            filter_spec,
            raw_pdf_dir=raw_pdf_dir if raw_pdf_dir is not None else DEFAULT_RAW_PDF_DIR,
            client=client,
            model=model,
        )
    elif source == "real":
        result, verification_exc = _run_real_pipeline(
            question,
            filter_spec,
            raw_pdf_dir=raw_pdf_dir if raw_pdf_dir is not None else DEFAULT_REAL_DATA_DIR,
            client=client,
            model=model,
        )
    else:
        raise ValueError(f"Unknown source {source!r}; expected 'synthetic' or 'real'")

    _write_run_log(result, run_log_dir)

    if verification_exc is not None:
        # Re-raise now that the audit record is safely on disk - Phase 5's
        # contract is "must not be proceeded past", and that still holds for
        # both modes; only the timing of the write changes so the failure is
        # diagnosable. Re-raising the *original* exception object (rather
        # than reconstructing one from result fields) guarantees its
        # missing/unexpected lists can never drift from what actually
        # happened above.
        raise verification_exc

    return result


def _write_run_log(result: PipelineResult, run_log_dir: Path) -> Path:
    """Write *result* as a timestamped JSON file under *run_log_dir* and return its path."""
    run_log_dir = Path(run_log_dir)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # Filesystem-safe timestamp (no colons) derived from the same instant
    # recorded in result.timestamp, plus microseconds so two runs in the
    # same second still get distinct filenames.
    file_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = run_log_dir / f"run_{result.source}_{file_stamp}.json"
    destination.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=False))
    logger.info("Wrote pipeline run log to %s", destination)
    return destination


def _print_summary(result: PipelineResult) -> None:
    print(f"[source={result.source}] Question: {result.question}")
    print(f"Filter spec: {result.filter_spec}\n")

    print(f"Included funds ({len(result.included_funds)}):")
    for fund in result.included_funds:
        print(f"  - {fund.name}: expense_ratio={fund.expense_ratio}, aum={fund.aum}")

    print(f"\nExcluded funds ({len(result.excluded_funds)}):")
    for fund in result.excluded_funds:
        print(f"  - {fund.name}: {fund.reason}")

    print(
        f"\nExpected count: {result.expected_count}  Collected count: {result.collected_count}"
        f"  Match: {result.counts_match}"
    )
    if result.verification_error:
        print(f"\nVerification FAILED: {result.verification_error}")
    else:
        print(f"\nFinal answer (weighted average expense ratio): {result.final_answer}")


def _run_phase_7_demo() -> None:
    """Run the full pipeline once against the synthetic dataset and print the audit record.

    Unchanged by Phase 9 - still the default `python -m src.pipeline`
    behavior, per the instruction not to touch how synthetic mode runs.
    """
    _load_dotenv(PROJECT_ROOT / ".env")

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file), then re-run `python -m src.pipeline`."
        )

    question = "weighted average expense ratio for ESG funds, excluding funds closed in Q3"
    filter_spec = FilterSpec(is_esg=True, status="active")

    result = run_pipeline(question, filter_spec, source="synthetic")
    _print_summary(result)


def _run_phase_9_demo() -> None:
    """Run real-data mode against data/user_uploads/ and print the audit record.

    Not wired into `if __name__ == "__main__"` on purpose - `python -m
    src.pipeline` must keep running the Phase 7 synthetic demo unchanged.
    Run this one explicitly instead:
        python -c "from src.pipeline import _run_phase_9_demo; _run_phase_9_demo()"
    """
    _load_dotenv(PROJECT_ROOT / ".env")

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file)."
        )

    question = "weighted average expense ratio for ESG active funds"
    filter_spec = FilterSpec(is_esg=True, status="active")

    result = run_pipeline(question, filter_spec, source="real")
    _print_summary(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _run_phase_7_demo()
