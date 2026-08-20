"""Phase 7: full deterministic pipeline orchestration (Phases 2-6, chained).

This module is the thing everything else has been building toward: given a
natural-language question and the (explicit, hand-supplied) filter criteria
that answer it, run the whole chain -

    Phase 2 (pdf_extraction)   -> raw text per PDF
    Phase 3 (fund_filter)      -> which funds qualify, and why the rest don't
    Phase 4 (field_extraction) -> expense_ratio / aum per qualifying fund (LLM, forced tool-call)
    Phase 5 (verification)     -> expected-vs-collected completeness gate
    Phase 6 (calculator)       -> the final weighted-average number (plain arithmetic)

- and produce one JSON audit record: the answer, every included/excluded
fund with a reason, and the expected/collected counts Phase 5 checked. There
is no Phase 10 query parser yet, so `question` is carried through only as a
label for the audit record - it is never used to derive `filter_spec`; the
caller supplies both, deterministically.

The record is written to outputs/run_logs/ *before* re-raising a Phase 5
verification failure, so a failed run leaves the same kind of audit trail a
successful one does - the point of an audit trail is to show what happened
when something went wrong, not just when everything worked.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from groq import Groq

from src.calculator import weighted_average_expense_ratio
from src.field_extraction import DEFAULT_MODEL, ExtractedFields, extract_fund_fields
from src.fund_filter import FilterSpec, FundMetadata, filter_funds, parse_fund_metadata
from src.pdf_extraction import extract_pdf_content
from src.verification import ExtractionVerificationError, verify_extraction_completeness


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
DEFAULT_RUN_LOG_DIR = PROJECT_ROOT / "outputs" / "run_logs"


@dataclass(frozen=True)
class IncludedFund:
    """A fund that qualified (Phase 3) and was successfully extracted (Phase 4)."""

    name: str
    expense_ratio: float
    aum: float


@dataclass(frozen=True)
class ExcludedFund:
    """A fund left out of the final answer, and specifically why.

    `reason` distinguishes the two ways a fund can end up excluded: it never
    qualified per `filter_spec` (a Phase 3 decision, e.g. "not ESG"), or it
    qualified but Phase 4 failed to extract it (a Phase 4 failure, e.g.
    "extraction failed: ..."). Collapsing both into one silent "not
    included" would hide which phase is responsible when someone has to
    debug a run.
    """

    name: str
    reason: str


@dataclass(frozen=True)
class PipelineResult:
    """The full audit record for one end-to-end pipeline run."""

    timestamp: str
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


def _filter_spec_reason(metadata: FundMetadata, filter_spec: FilterSpec) -> str | None:
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


def run_pipeline(
    question: str,
    filter_spec: FilterSpec,
    *,
    raw_pdf_dir: Path = DEFAULT_RAW_PDF_DIR,
    run_log_dir: Path = DEFAULT_RUN_LOG_DIR,
    client: Groq | None = None,
    model: str = DEFAULT_MODEL,
) -> PipelineResult:
    """Run Phases 2-6 end to end for *question* / *filter_spec*, and write the audit record.

    Args:
        question: The natural-language question this run answers. Carried
            through into the audit record as a label only - Phase 10 (query
            parsing) doesn't exist yet, so it is never parsed into
            `filter_spec`; the caller supplies that separately.
        filter_spec: The deterministic Phase 3 filter criteria to apply.
        raw_pdf_dir: Directory of fund PDFs to run Phase 2 extraction over.
        run_log_dir: Directory the timestamped JSON audit record is written to.
        client: Groq client to reuse across all Phase 4 calls. Constructed
            fresh (`Groq()`) if omitted, same default as `extract_fund_fields`.
        model: Model passed through to every Phase 4 call.

    Returns:
        The `PipelineResult` that was also written to `run_log_dir`.

    Raises:
        ExtractionVerificationError: If Phase 5 finds the extracted fund set
            doesn't match what Phase 3 expected. The audit record is written
            to `run_log_dir` *before* this is raised, so the run is still
            diagnosable from disk even though the pipeline did not produce
            a final answer.
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
        and (reason := _filter_spec_reason(metadata, filter_spec)) is not None
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
    try:
        verify_extraction_completeness(
            expected_fund_names=qualifying_names,
            extracted_fund_names=[name for name, _ in successful],
        )
    except ExtractionVerificationError as exc:
        verification_error = str(exc)

    # --- Phase 6: the final number, only if verification passed ---
    final_answer: float | None = None
    if verification_error is None:
        final_answer = weighted_average_expense_ratio(
            {"expense_ratio": fund.expense_ratio, "aum": fund.aum} for fund in included_funds
        )

    result = PipelineResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
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

    _write_run_log(result, run_log_dir)

    if verification_error is not None:
        # Re-raise now that the audit record is safely on disk - Phase 5's
        # contract is "must not be proceeded past", and that still holds;
        # only the timing of the write changes so the failure is diagnosable.
        raise ExtractionVerificationError(
            expected_count=expected_count,
            extracted_count=collected_count,
            missing_funds=[n for n in qualifying_names if n not in {s[0] for s in successful}],
            unexpected_funds=[],
        )

    return result


def _write_run_log(result: PipelineResult, run_log_dir: Path) -> Path:
    """Write *result* as a timestamped JSON file under *run_log_dir* and return its path."""
    run_log_dir = Path(run_log_dir)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # Filesystem-safe timestamp (no colons) derived from the same instant
    # recorded in result.timestamp, plus microseconds so two runs in the
    # same second still get distinct filenames.
    file_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = run_log_dir / f"run_{file_stamp}.json"
    destination.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=False))
    logger.info("Wrote pipeline run log to %s", destination)
    return destination


def _print_summary(result: PipelineResult) -> None:
    print(f"Question: {result.question}")
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
    """Run the full pipeline once against the synthetic dataset and print the audit record."""
    _load_dotenv(PROJECT_ROOT / ".env")

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file), then re-run `python -m src.pipeline`."
        )

    question = "weighted average expense ratio for ESG funds, excluding funds closed in Q3"
    filter_spec = FilterSpec(is_esg=True, status="active")

    result = run_pipeline(question, filter_spec)
    _print_summary(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _run_phase_7_demo()
