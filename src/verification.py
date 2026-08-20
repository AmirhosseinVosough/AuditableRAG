"""Collection-completeness verification between Phase 3 and Phase 4.

Phase 3 (``fund_filter.filter_funds``) decides which funds *should* flow into
the weighted-average calculation. Phase 4 (``field_extraction.extract_fund_fields``)
then tries to pull numeric fields out of each of those funds individually, and
any single fund's LLM call can fail (malformed PDF, empty text, a dropped
tool call, a network error). If that failure is swallowed and Phase 6 simply
averages whatever fields *did* come back, the pipeline silently produces a
wrong answer computed over the wrong set of funds — with no signal that
anything was ever missing.

This module is the gate between Phase 4 and Phase 6: it compares the fund
identities Phase 3 expected against the fund identities Phase 4 actually
produced results for, and raises loudly on any mismatch instead of letting
the calculation proceed over an incomplete set.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


logger = logging.getLogger(__name__)

# The Phase 3 filter's expected fund count, per data/ground_truth.json. Not
# enforced by verify_extraction_completeness (which works for any count) —
# only used by the Phase 5 demo below to describe what it's showing.
_EXPECTED_QUALIFYING_FUND_COUNT = 6


class ExtractionVerificationError(Exception):
    """Raised when Phase 4's extracted funds don't match Phase 3's expected set.

    Carries the specific fund name(s) responsible so a human (or a retry
    policy) can act on exactly what went wrong, rather than a generic
    "counts don't match" message that would require re-diffing both lists
    by hand.
    """

    def __init__(
        self,
        *,
        expected_count: int,
        extracted_count: int,
        missing_funds: Sequence[str],
        unexpected_funds: Sequence[str],
    ) -> None:
        self.expected_count = expected_count
        self.extracted_count = extracted_count
        self.missing_funds = list(missing_funds)
        self.unexpected_funds = list(unexpected_funds)

        parts = [
            f"Extraction verification failed: Phase 3 expected {expected_count} "
            f"fund(s), Phase 4 successfully extracted {extracted_count}."
        ]
        if self.missing_funds:
            named = ", ".join(f"'{name}'" for name in self.missing_funds)
            parts.append(f"Missing fund(s) (expected but not extracted): {named}.")
        if self.unexpected_funds:
            named = ", ".join(f"'{name}'" for name in self.unexpected_funds)
            parts.append(f"Unexpected fund(s) (extracted but not expected): {named}.")

        super().__init__(" ".join(parts))


@dataclass(frozen=True)
class VerificationResult:
    """Successful outcome of `verify_extraction_completeness`.

    Kept as a real return value (rather than just returning None on success)
    so callers building an audit log — e.g. Phase 7's run log — have a
    structured record of exactly what was checked and passed.
    """

    expected_funds: tuple[str, ...]
    extracted_funds: tuple[str, ...]


def verify_extraction_completeness(
    expected_fund_names: Sequence[str],
    extracted_fund_names: Sequence[str],
) -> VerificationResult:
    """Confirm Phase 4 produced a result for every fund Phase 3 selected.

    This is an identity/count check only — it does not re-validate any of
    the numeric fields Phase 4 extracted, only that no fund was silently
    dropped between the two phases. It must run before Phase 6's
    calculation, because a missing fund wouldn't cause a crash there: it
    would just silently skew the weighted average over the wrong set of
    funds.

    Args:
        expected_fund_names: Qualifying fund names returned by
            `fund_filter.filter_funds` (Phase 3). This is the trusted
            baseline — the set of funds that *should* be in the final
            calculation.
        extracted_fund_names: The Phase-3 identity string for each fund
            whose Phase 4 extraction call succeeded. Callers must pass the
            same name string Phase 3 produced for that fund (the loop
            variable used to look up its text), not a name re-derived from
            the LLM's own output — the LLM can restate a fund's name with
            slightly different wording even when extraction otherwise
            succeeds, which would cause a false mismatch here.

    Returns:
        VerificationResult recording the expected and extracted fund sets,
        for callers that want a structured record of a passing check.

    Raises:
        ExtractionVerificationError: If any fund in `expected_fund_names` is
            absent from `extracted_fund_names` (a fund Phase 4 failed to
            extract), or if `extracted_fund_names` contains a fund that
            was never in `expected_fund_names` (a fund that should never
            have reached Phase 4 in the first place). Either case is a
            broken invariant between phases and must not be proceeded past.
    """
    expected_set = set(expected_fund_names)
    extracted_set = set(extracted_fund_names)

    # List comprehensions (not set difference) so the error message reports
    # missing/unexpected funds in a stable, readable order instead of
    # arbitrary set-iteration order.
    missing_funds = [name for name in expected_fund_names if name not in extracted_set]
    unexpected_funds = [name for name in extracted_fund_names if name not in expected_set]

    if missing_funds or unexpected_funds:
        logger.error(
            "Extraction verification failed: expected=%d extracted=%d missing=%s unexpected=%s",
            len(expected_fund_names),
            len(extracted_fund_names),
            missing_funds,
            unexpected_funds,
        )
        raise ExtractionVerificationError(
            expected_count=len(expected_fund_names),
            extracted_count=len(extracted_fund_names),
            missing_funds=missing_funds,
            unexpected_funds=unexpected_funds,
        )

    logger.info(
        "Extraction verification passed: all %d expected fund(s) were extracted.",
        len(expected_fund_names),
    )
    return VerificationResult(
        expected_funds=tuple(expected_fund_names),
        extracted_funds=tuple(extracted_fund_names),
    )


# --- Amended Phase 5: per-fund quality checks (bounds + name consistency) ---
#
# `verify_extraction_completeness` above answers "did every expected fund
# produce *a* result" - it says nothing about whether that result is
# *plausible*. A fund whose text says "Expense Ratio: 999%" (a corrupted
# document, a mis-OCR'd digit, a unit mix-up) or whose extracted name
# doesn't match the document Phase 3 actually parsed (the model answered
# about the wrong fund) passes the completeness check just fine - it's not
# missing, it's just wrong. This is the gap the build prompt's "Amendments
# to Phases 1-9" table calls out for this module, and Phase 12's agentic
# orchestrator requires it as part of its EVALUATE step (retrying a fund
# that fails this, not just one that failed to extract at all).
#
# Deliberately a *separate* function, not a change to
# verify_extraction_completeness's behavior or signature - existing callers
# (pipeline.py, test_pipeline.py) are unaffected. Bounds are conservative
# but real: no fund in this project's domain has an expense ratio anywhere
# near 10%, or zero/negative AUM.
_MAX_PLAUSIBLE_EXPENSE_RATIO_PERCENT = 10.0


def check_extraction_quality(
    *,
    expected_name: str,
    extracted_name: str,
    expense_ratio: float,
    aum: float,
) -> tuple[str, ...]:
    """Return every quality problem found with one fund's extracted fields; empty if none.

    Args:
        expected_name: The fund name Phase 3 parsed for this document -
            the trusted identity, independent of what the LLM says.
        extracted_name: The fund name Phase 4 (`extract_fund_fields`)
            reported for the same document.
        expense_ratio: Extracted expense ratio, as a percentage number
            (e.g. 0.45 for 0.45%).
        aum: Extracted assets under management, in millions of USD.

    Returns:
        A tuple of human-readable problem descriptions - one entry per
        problem found, so a caller can log or retry on the specific
        reason rather than a generic "invalid". Empty tuple means every
        check passed.
    """
    problems: list[str] = []

    if extracted_name != expected_name:
        problems.append(
            f"name mismatch: Phase 3 parsed {expected_name!r} for this document, "
            f"but extraction reported {extracted_name!r}"
        )

    if not (0 < expense_ratio < _MAX_PLAUSIBLE_EXPENSE_RATIO_PERCENT):
        problems.append(
            f"expense_ratio {expense_ratio!r} is outside the plausible range "
            f"(0, {_MAX_PLAUSIBLE_EXPENSE_RATIO_PERCENT})"
        )

    if not (aum > 0):
        problems.append(f"aum {aum!r} is not positive")

    return tuple(problems)


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, if present.

    Duplicated from field_extraction.py rather than imported: it's a private
    (leading-underscore) helper there, and each phase's `__main__` demo is
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


# --- Phase 5 stopping-condition demo: deliberately corrupt one PDF -----------
#
# Target a fund that genuinely qualifies (per data/ground_truth.json) so the
# demo proves a *real* mismatch gets caught, not a fund that was never
# expected in the first place.
_DEMO_CORRUPTION_TARGET_FILENAME = "F002_horizon_sustainable_bond.pdf"
_DEMO_CORRUPTION_TARGET_FUND_NAME = "Horizon Sustainable Bond Fund"


def _corrupted_pdf_bytes() -> bytes:
    """Bytes for a PDF that is broken past the header.

    This mirrors how real fund PDFs actually arrive corrupted (a partial
    download, a disk error): the ``%PDF`` header survives, but there is no
    valid cross-reference table or page tree behind it, so pdfplumber cannot
    walk it into pages.
    """
    return b"%PDF-1.4\n% Deliberately corrupted for the Phase 5 verification demo.\n"


def _extract_text_from_corrupted_copy(raw_pdf_dir: Path) -> str:
    """Corrupt a copy of the demo target PDF and run it through real Phase 2 extraction.

    Uses `pdf_extraction.extract_pdf_content` on the corrupted file rather
    than hand-crafting a broken string, so the failure this demonstrates is
    the one Phase 4 would actually see in production, not a stand-in for it.

    Returns:
        The extracted text, which in practice is "" — either because
        pdfplumber raises trying to parse the corrupted structure (caught
        and logged below) or because it opens but finds no readable pages.
    """
    from src.pdf_extraction import extract_pdf_content

    source_pdf = raw_pdf_dir / _DEMO_CORRUPTION_TARGET_FILENAME
    if not source_pdf.is_file():
        raise FileNotFoundError(
            f"Expected demo corruption target not found: {source_pdf}. "
            "Run `python -m src.generate_synthetic_data` first."
        )

    with tempfile.TemporaryDirectory(prefix="phase5_corrupted_pdf_") as tmp_dir:
        corrupted_pdf = Path(tmp_dir) / source_pdf.name
        corrupted_pdf.write_bytes(_corrupted_pdf_bytes())
        try:
            extraction = extract_pdf_content(corrupted_pdf)
        except Exception as exc:  # noqa: BLE001 - intentional: any parser failure means "unreadable"
            logger.info(
                "Corrupted PDF failed extraction outright (expected for this demo): %s", exc
            )
            return ""
        text = extraction["text"]
        assert isinstance(text, str)
        return text


def _run_phase_5_demo() -> None:
    """Run Phases 3-5 end to end, deliberately corrupting one fund's PDF first.

    This is the Phase 5 stopping-condition demonstration: Phase 3 filtering
    runs against the real, uncorrupted PDFs (so the "expected" list is a
    trustworthy baseline), then one qualifying fund's text is swapped for
    the output of a genuinely corrupted copy of its PDF before Phase 4
    extraction runs. That fund's Phase 4 call fails as a result, and
    `verify_extraction_completeness` must catch the resulting mismatch and
    name the missing fund rather than letting the run proceed.
    """
    from groq import Groq

    from src.field_extraction import ExtractedFields, extract_fund_fields
    from src.fund_filter import FilterSpec, filter_funds, parse_fund_metadata
    from src.pdf_extraction import extract_pdf_content

    project_root = Path(__file__).resolve().parents[1]
    _load_dotenv(project_root / ".env")

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file), then re-run `python -m src.verification`."
        )

    raw_pdf_dir = project_root / "data" / "raw_pdfs"
    pdf_paths = sorted(raw_pdf_dir.glob("*.pdf"))
    extracted_funds = [extract_pdf_content(pdf_path) for pdf_path in pdf_paths]

    # Phase 3: filtering runs against the real, uncorrupted PDFs, so
    # qualifying_names is the trustworthy "expected" baseline.
    qualifying_names = filter_funds(
        extracted_funds, FilterSpec(is_esg=True, status="active")
    )

    name_to_text: dict[str, str] = {}
    for fund in extracted_funds:
        text = fund["text"]
        if not isinstance(text, str):
            raise ValueError("Each extracted fund must contain a string 'text' field")
        name_to_text[parse_fund_metadata(text).name] = text

    if _DEMO_CORRUPTION_TARGET_FUND_NAME not in qualifying_names:
        raise RuntimeError(
            f"Demo corruption target {_DEMO_CORRUPTION_TARGET_FUND_NAME!r} is not "
            "in the Phase 3 qualifying list; the synthetic dataset may have changed."
        )

    print(f"Phase 3 filter expects {len(qualifying_names)} qualifying funds:")
    for name in qualifying_names:
        print(f"  - {name}")

    print(f"\n[demo] Corrupting the PDF for: {_DEMO_CORRUPTION_TARGET_FUND_NAME!r}")
    corrupted_text = _extract_text_from_corrupted_copy(raw_pdf_dir)
    print(f"[demo] Extracted text from the corrupted PDF: {corrupted_text!r} (empty, as expected)")
    name_to_text[_DEMO_CORRUPTION_TARGET_FUND_NAME] = corrupted_text

    client = Groq()
    successful: list[tuple[str, ExtractedFields]] = []
    failed: list[tuple[str, Exception]] = []
    print("\nRunning Phase 4 extraction for each qualifying fund...")
    for name in qualifying_names:
        try:
            extracted = extract_fund_fields(name_to_text[name], client=client)
        except Exception as exc:  # noqa: BLE001 - isolate one bad fund from the rest of the batch
            logger.error("Phase 4 extraction failed for %r: %s", name, exc)
            failed.append((name, exc))
            print(f"  - {name}: FAILED ({exc})")
            continue
        successful.append((name, extracted))
        print(f"  - {name}: ok")

    print(
        f"\nPhase 4 result: {len(successful)} succeeded, {len(failed)} failed "
        f"out of {len(qualifying_names)} expected."
    )

    print("\nRunning Phase 5 verification...")
    try:
        verify_extraction_completeness(
            expected_fund_names=qualifying_names,
            extracted_fund_names=[name for name, _ in successful],
        )
    except ExtractionVerificationError as exc:
        print(f"\nVerification FAILED (this is the expected outcome for this demo):\n  {exc}")
        return

    # Should be unreachable in this demo, since the corruption above always
    # forces at least one failure — kept as a real code path (not asserted
    # away) so this function is also correct when run against a clean set.
    print("\nVerification passed: no mismatch detected.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _run_phase_5_demo()
