"""Phase 9e: tests for real-data mode, plus a synthetic-mode regression check.

Three things, per the Phase 9 instructions:

1. `test_synthetic_mode_regression_still_passes` - confirms source="synthetic"
   still passes exactly as before. Self-contained in this file (not just
   relying on test_pipeline.py) because Phase 9e explicitly asks for that
   confirmation here too, not just implicitly via a separate file passing.

2. `test_real_data_field_extraction_matches_ground_truth` - the field-level
   check: for each real document, does extract_real_fund_fields produce the
   hand-verified value, or null where the hand-verified answer says null is
   the only honest outcome? Calls field_extraction directly (not through
   run_pipeline) so a field-level assertion never depends on how pipeline.py
   happens to bucket a fund for a particular filter_spec.

3. `test_real_pipeline_flags_unreadable_document_without_halting_the_batch`
   - the pipeline-level check: run the real end-to-end path and confirm the
   deliberately corrupted PDF is reported, not silently dropped or crashed
   on, while the rest of the batch still produces a real answer.

See data/real_ground_truth.json for the hand-verified values and, for each
document, *why* a field is expected to be unambiguous or legitimately
flaggable - a real extractor that flags genuine uncertainty is correct
there, not just one that reproduces a single "true" value; a confident
wrong value is the only real failure. That distinction is why these tests
check against `acceptable_*` value lists rather than single expected values.
"""

import json
import os
from pathlib import Path

import pytest

from src.extraction.field_extraction import extract_real_fund_fields
from src.fund_filter import FilterSpec
from src.extraction.pdf_extraction import extract_pdf_content
from src.pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_DIR = PROJECT_ROOT / "data" / "user_uploads"


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, if present.

    Duplicated from src/pipeline.py (same reasoning as test_pipeline.py):
    needs to run at import time, before @pytest.mark.skipif's condition is
    evaluated at collection.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")

requires_groq = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY is not set (env or .env) - Phase 4/9c need a real Groq call.",
)


@requires_groq
def test_synthetic_mode_regression_still_passes(tmp_path: Path) -> None:
    """source="synthetic" must still reproduce data/ground_truth.json exactly - Phase 9 must not touch this.

    Deliberately duplicates the core of test_pipeline.py's assertion rather
    than importing it, so this file is a self-contained regression check on
    its own, per Phase 9e's explicit instruction to confirm this here too.
    """
    ground_truth = json.loads((PROJECT_ROOT / "data" / "ground_truth.json").read_text())

    result = run_pipeline(
        ground_truth["query"],
        FilterSpec(is_esg=True, status="active"),
        source="synthetic",
        run_log_dir=tmp_path,
    )

    assert result.source == "synthetic"
    assert result.verification_error is None, result.verification_error
    included_names = [fund.name for fund in result.included_funds]
    assert included_names == ground_truth["qualifying_funds"]
    assert result.final_answer == pytest.approx(ground_truth["correct_answer"])


@requires_groq
def test_real_data_field_extraction_matches_ground_truth() -> None:
    """Per-document field-level check against data/real_ground_truth.json.

    Calls extract_pdf_content + extract_real_fund_fields directly (Phase 2 +
    9c), not the full pipeline - this isolates "did extraction get the
    field right, or correctly flag it" from "how did the pipeline bucket
    this fund for a particular filter_spec", which is a separate concern
    covered by the pipeline-level test below.
    """
    ground_truth = json.loads((PROJECT_ROOT / "data" / "real_ground_truth.json").read_text())

    for stem, expected in ground_truth["documents"].items():
        if stem == "corrupted_fact_sheet":
            continue  # covered by the "could not be read" pipeline-level test below - there is no text to extract fields from

        pdf_path = REAL_DATA_DIR / f"{stem}.pdf"
        assert pdf_path.is_file(), f"Expected real-data fixture not found: {pdf_path}"

        text = extract_pdf_content(pdf_path)["text"]
        fields = extract_real_fund_fields(text)

        acceptable_is_esg = expected.get("acceptable_is_esg", [expected["is_esg"]])
        acceptable_status = expected.get("acceptable_status", [expected["status"]])
        acceptable_aum = expected.get("acceptable_aum", [expected["aum_millions_usd"]])

        assert fields.is_esg in acceptable_is_esg, (
            f"{stem}: is_esg={fields.is_esg!r} not in acceptable {acceptable_is_esg!r} "
            f"(flags: {fields.flags})"
        )
        assert fields.status in acceptable_status, (
            f"{stem}: status={fields.status!r} not in acceptable {acceptable_status!r} "
            f"(flags: {fields.flags})"
        )

        # expense_ratio is unambiguous for every real document in the fixture
        # set (see data/real_ground_truth.json) - no acceptable-null case to
        # allow for, unlike is_esg/status/aum above.
        expected_expense_ratio = expected["expense_ratio_percent"]
        assert expected_expense_ratio is not None, f"{stem}: fixture data bug - no expected expense_ratio"
        assert fields.expense_ratio == pytest.approx(expected_expense_ratio), (
            f"{stem}: expense_ratio={fields.expense_ratio!r}, expected {expected_expense_ratio!r} "
            f"(flags: {fields.flags})"
        )

        if acceptable_aum == [None]:
            # spy_fact_sheet: the source document never states the fund's own
            # AUM - only a different, similar-sounding figure (average market
            # cap of holdings). A guessed number here is the one outcome that
            # is NOT acceptable, unlike is_esg/status above.
            assert fields.aum is None, (
                f"{stem}: aum={fields.aum!r}, but the ground truth says this document does not "
                f"state its own AUM - a non-null value here means a real figure got confused for it "
                f"(flags: {fields.flags})"
            )
        else:
            assert fields.aum == pytest.approx(expected["aum_millions_usd"]), (
                f"{stem}: aum={fields.aum!r}, expected {expected['aum_millions_usd']!r} "
                f"(flags: {fields.flags})"
            )


@requires_groq
def test_real_pipeline_flags_unreadable_document_without_halting_the_batch(tmp_path: Path) -> None:
    """End-to-end real-data run: the corrupted PDF is reported, not silently dropped or crashed on.

    Uses is_esg=True/status="active" so the qualifying funds in this fixture
    set (ESGU, ESGV's ESG determination aside - see real_ground_truth.json)
    are ones with clean numeric fields, so this run is expected to complete
    and produce a real answer, while still surfacing the corrupted document
    as excluded with a clear reason - exactly the "at least one deliberately
    messy case, flagged not guessed" requirement from Phase 9's stopping
    condition.
    """
    result = run_pipeline(
        "weighted average expense ratio for ESG active funds",
        FilterSpec(is_esg=True, status="active"),
        source="real",
        raw_pdf_dir=REAL_DATA_DIR,
        run_log_dir=tmp_path,
    )

    assert result.source == "real"

    excluded_by_name = {fund.name: fund.reason for fund in result.excluded_funds}
    corrupted_reason = excluded_by_name.get("corrupted_fact_sheet")
    assert corrupted_reason is not None, (
        "corrupted_fact_sheet was not reported in excluded_funds at all - it must never be "
        "silently dropped from the batch"
    )
    assert "could not be read" in corrupted_reason, (
        f"corrupted_fact_sheet was excluded, but not for being unreadable: {corrupted_reason!r}"
    )

    # The whole point of the bucketing described in pipeline.py's module
    # docstring: one unreadable document must not prevent the rest of the
    # batch from producing a real answer.
    assert result.verification_error is None, result.verification_error
    assert result.final_answer is not None

    written_logs = list(tmp_path.glob("run_real_*.json"))
    assert len(written_logs) == 1
