"""Phase 12: the agentic orchestrator - a real REASON -> ACT -> EVALUATE -> RETRY loop.

Per the core architectural mandate (AGENTIC_LAYER_BUILD_PROMPT.md): this
module contains NO filtering logic and NO arithmetic of its own. Every
actual decision about "does this fund qualify" (Phase 3, via
`retrieval.scope_documents`), "is this extracted value good" (the amended
Phase 5 quality check, `verification.check_extraction_quality`), and "what's
the final number" (Phase 6, `calculator.weighted_average_expense_ratio`) is
made by calling the existing deterministic tools, unchanged. What this
module owns is purely the orchestration: which tool to call next, whether to
retry, when to give up, and writing down *why* for every one of those
decisions - never the underlying judgment itself.

This is built on the synthetic deterministic core (Phases 3/4/5/6), not
real-data mode (Phase 9) - `field_extraction.extract_fund_fields` is what the
build prompt names for the ACT step, not `extract_real_fund_fields`. Real
data has its own extraction cascade (extraction_cascade.py); wiring that
into an agentic loop is a different, separate piece of work, not implied by
this phase's spec.

Loop structure (REASON -> ACT -> EVALUATE -> RETRY -> ACT -> ANSWER):

    REASON   - query_parser.parse_query(question) -> QuerySpec
    ACT      - retrieval.scope_documents(query_spec, ...) -> candidate names,
               then field_extraction.extract_fund_fields(...) per candidate
    EVALUATE - verification.check_extraction_quality(...) per result
    RETRY    - a fund that fails EVALUATE is re-extracted, up to
               MAX_RETRY_ATTEMPTS (a real loop counter with a hard bound in
               code - see the `while attempt < MAX_RETRY_ATTEMPTS` loop
               below, not just this docstring)
    (exclude) - still failing after the cap: excluded with a logged reason,
               the rest of the batch continues - UNLESS excluded funds would
               exceed MAX_EXCLUDED_FRACTION of the candidate set, in which
               case the whole run hard-stops (mirrors v1's "never silently
               proceed on bad data" principle - see pipeline.py's Phase 5
               gate for the same idea applied differently)
    ACT      - calculator.weighted_average_expense_ratio(...) on the final
               verified set
    ANSWER   - return the number plus the full decision trace: every retry,
               exclusion, and reason, in order
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from groq import Groq

from src.calculator import weighted_average_expense_ratio
from src.field_extraction import DEFAULT_MODEL, ExtractedFields, extract_fund_fields
from src.fund_filter import parse_fund_metadata
from src.pdf_extraction import extract_pdf_content
from src.query_parser import QuerySpec, parse_query
from src.retrieval import scope_documents
from src.verification import check_extraction_quality


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"

# Hard-coded, per the mandate that iteration caps are never left to the
# model's judgment. Phase 12's own spec number (the "Amendments" table's
# "2 attempts" was written for a pipeline.py retry that was never separately
# built - this orchestrator's RETRY step supersedes it).
MAX_RETRY_ATTEMPTS = 3

# If more than this fraction of the candidate set ends up excluded after
# exhausting retries, hard-stop instead of computing an answer over too
# incomplete a set - the agentic-layer version of v1's "never silently
# proceed on bad data" principle (pipeline.py's Phase 5 gate is the
# deterministic-core version of the same idea).
MAX_EXCLUDED_FRACTION = 0.5


@dataclass(frozen=True)
class DecisionTraceEntry:
    """One step of the ReAct loop's reasoning, in order. `detail` is always an explicit reason, never just a status code."""

    step: str  # "REASON" | "ACT" | "EVALUATE" | "RETRY" | "EXCLUDE" | "HARD-STOP" | "ANSWER"
    detail: str


def run_agentic_pipeline(
    question: str,
    *,
    raw_pdf_dir: Path = DEFAULT_RAW_PDF_DIR,
    client: Groq | None = None,
    model: str = DEFAULT_MODEL,
    _text_override: dict[str, str] | None = None,
) -> dict:
    """Run the full REASON -> ACT -> EVALUATE -> RETRY -> ACT -> ANSWER loop for *question*.

    Args:
        question: Natural-language question. Parsed into a QuerySpec via
            `query_parser.parse_query` - the orchestrator never derives
            filter criteria itself.
        raw_pdf_dir: Directory of synthetic fund PDFs (Phase 2 input).
        client: Groq client to reuse across every LLM call in the loop
            (query parsing and every extraction attempt). Constructed
            fresh (`Groq()`) if omitted.
        model: Model passed through to every LLM call.
        _text_override: Testing/demo hook only - substitutes a specific
            fund's text (keyed by the name Phase 3 parses for it) after
            scoping has already run against the real PDFs, so a
            deliberately corrupted fund can be exercised without touching
            which documents get scoped in. Not part of the orchestrator's
            real operating contract - see `_run_phase_12_broken_fund_demo`.

    Returns:
        A dict: {question, query_spec, final_answer, included_funds,
        excluded_funds, hard_stopped, hard_stop_reason, decision_trace}.
        `final_answer` is None if nothing qualified or the run hard-stopped.
        `decision_trace` is the full ordered log of every REASON/ACT/
        EVALUATE/RETRY/EXCLUDE/HARD-STOP/ANSWER step, each with an explicit
        reason string - a human should be able to reconstruct exactly what
        happened and why using only this list.
    """
    trace: list[DecisionTraceEntry] = []

    def log(step: str, detail: str) -> None:
        trace.append(DecisionTraceEntry(step=step, detail=detail))
        logger.info("[%s] %s", step, detail)

    active_client = client or Groq()

    # --- REASON: turn the question into structure. No filtering logic here -
    # parse_query either returns a QuerySpec or raises UnsupportedQueryError;
    # the orchestrator doesn't second-guess either outcome. ---
    query_spec: QuerySpec = parse_query(question, client=active_client, model=model)
    log("REASON", f"parsed question into {query_spec}")

    # --- ACT (scope): Phase 2 read, then Phase 11's scope_documents (which is
    # Phase 3's filter_funds under the hood) - not reimplemented here. ---
    pdf_paths = sorted(Path(raw_pdf_dir).glob("*.pdf"))
    extracted_funds = [extract_pdf_content(pdf_path) for pdf_path in pdf_paths]

    name_to_text: dict[str, str] = {}
    for fund in extracted_funds:
        text = fund["text"]
        if not isinstance(text, str):
            raise ValueError("Each extracted fund must contain a string 'text' field")
        name_to_text[parse_fund_metadata(text).name] = text

    if _text_override:
        name_to_text.update(_text_override)

    candidate_names = scope_documents(query_spec, extracted_funds)
    log(
        "ACT",
        f"scope_documents selected {len(candidate_names)} candidate(s) out of "
        f"{len(extracted_funds)} document(s): {candidate_names}",
    )

    # --- ACT (extract) + EVALUATE + RETRY, per candidate ---
    included: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []

    for name in candidate_names:
        text = name_to_text[name]
        resolved: ExtractedFields | None = None

        attempt = 0
        while attempt < MAX_RETRY_ATTEMPTS:  # hard cap enforced here, in code - not just documented
            attempt += 1
            try:
                extracted = extract_fund_fields(text, client=active_client, model=model)
            except Exception as exc:  # noqa: BLE001 - any call failure is a retry candidate, isolated to this fund
                log(
                    "RETRY",
                    f"{name}: attempt {attempt}/{MAX_RETRY_ATTEMPTS} - extraction call failed: {exc}",
                )
                continue

            problems = check_extraction_quality(
                expected_name=name,
                extracted_name=extracted.fund_name,
                expense_ratio=extracted.expense_ratio,
                aum=extracted.aum,
            )
            if not problems:
                log("EVALUATE", f"{name}: attempt {attempt}/{MAX_RETRY_ATTEMPTS} passed every quality check")
                resolved = extracted
                break

            log(
                "RETRY",
                f"{name}: attempt {attempt}/{MAX_RETRY_ATTEMPTS} failed EVALUATE - {'; '.join(problems)}",
            )

        if resolved is not None:
            included.append({"name": name, "expense_ratio": resolved.expense_ratio, "aum": resolved.aum})
        else:
            reason = f"excluded after {MAX_RETRY_ATTEMPTS} failed attempts (see RETRY entries above)"
            log("EXCLUDE", f"{name}: {reason}")
            excluded.append({"name": name, "reason": reason})

    # --- hard-stop check: mirrors v1's "never silently proceed on bad data" ---
    if candidate_names:
        excluded_fraction = len(excluded) / len(candidate_names)
        if excluded_fraction > MAX_EXCLUDED_FRACTION:
            hard_stop_reason = (
                f"{len(excluded)}/{len(candidate_names)} candidates ({excluded_fraction:.0%}) were "
                f"excluded after exhausting retries, exceeding the {MAX_EXCLUDED_FRACTION:.0%} "
                "threshold - hard-stopping rather than compute an answer over too incomplete a set"
            )
            log("HARD-STOP", hard_stop_reason)
            return _build_result(
                question=question,
                query_spec=query_spec,
                final_answer=None,
                included=included,
                excluded=excluded,
                hard_stopped=True,
                hard_stop_reason=hard_stop_reason,
                trace=trace,
            )

    # --- ACT (calculate): Phase 6, unmodified. QuerySpec.requested_metric is
    # always "weighted_average_expense_ratio" today - the only metric
    # calculator.py computes, so there's nothing to dispatch on yet. ---
    final_answer: float | None = None
    if included:
        final_answer = weighted_average_expense_ratio(
            {"expense_ratio": fund["expense_ratio"], "aum": fund["aum"]} for fund in included
        )
        log(
            "ANSWER",
            f"weighted_average_expense_ratio over {len(included)} fund(s) = {final_answer}",
        )
    else:
        log("ANSWER", "no funds ended up included - no answer to compute")

    return _build_result(
        question=question,
        query_spec=query_spec,
        final_answer=final_answer,
        included=included,
        excluded=excluded,
        hard_stopped=False,
        hard_stop_reason=None,
        trace=trace,
    )


def _build_result(
    *,
    question: str,
    query_spec: QuerySpec,
    final_answer: float | None,
    included: Sequence[dict[str, object]],
    excluded: Sequence[dict[str, object]],
    hard_stopped: bool,
    hard_stop_reason: str | None,
    trace: Sequence[DecisionTraceEntry],
) -> dict:
    """Assemble the final dict return value - the single place its shape is defined."""
    return {
        "question": question,
        "query_spec": {
            "is_esg": query_spec.is_esg,
            "status": query_spec.status,
            "closed_quarter_exclusions": list(query_spec.closed_quarter_exclusions),
            "requested_metric": query_spec.requested_metric,
        },
        "final_answer": final_answer,
        "included_funds": list(included),
        "excluded_funds": list(excluded),
        "hard_stopped": hard_stopped,
        "hard_stop_reason": hard_stop_reason,
        "decision_trace": [asdict(entry) for entry in trace],
    }


# --- Phase 12 stopping-condition demos ----------------------------------------


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, if present. See field_extraction.py's copy."""
    import os

    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _print_result(result: dict) -> None:
    print(f"Question: {result['question']}")
    print(f"QuerySpec: {result['query_spec']}")
    print(f"\nIncluded ({len(result['included_funds'])}):")
    for fund in result["included_funds"]:
        print(f"  - {fund['name']}: expense_ratio={fund['expense_ratio']}, aum={fund['aum']}")
    print(f"\nExcluded ({len(result['excluded_funds'])}):")
    for fund in result["excluded_funds"]:
        print(f"  - {fund['name']}: {fund['reason']}")
    print(f"\nHard-stopped: {result['hard_stopped']}")
    if result["hard_stop_reason"]:
        print(f"Hard-stop reason: {result['hard_stop_reason']}")
    print(f"Final answer: {result['final_answer']}")
    print(f"\nDecision trace ({len(result['decision_trace'])} entries):")
    for entry in result["decision_trace"]:
        print(f"  [{entry['step']}] {entry['detail']}")


def _run_phase_12_clean_demo() -> None:
    """Run one clean pass: the original example question against the real, uncorrupted synthetic set."""
    print("=" * 70)
    print("CLEAN RUN (expect zero RETRY entries)")
    print("=" * 70)
    question = "weighted average expense ratio for ESG funds, excluding funds closed in Q3"
    result = run_agentic_pipeline(question)
    _print_result(result)

    retry_count = sum(1 for e in result["decision_trace"] if e["step"] == "RETRY")
    print(f"\nRETRY entries in this run: {retry_count} (expected 0)")


def _run_phase_12_broken_fund_demo() -> None:
    """Run against a deliberately broken fund: forces a retry every attempt, then an exclusion.

    Corrupts Evergreen ESG Equity Fund's *text* (not its PDF) after Phase 3
    scoping already ran on the real, uncorrupted PDF - so scope_documents
    still correctly includes it as a candidate (matching what a real user
    would see: the document looked fine enough to be scoped in, and only
    fails once someone actually tries to read the number out of it). The
    expense ratio line is rewritten to an impossible 999.00% - real enough
    text that extract_fund_fields will faithfully (and correctly) extract
    999.0, which then fails check_extraction_quality's bounds check on
    every attempt, since the underlying text never changes between
    retries - proving retry doesn't paper over a genuinely bad input, and
    that the loop terminates via the hard MAX_RETRY_ATTEMPTS cap rather than
    retrying forever.
    """
    print("\n" + "=" * 70)
    print("BROKEN-FUND RUN (expect 3 RETRY entries then 1 EXCLUDE, for one fund)")
    print("=" * 70)

    project_root = Path(__file__).resolve().parents[1]
    target_name = "Evergreen ESG Equity Fund"
    target_pdf = project_root / "data" / "raw_pdfs" / "F001_evergreen_esg_equity.pdf"
    real_text = extract_pdf_content(target_pdf)["text"]
    corrupted_text = real_text.replace("0.45%", "999.00%")
    if corrupted_text == real_text:
        raise RuntimeError(
            f"Expected to find '0.45%' in {target_pdf.name}'s text to corrupt - "
            "the synthetic fixture data may have changed."
        )

    question = "weighted average expense ratio for ESG funds, excluding funds closed in Q3"
    result = run_agentic_pipeline(question, _text_override={target_name: corrupted_text})
    _print_result(result)

    retry_count = sum(
        1 for e in result["decision_trace"] if e["step"] == "RETRY" and e["detail"].startswith(target_name)
    )
    excluded_names = [fund["name"] for fund in result["excluded_funds"]]
    print(f"\n{target_name} RETRY entries: {retry_count} (expected {MAX_RETRY_ATTEMPTS})")
    print(f"{target_name} excluded: {target_name in excluded_names} (expected True)")


def _run_phase_12_demo() -> None:
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    import os

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file), then re-run `python -m src.agent_orchestrator`."
        )

    _run_phase_12_clean_demo()
    _run_phase_12_broken_fund_demo()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _run_phase_12_demo()
