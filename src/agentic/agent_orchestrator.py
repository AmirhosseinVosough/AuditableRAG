"""Phase 12: the agentic orchestrator - a real REASON -> ACT -> EVALUATE -> RETRY loop.

Per the core architectural mandate (AGENTIC_LAYER_BUILD_PROMPT.md): this
module contains NO filtering logic and NO arithmetic of its own. Every
actual decision about "does this fund qualify" (Phase 3, via
`retrieval.scope_documents`), "is this extracted value good" (the amended
Phase 5 quality check, `verification.check_extraction_quality`), "is a
failed value borderline or obviously garbage" (Phase 13, via
`verification.is_borderline_quality_failure`), and "what's the final number"
(Phase 6, `calculator.weighted_average_expense_ratio`) is made by calling
the existing deterministic tools, unchanged. What this module owns is purely
the orchestration: which tool to call next, whether to retry, when to give
up, which bucket a give-up lands in, and writing down *why* for every one of
those decisions - never the underlying judgment itself.

This module now covers two source modes, dispatched by `run_agentic_pipeline`'s
`source` parameter, the same way `pipeline.py`'s `run_pipeline` dispatches
between `_run_synthetic_pipeline`/`_run_real_pipeline` (Phase 9b):

    source="synthetic" (default) -> `_run_agentic_synthetic` - Phases 3/4/5/6,
        unchanged, the loop structure described below.
    source="real" -> `_run_agentic_real` (Phase 16) - the same REASON/ACT/
        EVALUATE/RETRY/ESCALATE vocabulary, but ACT-scope goes through
        `real_classifier.classify_esg_status`/`filter_real_funds` (cheap,
        narrowed, small model) instead of regex, and ACT-extract goes
        through `extraction_cascade.extract_with_cascade` instead of
        `field_extraction.extract_fund_fields` - see that function's own
        docstring for the full design, including why a clean-but-uncertain
        cascade result is escalated immediately rather than retried (the
        cascade already exhausted its own internal fallback tiers before
        ever returning, so an outer retry on a clean result would almost
        certainly reproduce the identical answer for extra cost).

Loop structure (REASON -> ACT -> EVALUATE -> RETRY -> ACT -> ANSWER),
`_run_agentic_synthetic`'s version - see `_run_agentic_real`'s own docstring
for how each step differs on the real-data path:

    REASON   - query_parser.parse_query(question) -> QuerySpec
    ACT      - retrieval.scope_documents(query_spec, ...) -> candidate names,
               then field_extraction.extract_fund_fields(...) per candidate
    EVALUATE - verification.check_extraction_quality(...) per result
    RETRY    - a fund that fails EVALUATE is re-extracted, up to
               MAX_RETRY_ATTEMPTS (a real loop counter with a hard bound in
               code - see the `while attempt < MAX_RETRY_ATTEMPTS` loop
               below, not just this docstring)
    (give up) - still failing after the cap: `is_borderline_quality_failure`
               (Phase 13) decides which of two outcomes -
                 - borderline (near-miss, name matched): ESCALATE, into
                   `needs_human_review` with a logged reason
                 - obvious garbage (wildly off, or a name mismatch): EXCLUDE,
                   into `excluded_funds` with a logged reason
               the rest of the batch continues either way - UNLESS
               excluded + needs_human_review together would exceed
               MAX_EXCLUDED_FRACTION of the candidate set, in which case the
               whole run hard-stops (mirrors v1's "never silently proceed on
               bad data" principle - see pipeline.py's Phase 5 gate for the
               same idea applied differently)
    ACT      - calculator.weighted_average_expense_ratio(...) on the final
               verified set (included_funds only - needs_human_review funds
               are excluded from the calculation exactly like excluded_funds
               are, pending a human's resolution)
    ANSWER   - return the number plus the full decision trace: every retry,
               exclusion, escalation, and reason, in order
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from groq import Groq

from src.calculator import weighted_average_expense_ratio
from src.cascade.extraction_cascade import extract_with_cascade
from src.extraction.field_extraction import DEFAULT_MODEL, ExtractedFields, extract_fund_fields
from src.extraction.real_data_loader import load_real_pdfs
from src.fund_filter import FilterSpec, parse_fund_metadata
from src.extraction.pdf_extraction import extract_pdf_content
from src.agentic.query_parser import QuerySpec, parse_query
from src.agentic.real_classifier import RealFundMetadata, classify_esg_status, filter_real_funds
from src.agentic.retrieval import scope_documents
from src.verification import check_extraction_quality, is_borderline_quality_failure


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
# Own name, not pipeline.py's DEFAULT_REAL_DATA_DIR - this module's existing
# DEFAULT_RAW_PDF_DIR already means "the synthetic dir", so reusing that
# other module's name here (same value, data/user_uploads/) would collide
# in meaning even though the path happens to match.
DEFAULT_REAL_RAW_PDF_DIR = PROJECT_ROOT / "data" / "user_uploads"
DEFAULT_RUN_LOG_DIR = PROJECT_ROOT / "outputs" / "run_logs"

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

# Phase 16: a SEPARATE, smaller retry cap for the real-data path's two
# LLM-boundary calls (classify_esg_status, extract_with_cascade) -
# deliberately not MAX_RETRY_ATTEMPTS reused. Both calls are deterministic
# (temperature=0, fixed seed), and extract_with_cascade in particular
# already exhausts its own internal fallback tiers (regex -> BM25 ->
# semantic -> LLM -> narrowing-miss retry -> table-data -> OCR) before ever
# returning - retrying the *whole* call again up to MAX_RETRY_ATTEMPTS times
# on a clean-but-uncertain result would almost certainly reproduce the
# identical answer (same regex hit, same page, same LLM answer) for up to
# 3x the cost and zero realistic chance of a different outcome. This cap
# exists only for a genuine infrastructure failure (a network error, a file
# I/O error) from the call itself - never for a clean result the cascade
# already flagged needs_human_review on, which is escalated immediately
# instead (see _run_agentic_real).
MAX_REAL_INFRA_RETRY_ATTEMPTS = 2


# --- Phase 14: LLM call recording, for the audit trail ----------------------
#
# query_parser.parse_query and field_extraction.extract_fund_fields are
# unchanged by this - per the mandate, they're called as tools, not
# modified. Instead this wraps the *client* object the orchestrator already
# owns and passes down to both: every LLM call site in this project calls
# `active_client.chat.completions.create(...)`, so intercepting that one
# method on the shared client captures every call transparently, tagged
# with which orchestrator step made it, without either module knowing this
# wrapper exists.
class _LLMCallRecorder:
    """Monkeypatches one Groq client's `chat.completions.create` to log every raw request/response.

    `current_label` is set by the orchestrator immediately before each known
    call site (REASON's parse_query, each candidate's extract_fund_fields
    attempt) so every recorded call can be attributed to the step that made
    it - the wrapper itself has no way to know that on its own.

    Never changes call behavior: a failed call is logged (with `error` set,
    `usage`/`raw_response_message` None) and then re-raised exactly as the
    real client raised it, so `model_fallback.call_with_model_fallback`'s
    existing retry/fallback logic is completely unaffected by this wrapper
    being present.
    """

    def __init__(self, client: Groq) -> None:
        self._client = client
        self.calls: list[dict[str, object]] = []
        self.current_label = "unlabeled"
        self._original_create = client.chat.completions.create
        client.chat.completions.create = self._recording_create  # type: ignore[method-assign]

    def _recording_create(self, **kwargs: Any):
        try:
            response = self._original_create(**kwargs)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - log every failure, then always re-raise unchanged
            self.calls.append(
                {
                    "label": self.current_label,
                    "model": kwargs.get("model"),
                    "request_messages": kwargs.get("messages"),
                    "error": str(exc),
                    "raw_response_message": None,
                    "usage": None,
                }
            )
            raise

        message = response.choices[0].message
        usage = response.usage
        self.calls.append(
            {
                "label": self.current_label,
                "model": kwargs.get("model"),
                "request_messages": kwargs.get("messages"),
                "error": None,
                "raw_response_message": message.model_dump(),
                "usage": usage.model_dump() if usage is not None else None,
            }
        )
        return response


# Snapshot pricing, $/1M tokens (input, output) - Groq's own published
# rates, confirmed against two independent sources both citing Groq's
# pricing page directly (https://www.cloudzero.com/blog/groq-pricing/,
# "pulled directly from [Groq's pricing page] as of April 2026"), checked
# 2026-08-24. Not fetched live per-run: Groq has no pricing API, and rates
# can change without notice, so this is a point-in-time estimate for the
# audit trail, not a billing-accurate figure - revisit if these models'
# listed prices change.
_MODEL_PRICING_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


def _total_tokens(calls: Sequence[dict[str, object]]) -> int:
    """Sum `total_tokens` across every call that has usage data (i.e. every call that succeeded)."""
    total = 0
    for call in calls:
        usage = call.get("usage")
        if usage:
            total += usage["total_tokens"]  # type: ignore[index]
    return total


def _estimate_cost_usd(calls: Sequence[dict[str, object]]) -> float | None:
    """Sum estimated USD cost across every successful call, or None if any call's model has no known price.

    Returns None rather than a silently-partial total if any call used a
    model missing from `_MODEL_PRICING_PER_MILLION_TOKENS` - an incomplete
    number that looks complete would be worse than admitting the estimate
    can't be computed, matching this project's "never guess" rule.
    """
    total = 0.0
    for call in calls:
        usage = call.get("usage")
        if not usage:
            continue  # failed call - nothing was billed
        pricing = _MODEL_PRICING_PER_MILLION_TOKENS.get(call.get("model"))  # type: ignore[arg-type]
        if pricing is None:
            return None
        input_rate, output_rate = pricing
        total += usage["prompt_tokens"] / 1_000_000 * input_rate  # type: ignore[index]
        total += usage["completion_tokens"] / 1_000_000 * output_rate  # type: ignore[index]
    return total


@dataclass(frozen=True)
class DecisionTraceEntry:
    """One step of the ReAct loop's reasoning, in order. `detail` is always an explicit reason, never just a status code."""

    step: str  # "REASON" | "ACT" | "EVALUATE" | "RETRY" | "EXCLUDE" | "ESCALATE" | "HARD-STOP" | "ANSWER"
    detail: str


def run_agentic_pipeline(
    question: str,
    *,
    source: Literal["synthetic", "real"] = "synthetic",
    raw_pdf_dir: Path | None = None,
    run_log_dir: Path = DEFAULT_RUN_LOG_DIR,
    client: Groq | None = None,
    model: str = DEFAULT_MODEL,
    _text_override: dict[str, str] | None = None,
) -> dict:
    """Dispatch to the synthetic or real-data agentic loop - Phase 16's entry point, per source.

    Contains no orchestration logic of its own - mirrors `pipeline.py`'s
    `run_pipeline`, which solved the identical "two genuinely different code
    paths under one name" problem for Phase 9b. `source="synthetic"`
    (default) behaves exactly as this function always has -
    `_run_agentic_synthetic` is today's entire loop body, renamed, otherwise
    byte-for-byte unchanged. `source="real"` calls `_run_agentic_real`
    (Phase 16) - see that function's docstring for how ACT-scope/ACT-extract
    differ on real documents.

    Args:
        question: Natural-language question. Parsed into a QuerySpec via
            `query_parser.parse_query` - shared, unmodified, by both paths.
        source: "synthetic" reads `raw_pdf_dir` (default `DEFAULT_RAW_PDF_DIR`)
            via Phase 2/3. "real" reads `raw_pdf_dir` (default
            `DEFAULT_REAL_RAW_PDF_DIR`) via `real_data_loader.load_real_pdfs`.
        raw_pdf_dir: Directory of PDFs to read. `None` (default) resolves to
            the correct default for *source* - pass explicitly to override
            either. This default changing from a concrete path to `None` is
            a signature-shape change only: every existing synthetic caller
            that never passed `raw_pdf_dir` gets byte-for-byte the same
            resolved value as before.
        run_log_dir: Directory the timestamped JSON audit record (Phase 14)
            is written to - same home `pipeline.py`'s Phase 7 run log uses
            (`outputs/run_logs/`), distinguished by filename prefix.
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
        A dict: {timestamp, question, query_spec, final_answer,
        included_funds, excluded_funds, needs_human_review, hard_stopped,
        hard_stop_reason, decision_trace, llm_calls, total_tokens,
        estimated_cost_usd}. `final_answer` is None if nothing qualified or
        the run hard-stopped. `needs_human_review` (Phase 13) holds funds
        that exhausted every retry but were a borderline-plausible
        near-miss rather than obvious garbage - see
        `is_borderline_quality_failure` - kept separate from
        `excluded_funds` so a run-log reader can tell "the agent is
        confident this is bad data" apart from "the agent couldn't decide,
        a human should". `decision_trace` is the full ordered log of every
        REASON/ACT/EVALUATE/RETRY/EXCLUDE/ESCALATE/HARD-STOP/ANSWER step,
        each with an explicit reason string - a human should be able to
        reconstruct exactly what happened and why using only this list.
        `llm_calls` (Phase 14) is the raw request/response for every LLM
        call made during this run, in order, each tagged with which step
        made it (see `_LLMCallRecorder`) - including calls that failed
        outright (`error` set) and calls a fallback model made after the
        primary model failed. `total_tokens`/`estimated_cost_usd` are
        summed from `llm_calls`' usage data; `estimated_cost_usd` is None
        if any call used a model with no known price (see
        `_MODEL_PRICING_PER_MILLION_TOKENS`) - never a silently-partial
        number. This full dict is also written to `run_log_dir` as JSON
        before returning (or before a hard-stop return), so a run is
        reconstructable from disk without re-running anything - Phase 14's
        actual goal.

    Raises:
        ValueError: If `source` is neither "synthetic" nor "real".
    """
    if source == "synthetic":
        return _run_agentic_synthetic(
            question,
            raw_pdf_dir=raw_pdf_dir if raw_pdf_dir is not None else DEFAULT_RAW_PDF_DIR,
            run_log_dir=run_log_dir,
            client=client,
            model=model,
            _text_override=_text_override,
        )
    elif source == "real":
        return _run_agentic_real(
            question,
            raw_pdf_dir=raw_pdf_dir if raw_pdf_dir is not None else DEFAULT_REAL_RAW_PDF_DIR,
            run_log_dir=run_log_dir,
            client=client,
            model=model,
        )
    else:
        raise ValueError(f"Unknown source {source!r}; expected 'synthetic' or 'real'")


def _run_agentic_synthetic(
    question: str,
    *,
    raw_pdf_dir: Path,
    run_log_dir: Path,
    client: Groq | None,
    model: str,
    _text_override: dict[str, str] | None,
) -> dict:
    """Phase 12/13/14's original loop body - unchanged, just renamed for Phase 16's dispatch.

    See `run_agentic_pipeline`'s docstring for the full contract (args,
    return shape). This function's own body is untouched from before Phase
    16 - it does not know `_run_agentic_real` exists.
    """
    trace: list[DecisionTraceEntry] = []

    def log(step: str, detail: str) -> None:
        trace.append(DecisionTraceEntry(step=step, detail=detail))
        logger.info("[%s] %s", step, detail)

    active_client = client or Groq()
    recorder = _LLMCallRecorder(active_client)

    # --- REASON: turn the question into structure. No filtering logic here -
    # parse_query either returns a QuerySpec or raises UnsupportedQueryError;
    # the orchestrator doesn't second-guess either outcome. ---
    recorder.current_label = "REASON: parse_query"
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
    needs_human_review: list[dict[str, object]] = []

    for name in candidate_names:
        text = name_to_text[name]
        resolved: ExtractedFields | None = None
        # Phase 13: the last attempt's fields survive even a failed quality
        # check, so a fund that never passes still has something to classify
        # as borderline-vs-garbage after the retry loop gives up on it. None
        # only if every attempt's *call* failed outright (no fields exist).
        last_extracted: ExtractedFields | None = None
        last_problems: tuple[str, ...] = ()

        attempt = 0
        while attempt < MAX_RETRY_ATTEMPTS:  # hard cap enforced here, in code - not just documented
            attempt += 1
            recorder.current_label = f"ACT: extract_fund_fields({name}, attempt {attempt}/{MAX_RETRY_ATTEMPTS})"
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

            last_extracted = extracted
            last_problems = problems
            log(
                "RETRY",
                f"{name}: attempt {attempt}/{MAX_RETRY_ATTEMPTS} failed EVALUATE - {'; '.join(problems)}",
            )

        if resolved is not None:
            included.append({"name": name, "expense_ratio": resolved.expense_ratio, "aum": resolved.aum})
        elif last_extracted is not None and is_borderline_quality_failure(
            expected_name=name,
            extracted_name=last_extracted.fund_name,
            expense_ratio=last_extracted.expense_ratio,
            aum=last_extracted.aum,
        ):
            # Phase 13: a near-miss, not obvious garbage - surface it for a
            # human instead of silently discarding it like a clean exclude.
            reason = (
                f"exhausted {MAX_RETRY_ATTEMPTS} attempts, still failing EVALUATE but within a "
                f"plausible margin (last failure: {'; '.join(last_problems)}) - needs a human's "
                "judgment call, not an automatic decision"
            )
            log("ESCALATE", f"{name}: {reason}")
            needs_human_review.append(
                {
                    "name": name,
                    "reason": reason,
                    "expense_ratio": last_extracted.expense_ratio,
                    "aum": last_extracted.aum,
                }
            )
        else:
            reason = (
                f"excluded after {MAX_RETRY_ATTEMPTS} failed attempts (see RETRY entries above)"
                if last_extracted is not None
                else f"excluded after {MAX_RETRY_ATTEMPTS} failed attempts - every extraction call failed outright"
            )
            log("EXCLUDE", f"{name}: {reason}")
            excluded.append({"name": name, "reason": reason})

    # --- hard-stop check: mirrors v1's "never silently proceed on bad data" ---
    # needs_human_review counts alongside excluded here - a fund awaiting a
    # human's judgment is just as absent from the final calculation as an
    # excluded one, and the hard-stop exists to catch "too much of the batch
    # never produced a usable value," not specifically "too much was excluded."
    if candidate_names:
        unresolved_count = len(excluded) + len(needs_human_review)
        unresolved_fraction = unresolved_count / len(candidate_names)
        if unresolved_fraction > MAX_EXCLUDED_FRACTION:
            hard_stop_reason = (
                f"{unresolved_count}/{len(candidate_names)} candidates ({unresolved_fraction:.0%}) "
                f"ended up excluded or needing human review after exhausting retries, exceeding the "
                f"{MAX_EXCLUDED_FRACTION:.0%} threshold - hard-stopping rather than compute an answer "
                "over too incomplete a set"
            )
            log("HARD-STOP", hard_stop_reason)
            result = _build_result(
                question=question,
                query_spec=query_spec,
                final_answer=None,
                included=included,
                excluded=excluded,
                needs_human_review=needs_human_review,
                hard_stopped=True,
                hard_stop_reason=hard_stop_reason,
                trace=trace,
                llm_calls=recorder.calls,
            )
            _write_agentic_run_log(result, run_log_dir)
            return result

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

    result = _build_result(
        question=question,
        query_spec=query_spec,
        final_answer=final_answer,
        included=included,
        excluded=excluded,
        needs_human_review=needs_human_review,
        hard_stopped=False,
        hard_stop_reason=None,
        trace=trace,
        llm_calls=recorder.calls,
    )
    _write_agentic_run_log(result, run_log_dir)
    return result


def _real_metadata_filter_reason(metadata: RealFundMetadata, filter_spec: FilterSpec) -> str:
    """Human-readable reason a confidently-classified real document fails *filter_spec*.

    Mirrors `pipeline.py`'s `_synthetic_filter_reason` (same is_esg/status
    phrasing), over `RealFundMetadata` instead of `FundMetadata`. Only
    called once a candidate's `flags` are empty - a confident classification
    that simply doesn't match the query, not an undetermined one (those use
    `flags` directly as the reason instead - see `_run_agentic_real`).
    """
    reasons = []
    if filter_spec.is_esg is not None and metadata.is_esg != filter_spec.is_esg:
        reasons.append("is ESG" if metadata.is_esg else "not ESG")
    if filter_spec.status is not None and metadata.status != filter_spec.status.lower():
        reasons.append(f"status is {metadata.status}")
    return " and ".join(reasons) if reasons else "did not match the query's is_esg/status constraints"


def _run_agentic_real(
    question: str,
    *,
    raw_pdf_dir: Path,
    run_log_dir: Path,
    client: Groq | None,
    model: str,
) -> dict:
    """Phase 16: the real-data agentic loop - same REASON/ACT/EVALUATE/RETRY/ESCALATE vocabulary, real documents.

    Fully self-contained (own `trace`/`log`/`active_client`/`recorder`), not
    a thinner shared body with `_run_agentic_synthetic` - same precedent
    `pipeline.py`'s `_run_synthetic_pipeline`/`_run_real_pipeline` already
    set for this project. `retrieval.py`, `real_classifier.py`,
    `extraction_cascade.py`, `verification.py`, and `calculator.py` are all
    reused exactly as they exist today - only this module gains new code.

    Two-stage ACT, cheap before expensive:

    ACT-scope (cheap): `real_data_loader.load_real_pdfs` -> per-document
        `extract_pdf_content` -> `real_classifier.classify_esg_status`
        (small model, BM25/semantic-narrowed) -> `real_classifier.filter_real_funds`.
        Deliberately NOT `retrieval.scope_real_documents` as one batched
        call - that function lets one candidate's classification failure
        abort the whole batch (see its own docstring); this loop isolates
        each document's classification in its own try/except instead, so
        one unreadable or unclassifiable document never takes the rest of
        the batch down with it.

        `classify_esg_status` is deliberately called WITHOUT `model=model` -
        it keeps its own `CLASSIFIER_MODEL` default (the smaller model this
        cheap stage exists to use), not this orchestrator's `model` param,
        which is reserved for the expensive ACT-extract stage below. Easy to
        get backwards by reflexively threading `model` everywhere.

        Candidate identity is the PDF filename/stem throughout, not the real
        fund name - no reliable name exists until the expensive cascade
        below extracts one (which may itself come back null on a messy
        document). Once known, it's carried as a separate `display_name`
        field on `included_funds`/`needs_human_review` entries, never used
        as the key.

        `extract_pdf_content` is called here for `classify_esg_status`'s
        `pages`, and `extract_with_cascade` below parses the same PDF again
        internally - a real, accepted double-parse. Left as-is rather than
        threading pre-extracted pages into `extraction_cascade.py` (which
        would mean touching that module); pdfplumber parsing a few-page
        fact sheet is milliseconds next to the LLM calls in the same flow.

    ACT-extract (expensive), per candidate that survived the cheap filter:
        `extraction_cascade.extract_with_cascade`. This call is already
        deterministic (temperature=0, fixed seed) and already exhausts its
        own internal fallback tiers (regex -> BM25 -> semantic -> LLM ->
        narrowing-miss retry -> table-data -> OCR) before ever returning -
        so unlike the synthetic path's extract_fund_fields, a clean-but-
        uncertain result (`CascadeResult.needs_human_review=True`) is NEVER
        retried here. Retrying it would almost certainly reproduce the
        identical answer (same regex hit, same page, same LLM answer) for
        extra cost and zero realistic chance of a different outcome. The
        bounded `MAX_REAL_INFRA_RETRY_ATTEMPTS` retry loop exists only for a
        genuine exception from the call itself (a network error, a file I/O
        failure) - a real infrastructure failure, not a judgment call.

    EVALUATE: read `CascadeResult.needs_human_review` directly - no new
        function, no re-derived judgment. `False` -> `included_funds`
        (`CascadeResult`'s own final block guarantees `expense_ratio`/`aum`
        are both non-null floats whenever `needs_human_review` is `False`).
        `True` -> straight to ESCALATE (Phase 13's existing mechanism, new
        trigger condition), with citations built from
        `CascadeResult.resolutions`' `SourceLocation`s - carried straight
        through, not re-plumbed.

    Hard-stop: `(excluded-after-scoping + needs_human_review) / candidates
        > MAX_EXCLUDED_FRACTION`, same constant as the synthetic path, but
        the numerator counts ONLY post-scope failures/escalations, tracked
        separately from `len(excluded_funds)` - `excluded_funds` here also
        holds cheap-stage exclusions (documents that never became
        candidates at all), which must NOT count toward this fraction, or
        the threshold would be measuring against the wrong denominator
        (unlike the synthetic path, which never reports its own Phase-3
        scoping exclusions in `excluded_funds` at all, so no such split was
        needed there).
    """
    trace: list[DecisionTraceEntry] = []

    def log(step: str, detail: str) -> None:
        trace.append(DecisionTraceEntry(step=step, detail=detail))
        logger.info("[%s] %s", step, detail)

    active_client = client or Groq()
    recorder = _LLMCallRecorder(active_client)

    # --- REASON: identical to the synthetic path - shared vocabulary, not a
    # real-data-specific step. ---
    recorder.current_label = "REASON: parse_query"
    query_spec: QuerySpec = parse_query(question, client=active_client, model=model)
    log("REASON", f"parsed question into {query_spec}")

    # --- ACT-scope (cheap): load -> read -> classify -> filter, per document,
    # each document's classification isolated from the rest of the batch. ---
    pdf_paths = load_real_pdfs(Path(raw_pdf_dir))
    identity_to_path = {pdf_path.stem: pdf_path for pdf_path in pdf_paths}

    excluded_funds: list[dict[str, object]] = []
    needs_human_review: list[dict[str, object]] = []
    included_funds: list[dict[str, object]] = []

    classified: list[tuple[str, RealFundMetadata]] = []
    for pdf_path in pdf_paths:
        identity = pdf_path.stem

        try:
            # extract_with_cascade will parse this same PDF again internally
            # for ACT-extract - accepted, not fixed (see this function's
            # docstring's "double-parse" paragraph).
            pages = cast(list[str], extract_pdf_content(pdf_path)["pages"])
        except Exception as exc:  # noqa: BLE001 - one unreadable document isolated from the rest
            reason = f"could not be read: {exc}"
            log("EXCLUDE", f"{identity}: {reason}")
            excluded_funds.append({"name": identity, "reason": reason})
            continue

        metadata: RealFundMetadata | None = None
        last_exc: Exception | None = None
        attempt = 0
        while attempt < MAX_REAL_INFRA_RETRY_ATTEMPTS:  # hard cap enforced here, in code - not just documented
            attempt += 1
            recorder.current_label = (
                f"ACT-scope: classify_esg_status({identity}, attempt {attempt}/{MAX_REAL_INFRA_RETRY_ATTEMPTS})"
            )
            try:
                # No model=model here - deliberately keeps classify_esg_status's
                # own smaller CLASSIFIER_MODEL default. See this function's docstring.
                metadata = classify_esg_status(identity, pages, question, client=active_client)
                break
            except Exception as exc:  # noqa: BLE001 - infra-failure isolation, one document at a time
                last_exc = exc
                log(
                    "RETRY",
                    f"{identity}: attempt {attempt}/{MAX_REAL_INFRA_RETRY_ATTEMPTS} - "
                    f"classify_esg_status call failed: {exc}",
                )

        if metadata is None:
            reason = f"couldn't determine ESG/active status - every classification attempt failed: {last_exc}"
            log("EXCLUDE", f"{identity}: {reason}")
            excluded_funds.append({"name": identity, "reason": reason})
            continue

        classified.append((identity, metadata))

    # filter_real_funds never raises (pure comparison over already-classified
    # metadata) - built inline rather than via retrieval._query_spec_to_filter_spec,
    # which is a private helper there; duplicating this one-line translation
    # beats coupling two already-built modules together, the same tradeoff
    # real_classifier.py's own _coerce_optional docstring already made.
    filter_spec = FilterSpec(is_esg=query_spec.is_esg, status=query_spec.status)
    candidate_names = filter_real_funds([metadata for _, metadata in classified], filter_spec)
    candidate_set = set(candidate_names)

    for identity, metadata in classified:
        if identity in candidate_set:
            continue
        reason = "; ".join(metadata.flags) if metadata.flags else _real_metadata_filter_reason(metadata, filter_spec)
        log("EXCLUDE", f"{identity}: {reason}")
        excluded_funds.append({"name": identity, "reason": reason})

    log(
        "ACT",
        f"classify_esg_status + filter_real_funds selected {len(candidate_names)} candidate(s) out of "
        f"{len(pdf_paths)} document(s): {candidate_names}",
    )

    # --- ACT-extract (expensive) + EVALUATE, per candidate that survived the cheap scope ---
    # Tracked separately from len(excluded_funds) - see this function's
    # docstring's "Hard-stop" paragraph for why that split is required here.
    post_scope_unresolved_count = 0

    for name in candidate_names:
        pdf_path = identity_to_path[name]
        cascade_result: Any | None = None
        last_exc = None
        attempt = 0
        while attempt < MAX_REAL_INFRA_RETRY_ATTEMPTS:  # hard cap enforced here, in code - not just documented
            attempt += 1
            recorder.current_label = (
                f"ACT-extract: extract_with_cascade({name}, attempt {attempt}/{MAX_REAL_INFRA_RETRY_ATTEMPTS})"
            )
            try:
                cascade_result = extract_with_cascade(pdf_path, question, client=active_client, model=model)
                break
            except Exception as exc:  # noqa: BLE001 - infra-failure only; a clean-but-uncertain result is NEVER retried (see docstring)
                last_exc = exc
                log(
                    "RETRY",
                    f"{name}: attempt {attempt}/{MAX_REAL_INFRA_RETRY_ATTEMPTS} - "
                    f"extract_with_cascade call failed: {exc}",
                )

        if cascade_result is None:
            reason = (
                f"excluded after {MAX_REAL_INFRA_RETRY_ATTEMPTS} failed attempts - "
                f"extract_with_cascade failed outright every time: {last_exc}"
            )
            log("EXCLUDE", f"{name}: {reason}")
            excluded_funds.append({"name": name, "reason": reason})
            post_scope_unresolved_count += 1
            continue

        fields = cascade_result.fields
        display_name = fields.fund_name or name

        if cascade_result.needs_human_review:
            # Phase 13's existing mechanism, new trigger condition - no
            # retry, per this function's docstring. Citations carried
            # straight from the cascade's own resolutions, not re-derived.
            reason = "; ".join(cascade_result.review_reasons) or "cascade flagged this result for human review"
            citations = {
                field_name: {
                    "file": resolution.source.file,
                    "page": resolution.source.page,
                    "snippet": resolution.source.snippet,
                    "resolved_by": resolution.resolved_by,
                }
                for field_name, resolution in cascade_result.resolutions.items()
                if resolution.source is not None
            }
            log("ESCALATE", f"{name}: {reason}")
            needs_human_review.append(
                {
                    "name": name,
                    "display_name": display_name,
                    "reason": reason,
                    "expense_ratio": fields.expense_ratio,
                    "aum": fields.aum,
                    "citations": citations,
                }
            )
            post_scope_unresolved_count += 1
            continue

        resolution_summary = ", ".join(f"{field}={r.resolved_by}" for field, r in cascade_result.resolutions.items())
        log("EVALUATE", f"{name}: cascade resolved cleanly ({resolution_summary})")
        included_funds.append(
            {
                "name": name,
                "display_name": display_name,
                "expense_ratio": fields.expense_ratio,
                "aum": fields.aum,
            }
        )

    # --- hard-stop check: same constant as the synthetic path, denominator
    # is the cheap-scope survivor count, numerator is post-scope failures
    # only (see this function's docstring) ---
    if candidate_names:
        unresolved_fraction = post_scope_unresolved_count / len(candidate_names)
        if unresolved_fraction > MAX_EXCLUDED_FRACTION:
            hard_stop_reason = (
                f"{post_scope_unresolved_count}/{len(candidate_names)} candidates ({unresolved_fraction:.0%}) "
                f"ended up excluded or needing human review after surviving the cheap ESG/status filter, "
                f"exceeding the {MAX_EXCLUDED_FRACTION:.0%} threshold - hard-stopping rather than compute an "
                "answer over too incomplete a set"
            )
            log("HARD-STOP", hard_stop_reason)
            result = _build_result(
                question=question,
                query_spec=query_spec,
                final_answer=None,
                included=included_funds,
                excluded=excluded_funds,
                needs_human_review=needs_human_review,
                hard_stopped=True,
                hard_stop_reason=hard_stop_reason,
                trace=trace,
                llm_calls=recorder.calls,
            )
            _write_agentic_run_log(result, run_log_dir, filename_prefix="run_agentic_real")
            return result

    # --- ACT-calculate / ANSWER: Phase 6, unmodified. ---
    final_answer: float | None = None
    if included_funds:
        final_answer = weighted_average_expense_ratio(
            {"expense_ratio": fund["expense_ratio"], "aum": fund["aum"]} for fund in included_funds
        )
        log(
            "ANSWER",
            f"weighted_average_expense_ratio over {len(included_funds)} fund(s) = {final_answer}",
        )
    else:
        log("ANSWER", "no funds ended up included - no answer to compute")

    result = _build_result(
        question=question,
        query_spec=query_spec,
        final_answer=final_answer,
        included=included_funds,
        excluded=excluded_funds,
        needs_human_review=needs_human_review,
        hard_stopped=False,
        hard_stop_reason=None,
        trace=trace,
        llm_calls=recorder.calls,
    )
    _write_agentic_run_log(result, run_log_dir, filename_prefix="run_agentic_real")
    return result


def _build_result(
    *,
    question: str,
    query_spec: QuerySpec,
    final_answer: float | None,
    included: Sequence[dict[str, object]],
    excluded: Sequence[dict[str, object]],
    needs_human_review: Sequence[dict[str, object]],
    hard_stopped: bool,
    hard_stop_reason: str | None,
    trace: Sequence[DecisionTraceEntry],
    llm_calls: Sequence[dict[str, object]],
) -> dict:
    """Assemble the final dict return value - the single place its shape is defined."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
        "needs_human_review": list(needs_human_review),
        "hard_stopped": hard_stopped,
        "hard_stop_reason": hard_stop_reason,
        "decision_trace": [asdict(entry) for entry in trace],
        "llm_calls": list(llm_calls),
        "total_tokens": _total_tokens(llm_calls),
        "estimated_cost_usd": _estimate_cost_usd(llm_calls),
    }


def _write_agentic_run_log(result: dict, run_log_dir: Path, *, filename_prefix: str = "run_agentic") -> Path:
    """Write *result* as a timestamped JSON file under *run_log_dir* and return its path.

    Same run-log home `pipeline.py`'s Phase 7 already uses
    (`outputs/run_logs/`) - Phase 14 extends the audit-trail concept, not
    invents a second home for it. Filename prefix (`run_agentic_` vs.
    Phase 7's `run_{source}_`) keeps the two kinds of run log
    distinguishable in the same directory without colliding.

    `filename_prefix` defaults to preserve `_run_agentic_synthetic`'s exact
    existing filename (`run_agentic_{timestamp}.json`) byte-for-byte;
    `_run_agentic_real` (Phase 16) passes `"run_agentic_real"`.
    """
    run_log_dir = Path(run_log_dir)
    run_log_dir.mkdir(parents=True, exist_ok=True)
    file_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = run_log_dir / f"{filename_prefix}_{file_stamp}.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=False, default=str))
    logger.info("Wrote agentic run log to %s", destination)
    return destination


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
        # display_name/citations only exist on Phase 16's real-data entries -
        # .get(...) keeps this working unchanged for the synthetic path's
        # plain {name, expense_ratio, aum} dicts.
        label = fund.get("display_name") or fund["name"]
        print(f"  - {fund['name']} ({label}): expense_ratio={fund['expense_ratio']}, aum={fund['aum']}")
    print(f"\nExcluded ({len(result['excluded_funds'])}):")
    for fund in result["excluded_funds"]:
        print(f"  - {fund['name']}: {fund['reason']}")
    print(f"\nNeeds human review ({len(result['needs_human_review'])}):")
    for fund in result["needs_human_review"]:
        label = fund.get("display_name") or fund["name"]
        citations = fund.get("citations")
        citation_note = f", citations={citations}" if citations else ""
        print(
            f"  - {fund['name']} ({label}) (expense_ratio={fund['expense_ratio']}, aum={fund['aum']}): "
            f"{fund['reason']}{citation_note}"
        )
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

    project_root = Path(__file__).resolve().parents[2]
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
    _load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    import os

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file), then re-run `python -m src.agentic.agent_orchestrator`."
        )

    _run_phase_12_clean_demo()
    _run_phase_12_broken_fund_demo()


def _run_phase_13_three_bucket_demo() -> None:
    """One run producing a fund in each of the three buckets - Phase 13's stopping condition.

    Two funds are corrupted after Phase 3 scoping already ran on the real,
    uncorrupted PDFs (same technique as `_run_phase_12_broken_fund_demo`),
    each engineered to land in a different terminal bucket:

    - Evergreen ESG Equity Fund: expense ratio rewritten to an impossible
      999.00% - wildly outside any plausible range, so
      `is_borderline_quality_failure` correctly calls it obvious garbage.
      Every retry sees the same corrupted text and fails identically ->
      EXCLUDE.
    - Horizon Sustainable Bond Fund: expense ratio rewritten to 10.50% -
      just 0.5 points past the 10.0 plausible-range bound, well inside the
      2.0-point borderline margin, and its name still matches. A genuine
      near-miss a human could plausibly confirm as real -> ESCALATE, into
      needs_human_review, not silently discarded.

    The other 4 qualifying funds are untouched real PDFs, so they pass
    EVALUATE on attempt 1 -> included. One run, three buckets, no fabricated
    trace entries - every outcome here is the real code path for the input
    it was given.
    """
    print("\n" + "=" * 70)
    print("THREE-BUCKET RUN (expect: included + 1 EXCLUDE + 1 ESCALATE)")
    print("=" * 70)

    project_root = Path(__file__).resolve().parents[2]

    def _read_text(pdf_path: Path) -> str:
        text = extract_pdf_content(pdf_path)["text"]
        assert isinstance(text, str)  # extract_pdf_content always returns "text" as str
        return text

    garbage_name = "Evergreen ESG Equity Fund"
    garbage_pdf = project_root / "data" / "raw_pdfs" / "F001_evergreen_esg_equity.pdf"
    garbage_real_text = _read_text(garbage_pdf)
    garbage_text = garbage_real_text.replace("0.45%", "999.00%")

    borderline_name = "Horizon Sustainable Bond Fund"
    borderline_pdf = project_root / "data" / "raw_pdfs" / "F002_horizon_sustainable_bond.pdf"
    borderline_real_text = _read_text(borderline_pdf)
    borderline_text = borderline_real_text.replace("0.38%", "10.50%")

    for original_pdf, real_text, corrupted_text in (
        (garbage_pdf, garbage_real_text, garbage_text),
        (borderline_pdf, borderline_real_text, borderline_text),
    ):
        if corrupted_text == real_text:
            raise RuntimeError(
                f"Expected text substitution to change {original_pdf.name}'s text - "
                "the synthetic fixture data may have changed."
            )

    question = "weighted average expense ratio for ESG funds, excluding funds closed in Q3"
    result = run_agentic_pipeline(
        question,
        _text_override={garbage_name: garbage_text, borderline_name: borderline_text},
    )
    _print_result(result)

    included_names = [fund["name"] for fund in result["included_funds"]]
    excluded_names = [fund["name"] for fund in result["excluded_funds"]]
    review_names = [fund["name"] for fund in result["needs_human_review"]]

    print(f"\n{garbage_name} excluded: {garbage_name in excluded_names} (expected True)")
    print(f"{borderline_name} needs human review: {borderline_name in review_names} (expected True)")
    print(f"Included count: {len(included_names)} (expected 4 - 6 qualifying minus the 2 corrupted)")


def _run_phase_13_demo() -> None:
    _load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    import os

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file), then re-run `python -m src.agentic.agent_orchestrator`."
        )

    _run_phase_13_three_bucket_demo()


def _run_phase_16_real_demo() -> None:
    """Run against data/user_uploads/'s real fact sheets - Phase 16's stopping condition.

    Unlike Phase 12/13's synthetic demos, nothing is hand-corrupted here -
    this runs the real cascade against real documents exactly as a user
    would encounter them. Whether needs_human_review ends up populated
    (rather than every fund cleanly resolving) depends on the live LLM call
    against these specific real fact sheets, not a fabricated input - not
    guaranteed identical on every run, unlike the synthetic demos' pinned
    corruption. Reported honestly either way, not assumed from this code.
    """
    print("\n" + "=" * 70)
    print("REAL-DATA RUN (data/user_uploads/)")
    print("=" * 70)
    # Matches pipeline.py's own Phase 9 demo question, for direct comparability.
    question = "weighted average expense ratio for ESG active funds"
    result = run_agentic_pipeline(question, source="real")
    _print_result(result)

    # classify_esg_status/extract_with_cascade only appear as labels on
    # llm_calls entries (set via recorder.current_label) - decision_trace's
    # detail text never repeats the function name, so it must be checked
    # there, not against decision_trace.
    scope_calls = [c for c in result["llm_calls"] if "classify_esg_status" in c["label"]]
    extract_calls = [c for c in result["llm_calls"] if "extract_with_cascade" in c["label"]]
    print(f"\nACT-scope (classify_esg_status) llm_calls: {len(scope_calls)}")
    print(f"ACT-extract (extract_with_cascade) llm_calls: {len(extract_calls)}")
    print(f"needs_human_review count: {len(result['needs_human_review'])}")
    print(f"total_tokens: {result['total_tokens']}, estimated_cost_usd: {result['estimated_cost_usd']}")


def _run_phase_16_demo() -> None:
    _load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    import os

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file), then re-run `python -m src.agentic.agent_orchestrator`."
        )

    _run_phase_16_real_demo()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _run_phase_12_demo()
    _run_phase_13_demo()
    _run_phase_16_demo()
