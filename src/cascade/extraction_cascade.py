from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from groq import APIStatusError, Groq, APIConnectionError

from src.cascade import bm25_search, ocr, semantic_search
from src.extraction.field_extraction import DEFAULT_MODEL, RealExtractedFields, extract_real_fund_fields
from src.extraction.pdf_extraction import Table, extract_pdf_content
from src.shared.source_location import SourceLocation


logger = logging.getLogger(__name__)

# Bounded, hard-coded - never left to the model's judgment about when to
# stop, matching this project's retry-cap convention everywhere else.
_MAX_LLM_RETRY_ATTEMPTS = 2


@dataclass(frozen=True)
class FieldResolution:
    value: object
    resolved_by: str
    source: SourceLocation | None


@dataclass(frozen=True)
class CascadeResult:

    fields: RealExtractedFields
    resolutions: dict[str, FieldResolution]
    needs_human_review: bool
    review_reasons: tuple[str, ...]


# --- Tier 1: regex pre-pass (expense_ratio / aum only) -----------------------

_EXPENSE_RATIO_PATTERN = re.compile(
    r"(?:net\s+|gross\s+)?expense\s*ratio[^\n%]{0,40}?(\d+\.\d+)\s*%?",
    re.IGNORECASE,
)

# Looks for a "net assets"/"aum" label followed by a dollar figure and,
# optionally, a scale word. Real documents phrase this wildly differently
# ("Net Assets of Fund (M): $17,746.93", "ETF total net assets $13,178
# million", "Net Assets ($) 577,188,130") - see _parse_aum_value for how
# the scale is (conservatively) resolved.
_AUM_PATTERN = re.compile(
    r"(?:total\s+net\s+assets|net\s+assets(?:\s+of\s+fund)?|assets\s+under\s+management|\baum\b)"
    r"[^\n]{0,30}?\$?\s*([\d,]+\.?\d*)\s*(million|billion|m\b|b\b)?",
    re.IGNORECASE,
)


def _parse_aum_value(number_str: str, unit: str | None) -> float | None:
    """Convert a regex-captured AUM number + optional unit word into millions of USD.

    Deliberately conservative: returns None (no confident conversion, not a
    match) rather than guessing when the unit is ambiguous, instead of
    defaulting to "probably millions". A bare number with no scale cue is
    exactly the kind of thing this project's "never guess" rule exists to
    catch, even in a heuristic pre-pass tier that isn't the final answer.
    """
    try:
        number = float(number_str.replace(",", ""))
    except ValueError:
        return None

    if unit:
        unit_lower = unit.lower().rstrip(".")
        if unit_lower in ("million", "m"):
            return number
        if unit_lower in ("billion", "b"):
            return number * 1000

    # No unit word given. Only trust magnitude when it's unambiguous: a real
    # fund's AUM stated as a bare dollar figure runs into the hundreds of
    # millions or more - i.e. at least 7 digits. Anything smaller with no
    # unit hint is genuinely ambiguous and is skipped rather than guessed.
    if number >= 1_000_000:
        return number / 1_000_000

    return None


def _regex_prepass(file_name: str, pages: list[str]) -> dict[str, tuple[float, SourceLocation]]:
    """Best-effort regex scan for expense_ratio/aum, per page, with real page numbers.

    Never attempts is_esg/status - those require reading for substance, not
    a keyword/number pattern (a document can say "ESG" while explicitly
    stating it does *not* screen on ESG criteria; a regex has no way to
    tell). That stays LLM-only, per Phase 9's own design decision.

    Returns only the fields it actually found - callers must not assume
    both keys are present.
    """
    hits: dict[str, tuple[float, SourceLocation]] = {}

    for page_number, page_text in enumerate(pages, start=1):
        if "expense_ratio" not in hits:
            match = _EXPENSE_RATIO_PATTERN.search(page_text)
            if match:
                try:
                    value = float(match.group(1))
                except ValueError:
                    value = None
                # Sanity bound: real expense ratios are always well under
                # 10% - anything higher almost certainly means the pattern
                # matched an unrelated number (e.g. a return percentage).
                if value is not None and 0 < value < 10:
                    hits["expense_ratio"] = (
                        value,
                        SourceLocation(file=file_name, page=page_number, snippet=match.group(0).strip()),
                    )

        if "aum" not in hits:
            match = _AUM_PATTERN.search(page_text)
            if match:
                value = _parse_aum_value(match.group(1), match.group(2))
                if value is not None:
                    hits["aum"] = (
                        value,
                        SourceLocation(file=file_name, page=page_number, snippet=match.group(0).strip()),
                    )

    return hits



# --- Bounded retry wrapper for LLM calls --------------------------------------

def _extract_with_retry(text: str, *, client: Groq, model: str) -> RealExtractedFields | None:
    """Call extract_real_fund_fields, retrying up to `_MAX_LLM_RETRY_ATTEMPTS` on transient failures.

    Only retries exceptions that plausibly mean "the model glitched this
    time" (malformed JSON, missing tool call, a network/API error)"""
    #Deterministic Error (Broken Code)The Example: You have a typo in your code, like writing clent instead of client.What happens: Every single time the code runs, it will crash. It is a permanent rule. 
    #Retrying 3 times changes nothing; it will fail 3 times.
    #2. Non-Deterministic Behavior (The LLM Surprise)The Example: The Groq LLM reads your text.What happens: Because LLMs are a bit unpredictable, it might give you perfect JSON data on the first try, but on the second try,
      #it might output a malformed, broken sentence. It is a surprise every time
    #.3. Transient Error (The Lightning Flash)The Example: Your internet connection blinks out for half a second while talking to Groq.What happens: The code crashes right now because of the network blink. But a second later, the internet is back. This is a super temporary glitch that a retry loop can fix."""
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_LLM_RETRY_ATTEMPTS + 1):
        try:
            return extract_real_fund_fields(text, client=client, model=model)
        except (APIConnectionError, APIStatusError, ValueError) as exc: 
             # we use it as a way to reduce token usage for the llm when we receive api related errors like:
            #nternet drops out? It matches APIConnectionError. The loop will retry. (Good!)Groq is too busy (Rate limit)? It matches APIStatusError. The loop will retry. (Good!)You made a typo in the model name? This is a developer error. It will not match those exceptions.  
            last_exc = exc
            logger.warning("LLM extraction attempt %d/%d failed: %s", attempt, _MAX_LLM_RETRY_ATTEMPTS, exc)
    logger.error("LLM extraction failed after %d attempts: %s", _MAX_LLM_RETRY_ATTEMPTS, last_exc)
    return None


def _tables_to_text(tables: list[Table]) -> str:
    """Render pdfplumber-detected tables as pipe-separated text lines for a table-focused LLM pass."""
    lines: list[str] = []
    for table in tables:
        for row in table:
            lines.append(" | ".join(cell if cell is not None else "" for cell in row))
    return "\n".join(lines)


# --- Orchestrator -------------------------------------------------------------

def extract_with_cascade(
    pdf_path: Path,
    question: str,
    *,
    client: Groq | None = None,
    model: str = DEFAULT_MODEL,
) -> CascadeResult:
    # runs the full regex -> BM25 -> semantic -> LLM -> table-data -> OCR -> human-review cascade on one document
    # question is the real end user's question - no hardcoded search text anywhere in this cascade
    active_client = client or Groq()
    file_name = pdf_path.name
    review_reasons: list[str] = []
    resolutions: dict[str, FieldResolution] = {}

    # --- Phase 2 (unchanged) + "could not be read" ---
    try:
        extraction = extract_pdf_content(pdf_path)
    except Exception as exc:  # noqa: BLE001 - any parser failure means "unreadable"
        logger.error("Phase 2 extraction failed for %r: %s", file_name, exc)
        return _unreadable_result(file_name, reason=f"could not be read: {exc}")

    # `PDFExtraction` is a loosely-typed dict (str | list[str] | list[Table]),
    # so this narrows once, here, rather than at each of the several uses below.
    pages = cast(list[str], extraction["pages"])
    if not any(page.strip() for page in pages):
        # Tier 7, reached early rather than last: the document opened fine but
        # has no text layer (a scan). Nothing downstream has anything to work
        # with until OCR recovers some, so it runs here instead of after the
        # LLM tiers - "last resort" in priority, not in position.
        logger.info("%s: no extractable text - attempting OCR recovery", file_name)
        recovered_pages = ocr.recover_text(pdf_path)
        if recovered_pages is None:
            return _unreadable_result(file_name, reason="could not be read: no extractable text")
        logger.info("%s: OCR recovered text - continuing the cascade on it", file_name)
        pages = recovered_pages
        # Every value from this document now traces back to OCR output, which
        # is inherently noisier than a real text layer (a misread digit turns
        # 0.03% into 0.08% with no other symptom). The rest of the cascade
        # proceeds normally, but the run is flagged so a human knows the
        # numbers came from a scan - a confidence caveat, not a failure.
        review_reasons.append(
            "text was recovered by OCR (no text layer in the source PDF) - "
            "values are subject to OCR misreads and should be spot-checked"
        )

    full_text = "\n\n".join(pages)

    # --- Tier 1: regex pre-pass ---
    regex_hits = _regex_prepass(file_name, pages)

    # --- Tiers 2-3: narrow the LLM's context if a tier is confident ---
    narrowed_page_indices = bm25_search.rank_pages(pages, question)
    narrowing_tier = "bm25" if narrowed_page_indices is not None else None
    if narrowed_page_indices is None:
        narrowed_page_indices = semantic_search.rank_pages(pages, question)
        narrowing_tier = "semantic" if narrowed_page_indices is not None else None

    if narrowed_page_indices is not None:
        extraction_text = "\n\n".join(pages[i] for i in narrowed_page_indices)
        logger.info(
            "%s: narrowed to page(s) %s via %s",
            file_name,
            [i + 1 for i in narrowed_page_indices],
            narrowing_tier,
        )
    else:
        extraction_text = full_text

    # --- Tier 4: LLM extraction, bounded retry ---
    llm_fields = _extract_with_retry(extraction_text, client=active_client, model=model)
    if llm_fields is None:
        # Every LLM attempt on the narrowed/full text failed outright (not
        # "field not found" - a real API/parse failure every time). Retry
        # once more against the full document before giving up entirely, in
        # case the narrowing itself was the problem.
        if extraction_text != full_text:
            llm_fields = _extract_with_retry(full_text, client=active_client, model=model)
        if llm_fields is None:
            review_reasons.append("LLM extraction failed on every attempt (see logs)")
            llm_fields = RealExtractedFields(
                fund_name=None, is_esg=None, status=None, expense_ratio=None, aum=None, flags=()
            )

    # --- Tier 4b: narrowing-miss safety net - a field missing after narrowing might just be off the narrowed page(s), not off the document ---
    narrowing_miss_recovered: set[str] = set()
    if narrowed_page_indices is not None:
        missing_fields = [
            f for f in ("fund_name", "is_esg", "status", "expense_ratio", "aum") if getattr(llm_fields, f) is None
        ]
        if missing_fields:
            logger.info(
                "%s: %s missing after narrowing to page(s) %s - retrying against the full document",
                file_name,
                missing_fields,
                [i + 1 for i in narrowed_page_indices],
            )
            full_doc_fields = _extract_with_retry(full_text, client=active_client, model=model)
            if full_doc_fields is not None:
                merged = {f: getattr(llm_fields, f) for f in ("fund_name", "is_esg", "status", "expense_ratio", "aum")}
                for field_name in missing_fields:
                    recovered_value = getattr(full_doc_fields, field_name)
                    if recovered_value is not None:
                        merged[field_name] = recovered_value
                        narrowing_miss_recovered.add(field_name)
                # drop the first pass's "not found" flag for anything the retry actually recovered
                stale_flags = tuple(
                    f for f in llm_fields.flags if not any(f.startswith(f"{name}:") for name in narrowing_miss_recovered)
                )
                llm_fields = RealExtractedFields(
                    fund_name=merged["fund_name"],
                    is_esg=merged["is_esg"],
                    status=merged["status"],
                    expense_ratio=merged["expense_ratio"],
                    aum=merged["aum"],
                    flags=stale_flags + full_doc_fields.flags,
                )

    resolved = {
        "fund_name": llm_fields.fund_name,
        "is_esg": llm_fields.is_esg,
        "status": llm_fields.status,
        "expense_ratio": llm_fields.expense_ratio,
        "aum": llm_fields.aum,
    }
    for field_name in ("fund_name", "is_esg", "status"):
        # a field the safety net recovered is tagged distinctly from a normal narrowed/full-text hit
        llm_resolved_by = "llm_full_document_retry" if field_name in narrowing_miss_recovered else "llm"
        resolutions[field_name] = FieldResolution(
            value=resolved[field_name],
            resolved_by=llm_resolved_by if resolved[field_name] is not None else "unresolved",
            source=None,  # full/narrowed-doc LLM calls aren't asked to cite a page - see SourceLocation docstring
        )

    # --- Tier 5: cross-check regex hits against the LLM's numeric answers ---
    for field_name in ("expense_ratio", "aum"):
        llm_value = resolved[field_name]
        regex_hit = regex_hits.get(field_name)

        llm_resolved_by = "llm_full_document_retry" if field_name in narrowing_miss_recovered else "llm"

        if regex_hit is None:
            resolutions[field_name] = FieldResolution(
                value=llm_value, resolved_by=llm_resolved_by if llm_value is not None else "unresolved", source=None
            )
            continue

        regex_value, regex_source = regex_hit
        if llm_value is None:
            # The LLM saw this same text and did not confirm a regex hit
            # that heuristically looked plausible - treat as suspicious
            # rather than trusting the regex value unchecked (the same
            # risk a bad BM25/semantic match carries - see module docstring).
            review_reasons.append(
                f"{field_name}: regex found {regex_value!r} on page {regex_source.page}, "
                f"but the LLM did not confirm it - not trusted without review"
            )
            resolutions[field_name] = FieldResolution(value=None, resolved_by="unresolved", source=regex_source)
        elif abs(llm_value - regex_value) < 1e-6:
            resolutions[field_name] = FieldResolution(value=llm_value, resolved_by=llm_resolved_by, source=regex_source)
        else:
            review_reasons.append(
                f"{field_name}: regex found {regex_value!r} on page {regex_source.page}, "
                f"LLM found {llm_value!r} - disagreement, not resolved automatically"
            )
            resolutions[field_name] = FieldResolution(value=None, resolved_by="unresolved", source=regex_source)
            resolved[field_name] = None

    # --- Tier 6: table-data retry for anything still missing ---
    tables = extraction["tables"]
    assert isinstance(tables, list)
    still_missing = [f for f in ("expense_ratio", "aum") if resolved[f] is None and tables]
    if still_missing:
        table_text = _tables_to_text(tables)
        if table_text.strip():
            table_fields = _extract_with_retry(table_text, client=active_client, model=model)
            if table_fields is not None:
                for field_name in still_missing:
                    table_value = getattr(table_fields, field_name)
                    if table_value is not None:
                        resolved[field_name] = table_value
                        resolutions[field_name] = FieldResolution(
                            value=table_value,
                            resolved_by="llm_table_data",
                            source=SourceLocation(file=file_name, page=None, snippet=table_text[:200]),
                        )
                        # A field the table-data pass recovered is no longer
                        # unresolved - drop any disagreement/no-confirmation
                        # reason recorded for it above.
                        review_reasons[:] = [r for r in review_reasons if not r.startswith(f"{field_name}:")]

    final_fields = RealExtractedFields(
        fund_name=resolved["fund_name"],
        is_esg=resolved["is_esg"],
        status=resolved["status"],
        expense_ratio=resolved["expense_ratio"],
        aum=resolved["aum"],
        flags=llm_fields.flags,
    )

    for field_name, value in resolved.items():
        if value is None and not any(r.startswith(f"{field_name}:") for r in review_reasons):
            review_reasons.append(f"{field_name}: not found by any tier")

    return CascadeResult(
        fields=final_fields,
        resolutions=resolutions,
        needs_human_review=bool(review_reasons),
        review_reasons=tuple(review_reasons),
    )


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


def _run_cascade_demo() -> None:
    # runs the cascade against every PDF in data/user_uploads/, using a CLI-supplied question or a plain demo default
    import os
    import sys

    project_root = Path(__file__).resolve().parents[2]
    _load_dotenv(project_root / ".env")

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file), then re-run `python -m src.cascade.extraction_cascade`."
        )

    from src.extraction.real_data_loader import load_real_pdfs

    question = sys.argv[1] if len(sys.argv) > 1 else "What is the fund's ESG status, expense ratio, and net assets?"
    client = Groq()
    for pdf_path in load_real_pdfs(project_root / "data" / "user_uploads"):
        result = extract_with_cascade(pdf_path, question, client=client)
        print(f"\n=== {pdf_path.name} ===")
        print(f"  fields: {result.fields}")
        for name, resolution in result.resolutions.items():
            print(f"  {name}: {resolution.value!r} (via {resolution.resolved_by})")
        print(f"  needs_human_review: {result.needs_human_review}")
        for reason in result.review_reasons:
            print(f"    - {reason}")


def _unreadable_result(file_name: str, *, reason: str) -> CascadeResult:
    """Build the CascadeResult for a document with no extractable text at all."""
    empty_fields = RealExtractedFields(
        fund_name=None, is_esg=None, status=None, expense_ratio=None, aum=None, flags=()
    )
    return CascadeResult(
        fields=empty_fields,
        resolutions={
            name: FieldResolution(value=None, resolved_by="unresolved", source=SourceLocation(file=file_name, page=None, snippet=""))
            for name in ("fund_name", "is_esg", "status", "expense_ratio", "aum")
        },
        needs_human_review=True,
        review_reasons=(reason,),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _run_cascade_demo()
