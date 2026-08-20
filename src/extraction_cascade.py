"""Fallback extraction cascade for real-data mode: regex -> BM25 -> semantic -> LLM -> human review.

This sits *inside* Phase 9c's real-data extraction step, as a more
cost/latency-aware way to resolve a document's fields than always sending
the whole thing to the LLM - useful once real documents get long (full
prospectuses, N-CSRs) rather than the short fact sheets Phase 9's own
fixture set happened to use. It does not replace `field_extraction.py`'s
`extract_real_fund_fields` - every tier that reaches the LLM still calls it
unchanged; this module only decides *what text* to give it, cross-checks
its numeric answers against a cheap regex pass, and adds a table-data retry
and a bounded reprompt retry before giving up and flagging for review.

Tier order (see the design conversation this came out of for the reasoning
behind each choice, not repeated here):

    1. Regex pre-pass (expense_ratio/aum only - is_esg/status structurally
       need the LLM regardless, per Phase 9's own design decision; regex
       cannot reliably classify substance). Not a "success -> done" gate
       the way it can be for a single fixed document format - it's a cheap,
       best-effort cross-check input for step 5, since the LLM call still
       has to run either way to get is_esg/status.
    2. BM25 lexical ranking of pages, to narrow which page(s) get sent to
       the LLM instead of the whole document.
    3. Semantic/embedding ranking - STUB. Groq does not host an embedding
       model, so this needs either a new external embedding API or a heavy
       local model (sentence-transformers), both real dependency/cost
       decisions this module does not make unilaterally. Always reports "no
       confident match", so the cascade falls through to step 4 exactly as
       it does today (Phase 9c's existing behavior) - this stub makes the
       cascade neither better nor worse than today for this tier, only
       honest about not being implemented yet.
    4. LLM extraction on the narrowed (or, if nothing was confident, full)
       document text - `extract_real_fund_fields`, unchanged, wrapped in a
       bounded reprompt retry for transient failures only (not for "field
       genuinely not found", which no retry will fix).
    5. Cross-check regex hints (step 1) against the LLM's answer: agreement
       is a confidence signal, disagreement is flagged for human review
       rather than silently trusting either source.
    6. Table-data retry: if expense_ratio/aum is still missing and
       pdfplumber detected tables (`PDFExtraction.tables` - captured since
       Phase 2 but never read by anything until now), one more bounded LLM
       call scoped to just the rendered table text.
    7. OCR - STUB, for the same reason semantic search is: a real
       library/approach choice (pytesseract + system Tesseract vs. a
       vision-capable LLM call) that isn't this module's to make alone.
       Always reports "not recovered", so an unreadable document still ends
       up flagged "could not be read" exactly as it does today.
    8. Human review: anything still unresolved, or any cross-check
       disagreement, is reported in `CascadeResult.review_reasons` rather
       than guessed past.

This module is intentionally standalone - it is not yet wired into
`pipeline.py`'s `_run_real_pipeline`, which still calls
`extract_real_fund_fields` directly. Swapping that in is a separate,
separately-testable change.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from groq import Groq
from rank_bm25 import BM25Okapi

from src.field_extraction import DEFAULT_MODEL, RealExtractedFields, extract_real_fund_fields
from src.pdf_extraction import Table, extract_pdf_content
from src.source_location import SourceLocation


logger = logging.getLogger(__name__)

# Bounded, hard-coded - never left to the model's judgment about when to
# stop, matching this project's retry-cap convention everywhere else.
_MAX_LLM_RETRY_ATTEMPTS = 2

# BM25 "confident enough to narrow" test: the top-ranked page's score must
# be positive (some real term overlap happened at all) and must dominate
# the rest of the pages, not just edge them out - otherwise "confident" is
# doing the guessing this project exists to avoid. This is a heuristic, not
# a calibrated number - revisit once real long documents are available to
# tune against.
_BM25_QUERY = (
    "ESG environmental social governance expense ratio net assets total "
    "assets under management active closed fund status"
)
_BM25_DOMINANCE_RATIO = 1.5
_BM25_TOP_K_PAGES = 2


@dataclass(frozen=True)
class FieldResolution:
    """One field's resolved value, which tier resolved it, and where it came from.

    `resolved_by` is one of: "regex", "llm", "llm_table_data", or
    "unresolved". Kept as a plain string rather than a Literal/enum so a
    later tier (semantic, OCR) can be added without touching this type.
    """

    value: object
    resolved_by: str
    source: SourceLocation | None


@dataclass(frozen=True)
class CascadeResult:
    """The outcome of running the full extraction cascade on one document."""

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


# --- Tier 2: BM25 lexical page ranking ----------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization - good enough for BM25 term overlap."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_rank_pages(pages: list[str]) -> list[int] | None:
    """Return the top-K page indices (0-indexed) most relevant to `_BM25_QUERY`, or None if not confident.

    "Confident" requires the top score to be positive *and* to dominate the
    rest of the pages by `_BM25_DOMINANCE_RATIO` - a page that merely edges
    out the others isn't a strong enough signal to narrow the LLM's context
    to it. This is a heuristic threshold, not a calibrated one; see the
    module docstring.
    """
    if len(pages) <= _BM25_TOP_K_PAGES:
        # Nothing to narrow - the whole document is already small.
        return None

    tokenized_pages = [_tokenize(page) for page in pages]
    bm25 = BM25Okapi(tokenized_pages)
    scores = bm25.get_scores(_tokenize(_BM25_QUERY))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_score = scores[ranked[0]]
    if top_score <= 0:
        return None

    rest = [scores[i] for i in ranked[1:]] or [0.0]
    rest_average = sum(rest) / len(rest)
    if top_score < _BM25_DOMINANCE_RATIO * (rest_average + 1e-6):
        return None

    return ranked[:_BM25_TOP_K_PAGES]


# --- Tier 3: semantic/embedding ranking (STUB - see module docstring) --------

def _semantic_rank_pages(pages: list[str]) -> list[int] | None:  # noqa: ARG001 - signature kept for parity with _bm25_rank_pages
    """Not implemented. Always reports no confident match - see module docstring.

    Deliberately not raising NotImplementedError: this tier's whole job is
    to be a *fallback candidate*, so "not available" must degrade to "try
    the next tier", not crash the cascade. When this is implemented for
    real, it keeps this exact contract (page indices or None).
    """
    return None


# --- Tier 7: OCR (STUB - see module docstring) --------------------------------

def _ocr_recover_text(pdf_path: Path) -> list[str] | None:  # noqa: ARG001
    """Not implemented. Always reports no recovered text - see module docstring.

    Same "return None, don't raise" contract as `_semantic_rank_pages` -
    an unreadable document must still be reported as "could not be read",
    not crash the batch, until this is actually built.
    """
    return None


# --- Bounded retry wrapper for LLM calls --------------------------------------

def _extract_with_retry(text: str, *, client: Groq, model: str) -> RealExtractedFields | None:
    """Call extract_real_fund_fields, retrying up to `_MAX_LLM_RETRY_ATTEMPTS` on transient failures.

    Only retries exceptions that plausibly mean "the model glitched this
    time" (malformed JSON, missing tool call, a network/API error) - a
    field genuinely not being in the text is not a transient failure and
    retrying cannot fix it; that comes back as a normal (non-exceptional)
    RealExtractedFields with the field null, and is not retried here.

    Returns None (rather than raising) if every attempt fails, so the
    caller can route to the next cascade tier instead of crashing the
    whole document's processing - the caller is responsible for surfacing
    this as a review reason if no later tier recovers it.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_LLM_RETRY_ATTEMPTS + 1):
        try:
            return extract_real_fund_fields(text, client=client, model=model)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure here is a retry candidate, logged either way
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
    *,
    client: Groq | None = None,
    model: str = DEFAULT_MODEL,
) -> CascadeResult:
    """Run the full regex -> BM25 -> semantic -> LLM -> table-data -> OCR -> human-review cascade on one document.

    Args:
        pdf_path: The PDF to process.
        client: Groq client to reuse. Constructed fresh (`Groq()`) if omitted.
        model: Model to call for every LLM tier.

    Returns:
        A CascadeResult. `fields` is always populated (with nulls where
        nothing resolved); check `needs_human_review` and `review_reasons`
        rather than assuming a non-null field is fully trustworthy - a
        cross-check disagreement can leave `needs_human_review=True` even
        when both `fields` values are non-null.
    """
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

    pages = extraction["pages"]
    assert isinstance(pages, list)
    if not any(page.strip() for page in pages):
        recovered_pages = _ocr_recover_text(pdf_path)
        if recovered_pages is None:
            return _unreadable_result(file_name, reason="could not be read: no extractable text")
        pages = recovered_pages  # pragma: no cover - unreachable until OCR is implemented

    full_text = "\n\n".join(pages)

    # --- Tier 1: regex pre-pass ---
    regex_hits = _regex_prepass(file_name, pages)

    # --- Tiers 2-3: narrow the LLM's context if a tier is confident ---
    narrowed_page_indices = _bm25_rank_pages(pages)
    narrowing_tier = "bm25" if narrowed_page_indices is not None else None
    if narrowed_page_indices is None:
        narrowed_page_indices = _semantic_rank_pages(pages)
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

    resolved = {
        "fund_name": llm_fields.fund_name,
        "is_esg": llm_fields.is_esg,
        "status": llm_fields.status,
        "expense_ratio": llm_fields.expense_ratio,
        "aum": llm_fields.aum,
    }
    for field_name in ("fund_name", "is_esg", "status"):
        resolutions[field_name] = FieldResolution(
            value=resolved[field_name],
            resolved_by="llm" if resolved[field_name] is not None else "unresolved",
            source=None,  # full/narrowed-doc LLM calls aren't asked to cite a page - see SourceLocation docstring
        )

    # --- Tier 5: cross-check regex hits against the LLM's numeric answers ---
    for field_name in ("expense_ratio", "aum"):
        llm_value = resolved[field_name]
        regex_hit = regex_hits.get(field_name)

        if regex_hit is None:
            resolutions[field_name] = FieldResolution(
                value=llm_value, resolved_by="llm" if llm_value is not None else "unresolved", source=None
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
            resolutions[field_name] = FieldResolution(value=llm_value, resolved_by="llm", source=regex_source)
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
    """Run the cascade against every PDF in data/user_uploads/ and print each result."""
    import os

    project_root = Path(__file__).resolve().parents[1]
    _load_dotenv(project_root / ".env")

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file), then re-run `python -m src.extraction_cascade`."
        )

    from src.real_data_loader import load_real_pdfs

    client = Groq()
    for pdf_path in load_real_pdfs(project_root / "data" / "user_uploads"):
        result = extract_with_cascade(pdf_path, client=client)
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
