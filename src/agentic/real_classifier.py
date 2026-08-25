"""Real-data-mode counterpart to `fund_filter.parse_fund_metadata`: a cheap, narrowed LLM call for is_esg/status only.

Why this exists: `retrieval.scope_documents` (Phase 11) narrows candidate
documents *before* paying for full field extraction - but it does that by
calling `fund_filter.parse_fund_metadata`, whose regexes are keyed to the
synthetic fixture's exact field-label text (e.g. a literal "ESG: Yes" line).
Real documents don't write is_esg/status as a fixed label - they describe it
in free text ("the Fund integrates ESG factors...", "this Fund does not
employ exclusionary screens..."), which regex cannot reliably read for
meaning. `extraction_cascade.py`'s own regex pre-pass already made this call
for the same two fields - see its `_regex_prepass` docstring - and
deliberately left them LLM-only.

This module is that LLM-only step, kept as cheap as it can be:

1. Narrow the document to the page(s) actually about ESG/status first, for
   free, via the same `bm25_search`/`semantic_search` tiers
   `extraction_cascade.py` uses (BM25 first, semantic only as a fallback).
   No LLM call has happened yet at this point.
2. Make ONE LLM call over just that narrowed text, asking for only
   `is_esg`/`status` - not the full 5-field schema
   `field_extraction.extract_real_fund_fields` uses. Smaller input, smaller
   output, and defaults to this project's smaller verified model (see
   `CLASSIFIER_MODEL`) rather than its strongest one.

`filter_real_funds` is the real-data counterpart to `fund_filter.filter_funds`
- it applies the identical is_esg/status comparison, but over already-
classified `RealFundMetadata` instead of parsing text via regex, because
`filter_funds` calls `parse_fund_metadata` internally and cannot accept a
pre-computed result. Neither `fund_filter.py` nor `extraction_cascade.py` is
modified by any of this - same "never extend Phase 3/9c itself" convention
`retrieval.py` and `extraction_cascade.py` already follow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable

from groq import Groq
from groq.types.chat import ChatCompletionToolChoiceOptionParam, ChatCompletionToolParam

from src.cascade import bm25_search, semantic_search
from src.fund_filter import FilterSpec
from src.shared.model_fallback import call_with_model_fallback, models_to_try


logger = logging.getLogger(__name__)

# The smaller of this project's two verified models (see model_fallback.py's
# FALLBACK_MODELS comment - both are verified against this project's actual
# forced tool-call schemas on real documents). A good default for this
# narrower, 2-field judgment; model_fallback's own chain still runs behind
# it (models_to_try(model) puts the 120b model right after it), so an
# outright failure here still falls back to the stronger model before
# giving up, same contract as extract_real_fund_fields.
CLASSIFIER_MODEL = "openai/gpt-oss-20b"

_TOOL_NAME = "record_esg_status"

_TOOL_SCHEMA: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": (
            "Record the ESG status and active/closed status of a fund, found "
            "in a real-world fund document's extracted text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "is_esg": {
                    "type": ["boolean", "null"],
                    "description": (
                        "True if the text describes the fund as ESG-focused, "
                        "sustainable, or screened on environmental/social/"
                        "governance criteria - judge this by what the text "
                        "says the fund actually does, not by whether the "
                        "fund's name or ticker contains the word 'ESG' (a "
                        "fund can be ESG-screened without 'ESG' in its name, "
                        "and vice versa). False if the text positively "
                        "indicates it is not ESG-focused. Omit this property "
                        "if the text gives no real basis to decide either way."
                    ),
                },
                "status": {
                    "type": ["string", "null"],
                    "enum": ["active", "closed", None],
                    "description": (
                        "'active' if the text indicates the fund is currently "
                        "operating; 'closed' if it indicates the fund has been "
                        "liquidated, closed, or terminated. Omit if the text "
                        "gives no basis to determine either."
                    ),
                },
            },
            # No "required" list, same reasoning as field_extraction.py's
            # _REAL_TOOL_SCHEMA: a real document may not state either field,
            # and the model must be free to omit rather than guess.
        },
    },
}

_SYSTEM_PROMPT = (
    "You determine only a fund's ESG status and active/closed status from "
    "real-world fund document text (fact sheets, prospectuses, shareholder "
    "reports) for a pipeline that must never guess. Use only the text "
    "provided - no outside knowledge of the fund. Call the provided tool "
    "exactly once. Include a property only when its value is explicitly "
    "stated in the text; omit it otherwise. Do not infer, estimate, or "
    "substitute a related-but-different figure. Only is_esg and status are "
    "wanted here - no other field."
)


@dataclass(frozen=True)
class RealFundMetadata:
    """Real-data counterpart to `fund_filter.FundMetadata` - is_esg/status only, both nullable.

    Nullable on purpose, like every real-data field in this project
    (`RealExtractedFields`): a real document not stating one of these is a
    legitimate, expected outcome, not a bug to work around. `flags` records
    why a field ended up None, one entry per field, same convention as
    `RealExtractedFields.flags`.
    """

    name: str
    is_esg: bool | None
    status: str | None  # "active" | "closed" | None
    flags: tuple[str, ...]


def classify_esg_status(
    name: str,
    pages: list[str],
    question: str,
    *,
    client: Groq | None = None,
    model: str = CLASSIFIER_MODEL,
) -> RealFundMetadata:
    """Cheaply determine one real document's is_esg/status, narrowing before the LLM call.

    Args:
        name: Document/fund identifier, carried straight into the returned
            `RealFundMetadata.name` - this function does no name parsing.
        pages: Per-page extracted text, e.g. `extract_pdf_content(path)["pages"]`.
            Must contain at least some non-blank text.
        question: The real end user's question, passed through unmodified to
            `bm25_search.rank_pages`/`semantic_search.rank_pages` - same
            narrowing contract `extraction_cascade.py` uses, not hardcoded
            here either.
        client: Groq client to use. Constructed fresh (`Groq()`) if omitted.
        model: Model to call first; see `CLASSIFIER_MODEL`.

    Returns:
        A `RealFundMetadata`. Either field the model did not report is None,
        with `flags` explaining why - "not found in document" if omitted,
        "unparseable value returned" if the model returned something
        off-schema (discarded rather than force-converted, same as
        `field_extraction._coerce_optional_field`).

    Raises:
        ValueError: If every page is blank, or if every model in the
            fallback chain failed outright - callers must catch this
            per-document so one bad document can't take down a batch, same
            requirement as `extract_real_fund_fields`.
    """
    full_text = "\n\n".join(pages)
    if not full_text.strip():
        raise ValueError(f"Cannot classify {name!r}: no extractable text")

    active_client = client or Groq()

    # Same two-tier narrowing extraction_cascade.py uses: BM25 first (free,
    # milliseconds), semantic only as a fallback when BM25 isn't confident.
    # Neither call reaches the LLM - both either report page indices or None.
    narrowed_indices = bm25_search.rank_pages(pages, question)
    narrowing_tier = "bm25" if narrowed_indices is not None else None
    if narrowed_indices is None:
        narrowed_indices = semantic_search.rank_pages(pages, question)
        narrowing_tier = "semantic" if narrowed_indices is not None else None

    if narrowed_indices is not None:
        classification_text = "\n\n".join(pages[i] for i in narrowed_indices)
        logger.debug(
            "%s: narrowed to page(s) %s via %s for is_esg/status classification",
            name,
            [i + 1 for i in narrowed_indices],
            narrowing_tier,
        )
    else:
        # Neither tier was confident - fall through to the full document,
        # same fallback extraction_cascade.py takes when narrowing misses.
        classification_text = full_text

    tool_choice: ChatCompletionToolChoiceOptionParam = {
        "type": "function",
        "function": {"name": _TOOL_NAME},
    }

    def _call(model_name: str) -> RealFundMetadata:
        response = active_client.chat.completions.create(
            model=model_name,
            # Narrowed input is a page or two, not a whole document, and the
            # output is only 2 fields - well under extract_real_fund_fields's
            # 1024 (5 fields, full document). Kept above the 150 that was
            # observed to truncate synthetic-mode tool calls, since these
            # models reason before answering regardless of schema size.
            max_tokens=768,
            seed=42,
            tools=[_TOOL_SCHEMA],
            tool_choice=tool_choice,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Document text:\n\n{classification_text}"},
            ],
        )

        tool_calls = response.choices[0].message.tool_calls or []
        for call in tool_calls:
            if call.function.name == _TOOL_NAME:
                try:
                    fields = json.loads(call.function.arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Model returned malformed tool-call arguments: {exc}") from exc
                return _build_real_fund_metadata(name, fields)

        raise ValueError("Model response did not include the expected tool call")

    return call_with_model_fallback(_call, models=models_to_try(model))


def _coerce_optional(
    fields: dict[str, object],
    key: str,
    expected_type: type | tuple[type, ...],
    flags: list[str],
) -> object | None:
    """Same contract as `field_extraction._coerce_optional_field` - duplicated, not imported.

    That function is field_extraction.py's private helper; reaching into
    another module's internals is exactly what this codebase's own
    convention avoids (see retrieval.py's module docstring). The ~15 lines
    are small enough that duplicating is cheaper than coupling the two
    modules together.
    """
    value = fields.get(key)
    if value is None:
        flags.append(f"{key}: not found in document")
        return None
    if not isinstance(expected_type, tuple) and isinstance(value, bool) and expected_type is not bool:
        # bool is a subclass of int - guards against a stray true/false
        # slipping past a check meant for a different type. Not load-bearing
        # for is_esg/status specifically (both expect bool/str), kept for
        # parity with the function this mirrors.
        flags.append(f"{key}: unparseable value returned ({value!r})")
        return None
    if not isinstance(value, expected_type):
        flags.append(f"{key}: unparseable value returned ({value!r})")
        return None
    return value


def _build_real_fund_metadata(name: str, fields: dict[str, object]) -> RealFundMetadata:
    """Turn the raw tool-call JSON into a RealFundMetadata, flagging anything off-schema."""
    flags: list[str] = []

    is_esg = _coerce_optional(fields, "is_esg", bool, flags)
    status = _coerce_optional(fields, "status", str, flags)
    if status is not None and status not in ("active", "closed"):
        flags.append(f"status: unparseable value returned ({status!r})")
        status = None

    return RealFundMetadata(
        name=name,
        is_esg=is_esg if isinstance(is_esg, bool) else None,
        status=status,
        flags=tuple(flags),
    )


def filter_real_funds(candidates: Iterable[RealFundMetadata], filter_spec: FilterSpec) -> list[str]:
    """Real-data counterpart to `fund_filter.filter_funds` - same comparison, pre-classified input.

    `filter_funds` cannot be reused directly: it calls `parse_fund_metadata`
    (regex) on raw text internally and has no way to accept an
    already-computed result. This applies the identical is_esg/status
    all-or-nothing comparison over `RealFundMetadata` instead.

    A fund whose is_esg/status couldn't be classified (None) is excluded
    whenever `filter_spec` constrains that dimension - never guessed into
    either bucket, matching this project's "never guess" rule everywhere
    else. Each candidate's `flags` explains why, for the caller's audit
    trail, before this function ever discards it.

    Returns:
        Qualifying fund names, in source order - same contract as `filter_funds`.
    """
    qualifying_funds: list[str] = []
    for fund in candidates:
        if filter_spec.is_esg is not None and fund.is_esg != filter_spec.is_esg:
            continue
        if filter_spec.status is not None and fund.status != filter_spec.status.lower():
            continue
        qualifying_funds.append(fund.name)
    return qualifying_funds
