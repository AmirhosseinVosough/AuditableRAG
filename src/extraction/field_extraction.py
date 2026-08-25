"""LLM-based numeric field extraction from a single fund's extracted text."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from groq import Groq
from groq.types.chat import ChatCompletionToolChoiceOptionParam, ChatCompletionToolParam

from src.shared.model_fallback import call_with_model_fallback, models_to_try

# llama-3.3-70b-versatile was retired from Groq. FALLBACK_MODELS[0] (see
# model_fallback.py) is openai/gpt-oss-120b - verified to force tool calls
# correctly. DEFAULT_MODEL stays a plain string (not the tuple) since every
# caller in this codebase passes a single `model: str` - the fallback chain
# behind it is applied automatically inside extract_fund_fields /
# extract_real_fund_fields, not something callers need to opt into.
DEFAULT_MODEL = "openai/gpt-oss-120b"

_TOOL_NAME = "record_fund_fields"

_TOOL_SCHEMA: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": (
            "Record the fund name, expense ratio, and assets under management "
            "found in a single fund fact sheet's extracted text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fund_name": {
                    "type": "string",
                    "description": "The full name of the fund exactly as written in the text.",
                },
                "expense_ratio": {
                    "type": "number",
                    "description": (
                        "The fund's expense ratio as a percentage number, e.g. an "
                        "expense ratio of 0.45% is recorded as 0.45."
                    ),
                },
                "aum": {
                    "type": "number",
                    "description": (
                        "Assets under management (fund size / net assets) in "
                        "millions of USD, e.g. $120M or $120 million is recorded as 120."
                    ),
                },
            },
            "required": ["fund_name", "expense_ratio", "aum"],
        },
    },
}

_SYSTEM_PROMPT = (
    "You extract structured numeric fields from a single fund fact sheet. "
    "Use only the text provided, do not guess or use outside knowledge, and "
    "call the provided tool exactly once with the extracted values."
)


@dataclass(frozen=True)
class ExtractedFields:
    fund_name: str
    expense_ratio: float
    aum: float


def extract_fund_fields(
    text: str,
    *,
    client: Groq | None = None,
    model: str = DEFAULT_MODEL,
) -> ExtractedFields:
    """Use an LLM tool call to pull {fund_name, expense_ratio, aum} out of *text*.

    Structured output is enforced by forcing the model to call the
    ``record_fund_fields`` tool rather than parsing free-form JSON out of a
    text response.

    If the call to *model* fails outright (rate limit, network error, a
    malformed/missing tool call - not a legitimate result), this
    automatically falls back to the next model in
    `model_fallback.FALLBACK_MODELS`, logging a warning first - see
    `model_fallback.call_with_model_fallback`. Only raises once every model
    in the chain has failed.
    """
    if not text.strip():
        raise ValueError("Cannot extract fields from empty fund text")

    active_client = client or Groq()

    tool_choice: ChatCompletionToolChoiceOptionParam = {
        "type": "function",
        "function": {"name": _TOOL_NAME}, #"auto" (Default): The model decides dynamically whether to chat normally or call one of your provided tools.
        #"none": The model is forbidden from calling any tools. It will only reply with standard text.
        # "required": The model must choose and call at least one tool from your list, but it gets to decide which specific tool to use.
    }

    def _call(model_name: str) -> ExtractedFields:
        response = active_client.chat.completions.create(
            model=model_name,
            max_tokens=150, #prevents generational loop
            seed=42, #helps with reproducibility of results, but it is not guaranteed to produce the same output every time.
            tools=[_TOOL_SCHEMA],
            tool_choice=tool_choice,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Fund fact sheet text:\n\n{text}"},
            ],
        )

        tool_calls = response.choices[0].message.tool_calls or []
        for call in tool_calls:
            if call.function.name == _TOOL_NAME:
                fields = json.loads(call.function.arguments)
                return ExtractedFields(
                    fund_name=str(fields.get("fund_name", "unknown fund")),
                    expense_ratio=float(fields.get("expense_ratio", 0.0)),
                    aum=float(fields.get("aum", 0.0)),
                )

        raise ValueError("Model response did not include the expected tool call")

    return call_with_model_fallback(_call, models=models_to_try(model))


# --- Phase 9c: real-data mode extraction -------------------------------------
#
# Everything below this line is new for Phase 9 and is used only by
# pipeline.py's source="real" path. `extract_fund_fields` above (Phase 4,
# synthetic mode) is untouched - real-data mode does not call it, and
# nothing here changes its behavior.
#
# The key difference: `extract_fund_fields` forces fund_name/expense_ratio/aum
# to always be present (`required` in the schema, `.get(key, default)` on the
# way out) because synthetic fixtures always contain all three. Real
# documents don't come with that guarantee - a real fact sheet may state an
# expense ratio but not AUM, or vice versa, or use a figure that looks like
# but is not the field we want (see `aum`'s description below). Forcing a
# value in that situation means guessing, which Phase 9c explicitly forbids.
# So every field here is optional, and a missing/malformed field is recorded
# in `flags` rather than defaulted or force-converted.

_REAL_TOOL_NAME = "record_real_fund_fields"

_REAL_TOOL_SCHEMA: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": _REAL_TOOL_NAME,
        "description": (
            "Record whatever fund fields can be confidently determined from a "
            "real-world fund document's extracted text. This text was not "
            "written for automated parsing - it may be a fact sheet, "
            "prospectus, or shareholder report, in any layout. Omit a "
            "property entirely (preferred), or set it to null, if its value "
            "is not explicitly stated in the text - both are accepted. "
            "Never guess, estimate, or infer a value from a "
            "related-but-different figure - a null/omitted field is far "
            "more useful downstream than a wrong one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fund_name": {
                    "type": "string",
                    "description": "The fund's full name exactly as written, if stated.",
                },
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
                "expense_ratio": {
                    "type": ["number", "null"],
                    "description": (
                        "The fund's expense ratio as a percentage number, e.g. "
                        "an expense ratio of 0.45% is recorded as 0.45. Omit "
                        "if no expense ratio figure is stated, or if you "
                        "cannot confidently convert the stated figure to a "
                        "plain percentage."
                    ),
                },
                "aum": {
                    "type": ["number", "null"],
                    "description": (
                        "Assets under management - the fund's OWN total net "
                        "assets - in millions of USD, e.g. $120M or $120 "
                        "million is recorded as 120. This must be the fund's "
                        "own net assets, not the average market cap of its "
                        "underlying holdings or any other similar-sounding "
                        "figure. Omit if the fund's own AUM/net-assets figure "
                        "is not stated."
                    ),
                },
            },
            # Deliberately no "required" list here (unlike _TOOL_SCHEMA above):
            # every field may legitimately be absent from a real document, and
            # the model must be free to omit rather than guess.
        },
    },
}

_REAL_SYSTEM_PROMPT = (
    "You extract structured fields from real-world fund documents (fact "
    "sheets, prospectuses, shareholder reports) for a pipeline that must "
    "never guess. Use only the text provided - no outside knowledge of the "
    "fund. Call the provided tool exactly once. Include a property only "
    "when its value is explicitly stated in the text; omit it otherwise. "
    "Do not infer, estimate, or substitute a related-but-different figure."
)


@dataclass(frozen=True)
class RealExtractedFields:
    """Best-effort structured fields pulled from one real-world fund document.

    Every field is nullable on purpose - real documents routinely omit or
    obscure at least one of these, and the whole point of real-data mode is
    reporting that honestly instead of guessing a plausible-looking number.
    `flags` records *why* each field ended up null (not found, or found but
    not usable), one entry per field, so the pipeline's audit trail can show
    a human exactly what went wrong without them re-reading the source PDF.
    """

    fund_name: str | None
    is_esg: bool | None
    status: str | None  # "active" | "closed" | None
    expense_ratio: float | None
    aum: float | None
    flags: tuple[str, ...]


def extract_real_fund_fields(
    text: str,
    *,
    client: Groq | None = None,
    model: str = DEFAULT_MODEL,
) -> RealExtractedFields:
    """Best-effort structured extraction for real-data mode (Phase 9c).

    Unlike `extract_fund_fields` (Phase 4, synthetic mode), every field here
    is allowed to come back null - a real document not stating its AUM is a
    legitimate, expected outcome, not a bug to work around. This function
    never raises on a missing or malformed *field* value; it only raises if
    there is no text at all to read, or the API call/response itself fails
    outright - both of which the caller (the real-data pipeline path) must
    still catch per-document, so one bad document can't take down the batch.

    Args:
        text: Extracted PDF text. Must be non-empty - a document with no
            extractable text (e.g. a scanned image page) is a document-level
            failure ("could not be read"), not a field-level one, and must
            be caught by the caller *before* this function is ever called.
        client: Groq client to use. Constructed fresh (`Groq()`) if omitted.
        model: Model to call.

    Returns:
        A `RealExtractedFields` record. Any field the model did not report
        is None, and `flags` explains why for every field - "not found in
        document" if the model omitted it, or "unparseable value returned"
        if the model returned something that doesn't match the expected
        type (discarded rather than force-converted).

    Raises:
        ValueError: If `text` is empty/whitespace-only, or if every model in
            the fallback chain (see below) failed - the last one's error is
            what's raised.

    Automatic model fallback: if the call to *model* fails outright (rate
    limit, network error, malformed tool-call JSON, missing tool call at
    all), this falls back to the next model in
    `model_fallback.FALLBACK_MODELS`, logging a warning first - see
    `model_fallback.call_with_model_fallback`. A field that legitimately
    comes back null is never a fallback trigger - that's a correct result,
    not a failure.
    """
    if not text.strip():
        raise ValueError("Cannot extract fields from empty document text")

    active_client = client or Groq()

    tool_choice: ChatCompletionToolChoiceOptionParam = {
        "type": "function",
        "function": {"name": _REAL_TOOL_NAME},
    }

    def _call(model_name: str) -> RealExtractedFields:
        response = active_client.chat.completions.create(
            model=model_name,
            # Real documents run 10-20K+ characters (vs. the synthetic fixture's
            # ~200) and this model reasons before answering - at the synthetic
            # call's max_tokens=150 the response was observed to get cut off
            # mid-tool-call (either an empty generation or truncated JSON
            # arguments) before it ever finished. 1024 leaves real headroom for
            # both the reasoning and the five-field tool call.
            max_tokens=1024,
            seed=42,
            tools=[_REAL_TOOL_SCHEMA],
            tool_choice=tool_choice,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _REAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Document text:\n\n{text}"},
            ],
        )

        tool_calls = response.choices[0].message.tool_calls or []
        for call in tool_calls:
            if call.function.name == _REAL_TOOL_NAME:
                try:
                    fields = json.loads(call.function.arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Model returned malformed tool-call arguments: {exc}") from exc
                return _build_real_extracted_fields(fields)

        raise ValueError("Model response did not include the expected tool call")

    return call_with_model_fallback(_call, models=models_to_try(model))


def _coerce_optional_field(
    fields: dict[str, object],
    key: str,
    expected_type: type | tuple[type, ...],
    flags: list[str],
) -> object | None:
    """Return fields[key] if present and correctly typed; else None, with a flag explaining why.

    Never raises and never force-converts - a wrongly-typed value is treated
    exactly like a missing one (None + a flag), not coerced with e.g.
    float(value), which could itself raise on a value like "N/A" or silently
    produce a wrong number for something like "45 bps".
    """
    value = fields.get(key)
    if value is None:
        flags.append(f"{key}: not found in document")
        return None
    if not isinstance(expected_type, tuple) and isinstance(value, bool) and expected_type is not bool:
        # bool is a subclass of int in Python - without this guard, a stray
        # `true`/`false` from the model would silently pass an (int, float)
        # check meant for expense_ratio/aum.
        flags.append(f"{key}: unparseable value returned ({value!r})")
        return None
    if not isinstance(value, expected_type):
        flags.append(f"{key}: unparseable value returned ({value!r})")
        return None
    return value


def _build_real_extracted_fields(fields: dict[str, object]) -> RealExtractedFields:
    """Turn the raw tool-call JSON into a RealExtractedFields, flagging anything off-schema.

    Deliberately does not use dict.get(key, default) the way
    `extract_fund_fields` does above - a missing key here means "not found,"
    and defaulting to 0.0/"" would silently turn that into a guessed value,
    which is exactly what Phase 9c says never to do.
    """
    flags: list[str] = []

    fund_name = _coerce_optional_field(fields, "fund_name", str, flags)
    is_esg = _coerce_optional_field(fields, "is_esg", bool, flags)
    status = _coerce_optional_field(fields, "status", str, flags)
    if status is not None and status not in ("active", "closed"):
        flags.append(f"status: unparseable value returned ({status!r})")
        status = None
    expense_ratio = _coerce_optional_field(fields, "expense_ratio", (int, float), flags)
    aum = _coerce_optional_field(fields, "aum", (int, float), flags)

    return RealExtractedFields(
        fund_name=fund_name if isinstance(fund_name, str) else None,
        is_esg=is_esg if isinstance(is_esg, bool) else None,
        status=status,
        expense_ratio=float(expense_ratio) if isinstance(expense_ratio, (int, float)) else None,
        aum=float(aum) if isinstance(aum, (int, float)) else None,
        flags=tuple(flags),
    )


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, if present.

    Avoids adding a python-dotenv dependency for this one use. Existing
    environment variables are never overwritten.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _run_phase_3_and_4() -> None:
    """Filter to qualifying funds (Phase 3), then extract fields for each (Phase 4)."""
    from src.fund_filter import FilterSpec, filter_funds, parse_fund_metadata
    from src.generate_synthetic_data import FUNDS
    from src.extraction.pdf_extraction import extract_pdf_content

    project_root = Path(__file__).resolve().parents[2]
    _load_dotenv(project_root / ".env")

    pdf_paths = sorted((project_root / "data" / "raw_pdfs").glob("*.pdf"))
    extracted_funds = [extract_pdf_content(pdf_path) for pdf_path in pdf_paths]

    qualifying_names = filter_funds(
        extracted_funds, FilterSpec(is_esg=True, status="active")
    )


    """Sending straight to LLM: Easy to code, but expensive,
    slow, and prone to silent data tracking errors.Using Phase 3 First: 
    Pre-clears the data for free, reduces your API bill by filtering out
    non-qualifying funds, and creates a reliable baseline for Phase 5
        to check against."""

    """Cleans Text Lines: It splits raw, messy text from your PDFs 
    into individual rows, strips away loose spaces, and throws out empty lines.
    Filters via Local Regex: It uses fast regular expressions to check if a fund
      is ESG and Active. If a fund fails this test, it is dropped immediately 
      via a continue command.Builds the Map and Expected Targets: It maps 
      the clean fund name to its raw text string inside the name_to_text look-up
        dictionary. At the same time,
          it builds a true_values dictionary from your local mock data to act as your verification checklist."""
    name_to_text: dict[str, str] = {}
    for fund in extracted_funds:
        text = fund["text"]
        if not isinstance(text, str):
            raise ValueError("Each extracted fund must contain a string 'text' field")
        name_to_text[parse_fund_metadata(text).name] = text
    true_values = {
        
        fund.fund_name: (fund.expense_ratio_percent, fund.aum_millions_usd)
        for fund in FUNDS
    }

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file), then re-run `python -m src.extraction.field_extraction`."
        )

    client = Groq()

    print(f"{len(qualifying_names)} qualifying funds from Phase 3 filter:")
    for name in qualifying_names:
        print(f"  - {name}")
    print()

    results: list[tuple[str, ExtractedFields]] = []
    for name in qualifying_names:
        text = name_to_text[name]
        extracted = extract_fund_fields(text, client=client)
        results.append((name, extracted))
        print(extracted)

    print()
    header = f"{'Fund':<34} {'Expense Ratio (extracted/true)':<32} {'AUM $M (extracted/true)':<26} Match"
    print(header)
    print("-" * len(header))
    for name, extracted in results:
        true_expense, true_aum = true_values[name]
        expense_match = abs(extracted.expense_ratio - true_expense) < 1e-6
        aum_match = abs(extracted.aum - true_aum) < 1e-6
        match = "OK" if expense_match and aum_match else "MISMATCH"
        print(
            f"{name:<34} "
            f"{extracted.expense_ratio:.2f} / {true_expense:.2f}{'':<20} "
            f"{extracted.aum:.0f} / {true_aum:.0f}{'':<15} "
            f"{match}"
        )


if __name__ == "__main__":
    _run_phase_3_and_4()
