"""LLM-based numeric field extraction from a single fund's extracted text."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from groq import Groq
from groq.types.chat import ChatCompletionToolChoiceOptionParam, ChatCompletionToolParam


DEFAULT_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was retired from Groq; verified this one still forces tool calls correctly.

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
    response = active_client.chat.completions.create(
        model=model,
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
    from src.pdf_extraction import extract_pdf_content

    project_root = Path(__file__).resolve().parents[1]
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
            "project-root .env file), then re-run `python -m src.field_extraction`."
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
