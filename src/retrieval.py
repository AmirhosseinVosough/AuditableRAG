"""Phase 11: decide which documents are candidates for a QuerySpec, before extraction runs on any of them.

Goal, verbatim from the build prompt: given a QuerySpec, decide which
documents are even candidates *before* running extraction on all of them -
structured metadata filtering, not vector/semantic search. On the 12-document
synthetic set this looks nearly identical to Phase 3's own filter, on
purpose - `scope_documents` does not reimplement filtering, it *is* Phase 3's
filter, reached through a QuerySpec-shaped door instead of a
FilterSpec-shaped one. That's deliberate: this is the first piece of the
agentic layer, and the architectural mandate is that the orchestrator (Phase
12) never contains filtering logic of its own - it only calls tools. This
module is what makes `fund_filter.filter_funds` callable by something that
only knows about QuerySpec, without the orchestrator having to construct a
FilterSpec itself or reach into fund_filter.py's internals.

`closed_quarter_exclusions` is deliberately not applied as an independent
filter dimension - see `_query_spec_to_filter_spec`'s docstring for why.

Real-data scale: see the module-level note near the bottom of this file, and
the demo output - the honest answer is that this function's *current*
implementation does not carry over to real-data mode unchanged, for the same
reason Phase 9c never calls `parse_fund_metadata` on real documents.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from src.fund_filter import FilterSpec, filter_funds
from src.query_parser import QuerySpec


def _query_spec_to_filter_spec(query_spec: QuerySpec) -> FilterSpec:
    """Translate the agentic layer's QuerySpec into Phase 3's FilterSpec.

    Only is_esg/status carry over - those are the two dimensions
    `fund_filter.FilterSpec` supports, and this function does not extend
    Phase 3 to add a third. `closed_quarter_exclusions` is intentionally not
    applied as its own filter here, for two reasons:

    1. `FundMetadata` (what `parse_fund_metadata` produces) only has a binary
       active/closed `status`, plus an informational `closed_quarter` on
       closed funds - there is no per-quarter *inclusion* semantics to filter
       on without changing Phase 3 itself, which this module (like Phase 9c)
       deliberately does not do.
    2. It would be redundant for this project's data anyway: status="active"
       already excludes every closed fund regardless of which quarter it
       closed in (confirmed by data/ground_truth.json - F009, closed in Q4,
       is excluded from the ESG-active answer exactly like the Q3 closures
       are, even though the original example question only names Q3). See
       Phase 10's own module docstring for the same observation from the
       query-parsing side.

    `requested_metric` is not used here either - which metric is wanted is
    Phase 12/6's concern (which calculator function to call), not a
    document-scoping concern.
    """
    return FilterSpec(is_esg=query_spec.is_esg, status=query_spec.status)


def scope_documents(
    query_spec: QuerySpec,
    extracted_funds: Iterable[Mapping[str, object]],
) -> list[str]:
    """Return the names of documents that are candidates for *query_spec*.

    This is Phase 3's `filter_funds`, reached through a QuerySpec instead of
    a hand-built FilterSpec - see the module docstring for why that
    indirection exists rather than folding scoping directly into the
    orchestrator.

    Args:
        query_spec: The parsed query (from `query_parser.parse_query`)
            describing which documents are wanted.
        extracted_funds: Phase 2 output - each a mapping with a string
            "text" field. Same shape `fund_filter.filter_funds` takes.

    Returns:
        Qualifying fund names, in source order - identical to what
        `filter_funds` would return for the equivalent `FilterSpec`.
    """
    filter_spec = _query_spec_to_filter_spec(query_spec)
    return filter_funds(extracted_funds, filter_spec)


# --- Phase 11 stopping-condition demo -----------------------------------------


def _run_phase_11_demo() -> None:
    """Parse the original example question, then show scope_documents narrowing the synthetic set."""
    import os
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    _load_dotenv(project_root / ".env")

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file), then re-run `python -m src.retrieval`."
        )

    from src.pdf_extraction import extract_pdf_content
    from src.query_parser import parse_query

    question = "weighted average expense ratio for ESG funds, excluding funds closed in Q3"
    query_spec = parse_query(question)
    print(f"Question: {question}")
    print(f"Parsed QuerySpec: {query_spec}\n")

    raw_pdf_dir = project_root / "data" / "raw_pdfs"
    pdf_paths = sorted(raw_pdf_dir.glob("*.pdf"))
    extracted_funds = [extract_pdf_content(pdf_path) for pdf_path in pdf_paths]

    scoped_names = scope_documents(query_spec, extracted_funds)
    print(f"Candidates out of {len(extracted_funds)} documents ({len(scoped_names)} scoped in):")
    for name in scoped_names:
        print(f"  - {name}")

    ground_truth_path = project_root / "data" / "ground_truth.json"
    if ground_truth_path.is_file():
        import json

        ground_truth = json.loads(ground_truth_path.read_text())
        match = scoped_names == ground_truth["qualifying_funds"]
        print(f"\nMatches data/ground_truth.json's qualifying_funds exactly: {match}")

    print(
        "\nAt real-data scale: this function's current body does not carry over "
        "unchanged, for the same reason Phase 9c's real-data extraction never "
        "calls parse_fund_metadata - its regexes are keyed to the synthetic "
        "fixture's exact field-label text and essentially never match an "
        "arbitrary real document. What DOES carry over is the shape of the "
        "idea: resolve is_esg/status cheaply, before paying for full field "
        "extraction, so a folder of hundreds of real PDFs doesn't mean "
        "hundreds of expensive LLM calls just to find out which ones even "
        "qualify. In real-data mode that cheap pre-pass would have to be "
        "something else - e.g. the regex pre-pass extraction_cascade.py "
        "already runs (fast, free, no LLM call, works when it works), or a "
        "lightweight classification pass distinct from the full 5-field "
        "extraction call - falling through to the full extraction (which is "
        "what determines is_esg/status today, per Phase 9's design decision) "
        "only for documents the cheap pass can't confidently classify. That "
        "narrowing step does not exist yet for real-data mode; this module "
        "is the synthetic-mode implementation of the general idea, not a "
        "real-data-ready one."
    )


def _load_dotenv(path) -> None:  # type: ignore[no-untyped-def]
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


if __name__ == "__main__":
    _run_phase_11_demo()
