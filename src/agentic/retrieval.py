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

Real-data scale: `scope_documents` below still does not carry over to
real-data mode unchanged, for the same reason Phase 9c never calls
`parse_fund_metadata` on real documents - see the demo output at the bottom
of this file for the full explanation. `scope_real_documents` is the
real-data counterpart: same QuerySpec-in, names-out shape, but backed by
`real_classifier.classify_esg_status` (a cheap, narrowed LLM call) instead
of regex - see `real_classifier.py`'s module docstring for why regex
couldn't do this job on real documents.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from groq import Groq

from src.fund_filter import FilterSpec, filter_funds
from src.agentic.query_parser import QuerySpec
from src.agentic.real_classifier import CLASSIFIER_MODEL, classify_esg_status, filter_real_funds


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


def scope_real_documents(
    query_spec: QuerySpec,
    candidates: Sequence[tuple[str, list[str]]],
    question: str,
    *,
    client: Groq | None = None,
    model: str = CLASSIFIER_MODEL,
) -> list[str]:
    """Real-data counterpart to `scope_documents` - same QuerySpec-in, names-out contract.

    Where `scope_documents` reaches `fund_filter.filter_funds` (regex, via
    `parse_fund_metadata`), this reaches `real_classifier.filter_real_funds`
    (a cheap, narrowed LLM call per candidate, via `classify_esg_status`) -
    see `real_classifier.py`'s module docstring for why regex can't do this
    job on real documents.

    Args:
        query_spec: The parsed query (from `query_parser.parse_query`)
            describing which documents are wanted.
        candidates: `(name, pages)` per real document - `pages` is
            `extract_pdf_content(path)["pages"]`, i.e. already PDF-extracted,
            not re-read from disk here.
        question: The real end user's question text, passed through to each
            `classify_esg_status` call for BM25/semantic narrowing - kept
            separate from `query_spec` because narrowing works over raw
            question text, not the structured spec parsed from it.
        client: Groq client to use for every classification call.
            Constructed fresh per call (`Groq()`) if omitted.
        model: Model to call first for each candidate; see
            `real_classifier.CLASSIFIER_MODEL`.

    Returns:
        Qualifying document names, in candidate order - same contract as
        `scope_documents`, so callers (the orchestrator) can use either
        interchangeably depending on synthetic vs. real-data mode.

    Raises:
        ValueError: Propagated from `classify_esg_status` if any candidate
            has no extractable text, or if classification fails outright for
            every model in the fallback chain - callers batching many
        n    real-data call site in this project.
    """
    filter_spec = _query_spec_to_filter_spec(query_spec)
    metadatas = [
        classify_esg_status(name, pages, question, client=client, model=model)
        for name, pages in candidates
    ]
    return filter_real_funds(metadatas, filter_spec)


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
            "project-root .env file), then re-run `python -m src.agentic.retrieval`."
        )

    from src.extraction.pdf_extraction import extract_pdf_content
    from src.agentic.query_parser import parse_query

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
        "\nAt real-data scale: this function's own body (scope_documents) still "
        "does not carry over unchanged, for the same reason Phase 9c's "
        "real-data extraction never calls parse_fund_metadata - its regexes "
        "are keyed to the synthetic fixture's exact field-label text and "
        "essentially never match an arbitrary real document. What DOES carry "
        "over is the shape of the idea: resolve is_esg/status cheaply, before "
        "paying for full field extraction, so a folder of hundreds of real "
        "PDFs doesn't mean hundreds of expensive LLM calls just to find out "
        "which ones even qualify. See scope_real_documents (this module) and "
        "real_classifier.py for the real-data-mode implementation of that "
        "idea: BM25/semantic narrowing (free) followed by one cheap, "
        "narrow-scope LLM call per candidate (is_esg/status only, not the "
        "full 5-field extraction) - falling through to the full extraction "
        "only for documents that survive scoping."
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
