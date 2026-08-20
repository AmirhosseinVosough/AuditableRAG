"""A shared provenance dataclass: where in a source document a piece of information came from.

Deliberately its own tiny module rather than living inside any one
extraction tier - `extraction_cascade.py`'s regex pass, BM25 ranking, LLM
extraction, and (eventually) human-review escalation all need to say "this
value/candidate/flag came from this file, this page, this text" in the
*same* shape, so a result that started as a BM25 hit and a result that
started as a human-review escalation land in the audit log identically
rather than needing per-tier translation. A module each of those can import
without any of them depending on each other keeps that possible without a
circular-import problem.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLocation:
    """Where a value, candidate match, or flagged issue came from in a source document.

    Every field here is meant to be genuinely known, not padded out for
    uniformity - see `page`'s note below.

    Attributes:
        file: The source PDF's filename (not a full path - the rest of the
            pipeline already treats the filename/stem as a document's
            identity; a full path would tie the audit log to one machine's
            directory layout for no benefit).
        page: 1-indexed page number the value/snippet came from, or None
            when the originating tier genuinely has no page-level
            granularity to report - e.g. a regex or BM25 hit knows exactly
            which page matched, but an LLM call given the whole,
            page-boundary-flattened document was never asked to cite one.
            None is an honest "not known", not a placeholder - never guess
            a page number to fill this in.
        snippet: The actual text that justifies the value - what a human
            reviewing this later would want to read without having to
            re-open the source PDF themselves.
    """

    file: str
    page: int | None
    snippet: str
