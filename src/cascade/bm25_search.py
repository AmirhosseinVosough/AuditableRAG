"""Cascade tier 2: BM25 lexical page ranking.

Split out of `extraction_cascade.py` so each retrieval tier lives in its own
file - see that module's docstring for where this sits in the overall
regex -> BM25 -> semantic -> LLM -> table-data -> OCR -> human-review order.

This tier's whole job is *narrowing*: given a document's pages, decide which
one or two pages are worth sending to the LLM instead of paying to send the
whole document. It never extracts a field itself, and it never decides a
document has no answer - it either reports "these pages" or "not confident,
ask the next tier".

BM25 is the cheap tier of the two ranking tiers: pure term-overlap counting,
no model to load, no network call, milliseconds per document. It runs first
for exactly that reason, and `semantic_search.py` is only consulted when this
comes back unconfident.

No search text is hardcoded here. `rank_pages` takes `query` as a required
argument - in the real pipeline that's the actual end user's question,
passed through unmodified all the way from `pipeline.py`. A question that
doesn't mention every field this cascade extracts can leave this tier
narrowing away a page a different field needed - see
`extraction_cascade.py`'s narrowing-miss retry for how that gap gets closed
rather than silently reported as "not in the document".
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


# "Confident enough to narrow" test: the top-ranked page's score must be
# positive (some real term overlap happened at all) and must dominate the
# rest of the pages, not just edge them out - otherwise "confident" is doing
# the guessing this project exists to avoid. This is a heuristic, not a
# calibrated number - revisit once real long documents are available to tune
# against.
DOMINANCE_RATIO = 1.5
TOP_K_PAGES = 2


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization - good enough for BM25 term overlap."""
    return re.findall(r"[a-z0-9]+", text.lower())


def rank_pages(pages: list[str], query: str) -> list[int] | None:
    """Return the top-K page indices (0-indexed) most relevant to *query*, or None if not confident.

    *query* is caller-supplied - see the module docstring for why nothing is
    hardcoded here.

    "Confident" requires the top score to be positive *and* to dominate the
    rest of the pages by `DOMINANCE_RATIO` - a page that merely edges out the
    others isn't a strong enough signal to narrow the LLM's context to it.
    This is a heuristic threshold, not a calibrated one; see the module
    docstring.

    Returns None (never raises) when the document is too small to be worth
    narrowing or when nothing clears the confidence bar - both mean "let the
    next tier decide", not "this document has no answer".
    """
    if len(pages) <= TOP_K_PAGES:
        # Nothing to narrow - the whole document is already small.
        return None

    tokenized_pages = [_tokenize(page) for page in pages]
    bm25 = BM25Okapi(tokenized_pages)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_score = scores[ranked[0]]
    if top_score <= 0:
        return None

    rest = [scores[i] for i in ranked[1:]] or [0.0]
    rest_average = sum(rest) / len(rest)
    if top_score < DOMINANCE_RATIO * (rest_average + 1e-6):
        return None

    return ranked[:TOP_K_PAGES]
