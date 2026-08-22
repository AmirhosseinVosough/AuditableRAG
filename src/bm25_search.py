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
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi


# The keyword bag this tier scores pages against - deliberately a bag of
# terms rather than a sentence, since BM25 matches on term overlap and has no
# notion of meaning (that's `semantic_search.py`'s job, phrased as a real
# question there for the same reason).
QUERY = (
    "ESG environmental social governance expense ratio net assets total "
    "assets under management active closed fund status"
)

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


def rank_pages(pages: list[str]) -> list[int] | None:
    """Return the top-K page indices (0-indexed) most relevant to `QUERY`, or None if not confident.

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
    scores = bm25.get_scores(_tokenize(QUERY))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_score = scores[ranked[0]]
    if top_score <= 0:
        return None

    rest = [scores[i] for i in ranked[1:]] or [0.0]
    rest_average = sum(rest) / len(rest)
    if top_score < DOMINANCE_RATIO * (rest_average + 1e-6):
        return None

    return ranked[:TOP_K_PAGES]
