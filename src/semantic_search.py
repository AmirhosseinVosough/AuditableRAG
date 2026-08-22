"""Cascade tier 3: semantic/embedding page ranking, via a local sentence-transformers model.

Split out of `extraction_cascade.py` so each retrieval tier lives in its own
file - see that module's docstring for where this sits in the overall
regex -> BM25 -> semantic -> LLM -> table-data -> OCR -> human-review order.

Same contract as `bm25_search.rank_pages`, deliberately: take a document's
pages, return the indices worth sending to the LLM, or None for "not
confident, ask the next tier". The two are interchangeable from the
orchestrator's point of view - which is why the orchestrator can simply try
one and fall through to the other.

Why this tier exists at all, given BM25 already ranks pages: BM25 matches
term overlap, so it only finds a page that literally uses the words in the
query. A page that says "the fund applies sustainability screens to its
holdings" and never writes "ESG" scores near zero on the BM25 query, but is
exactly the page we want. Embeddings match meaning rather than spelling, so
they catch that case - at the cost of being much slower (a model to load,
then a forward pass per page, vs. BM25's plain counting). That cost is why
this runs *second*, only when BM25 wasn't confident, rather than replacing
it.

Why a local model rather than an embedding API: Groq (this project's only
LLM provider) does not host an embeddings endpoint, so the alternative was
adding a second paid provider and API key purely for this tier.
all-MiniLM-L6-v2 is ~80MB, runs fine on CPU, costs nothing per call, and
sends no document text off the machine - which also sidesteps the data
privacy concern the README's risk table raises about third-party APIs. It is
downloaded once from Hugging Face on first use and cached under
~/.cache/huggingface thereafter.

Failure policy, matching every other tier here: this module never raises. A
missing package, a failed first-run download (no internet), or an embedding
error all degrade to "no confident match", so the cascade falls through to
the LLM tier exactly as it did before this tier existed. A broken model
makes the cascade cheaper-but-dumber, never broken.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

# Small (~80MB), CPU-friendly, no API key/cost - see the module docstring for
# why this was chosen over a paid embedding API.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Natural-language rephrasing of `bm25_search.QUERY`'s keyword bag - embedding
# similarity works on meaning, not term overlap, so this is phrased as a real
# question rather than a keyword list.
QUERY = (
    "Does this page state the fund's ESG or sustainability screening, its "
    "expense ratio, its net assets or assets under management, or whether "
    "the fund is currently open or closed?"
)

# Pages are scored by their best-matching *chunk*, not as a whole - see
# `_chunk`. ~350 characters is roughly a paragraph: long enough to carry
# meaning on its own, short enough that one line stating an expense ratio
# still dominates the chunk it sits in. The overlap keeps a fact that
# straddles a boundary from being split across two chunks and diluted in both.
CHUNK_SIZE = 350
CHUNK_OVERLAP = 80

# Confidence gate. Deliberately NOT the ratio test `bm25_search` uses: BM25
# scores start at 0 and spread widely, so a ratio is meaningful there. Cosine
# similarities between two real English texts sit in a compressed band (~0.4
# to ~0.65 across this project's documents - unrelated prose still scores ~0.4,
# it never approaches 0), so a 1.5x ratio would require an unreachable ~0.77
# and the tier would abstain on literally everything. What actually carries
# signal is the *margin* over the document's own baseline: how far the top page
# beats the average of the rest.
#
# Both numbers below were calibrated against the five real fact sheets in
# data/user_uploads/, scoring each page as relevant/not by whether it actually
# states the expense ratio, net assets, or ESG policy. On that set the
# top-ranked page's margin cleanly separates the two outcomes: the picks that
# were correct led by +0.124 and +0.064, while every incorrect pick led by only
# +0.031 to +0.036. MIN_MARGIN sits between those clusters, so the tier commits
# on the two it gets right and abstains on the three it would have got wrong.
#
# Five documents is a small calibration set and these thresholds should be
# revisited against longer, more varied documents (prospectuses, N-CSRs) - the
# case this tier actually exists for. They are deliberately tuned to abstain
# when unsure: a wrong page narrows the LLM's context to text that does not
# contain the answer, which is strictly worse than not narrowing at all.
MIN_SCORE = 0.45
MIN_MARGIN = 0.05

# Independently defined rather than imported from `bm25_search` - the two
# tiers happening to agree on 2 is a coincidence of tuning, not a constraint
# that should couple the modules together.
TOP_K_PAGES = 2

# Lazily-loaded singleton, reused across documents/calls within a process
# rather than reloaded per document - loading takes seconds, encoding takes
# milliseconds. `_model_unavailable` remembers a load failure so a
# missing/broken install doesn't retry (and re-log) the same slow failure on
# every document in a batch; it only resets on process restart.
_model: "SentenceTransformer | None" = None
_model_unavailable = False


def _get_model() -> "SentenceTransformer | None":
    """Return the shared sentence-transformers model, loading it on first call; None if unavailable.

    Never raises - a missing package or a failed download (e.g. no internet on
    first run) is logged once and reported as "unavailable" so callers degrade
    to the next cascade tier instead of crashing the batch, the same contract
    every other tier follows.
    """
    global _model, _model_unavailable

    if _model is not None:
        return _model
    if _model_unavailable:
        return None

    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    except Exception as exc:  # noqa: BLE001 - any load failure means "unavailable", not "crash the cascade"
        logger.warning(
            "Semantic ranking model (%s) failed to load: %s - the semantic tier will report no "
            "confident match for the rest of this run",
            MODEL_NAME,
            exc,
        )
        _model_unavailable = True
        return None

    return _model


def _chunk(text: str) -> list[str]:
    """Split one page into overlapping ~`CHUNK_SIZE`-character chunks; empty list for a blank page.

    Whitespace is collapsed first so that PDF-extracted text (full of ragged
    newlines and column padding) chunks by actual content length rather than
    by layout artifacts.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: list[str] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(normalized), step):
        chunks.append(normalized[start : start + CHUNK_SIZE])
    return chunks


def rank_pages(pages: list[str]) -> list[int] | None:
    """Return the top-K page indices (0-indexed) most semantically relevant to `QUERY`, or None if not confident.

    Only called when `bm25_search.rank_pages` was not confident (see
    `extraction_cascade.extract_with_cascade`) - this tier is deliberately not
    run unconditionally, since it's far slower than BM25 and BM25 already
    resolves the common case.

    Each page is scored by its single best-matching chunk rather than by
    embedding the page as a whole. Whole-page embeddings were measurably worse
    on this project's own documents: a 4-8K-character fact sheet page averages
    out to "this is a fund document", so every page scores nearly the same and
    the one line stating an expense ratio is washed out. Scoring by best chunk
    keeps that line's signal intact.

    Confidence requires the top page to clear both `MIN_SCORE` (an absolute
    floor) and `MIN_MARGIN` (a real lead over the other pages) - see those
    constants for how they were calibrated and why this abstains rather than
    guessing. Returns None (never raises) if the model is unavailable, if
    embedding fails, or if nothing clears the bar - every one of those means
    "let the next tier decide", not "this document has no answer".
    """
    if len(pages) <= TOP_K_PAGES:
        # Nothing to narrow - the whole document is already small.
        return None

    model = _get_model()
    if model is None:
        return None

    # Encode every page's chunks in one batched call rather than one call per
    # page - the per-call overhead dominates at this model size.
    page_chunks = [_chunk(page) for page in pages]
    flat_chunks = [chunk for chunks in page_chunks for chunk in chunks]
    if not flat_chunks:
        return None

    try:
        chunk_embeddings = model.encode(flat_chunks, normalize_embeddings=True, show_progress_bar=False)
        query_embedding = model.encode(QUERY, normalize_embeddings=True, show_progress_bar=False)
    except Exception as exc:  # noqa: BLE001 - an embedding failure degrades to the next tier, like a load failure does
        logger.warning("Semantic embedding failed: %s - falling through to the next tier", exc)
        return None

    # Both sides are L2-normalized, so the dot product is exactly cosine
    # similarity - no separate normalization step needed here.
    chunk_scores = np.asarray(chunk_embeddings) @ np.asarray(query_embedding)

    # Fold the flat chunk scores back into one score per page: its best chunk.
    # A page with no text at all scores -1.0 (below any real cosine similarity)
    # so it can never win by default.
    scores: list[float] = []
    cursor = 0
    for chunks in page_chunks:
        if not chunks:
            scores.append(-1.0)
            continue
        scores.append(float(np.max(chunk_scores[cursor : cursor + len(chunks)])))
        cursor += len(chunks)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_score = scores[ranked[0]]
    if top_score < MIN_SCORE:
        logger.debug("Semantic: top score %.3f below MIN_SCORE %.2f - abstaining", top_score, MIN_SCORE)
        return None

    rest = [scores[i] for i in ranked[1:]] or [0.0]
    margin = top_score - (sum(rest) / len(rest))
    if margin < MIN_MARGIN:
        logger.debug(
            "Semantic: top page leads by only %.3f (below MIN_MARGIN %.2f) - abstaining rather than "
            "narrowing to a page it isn't confident about",
            margin,
            MIN_MARGIN,
        )
        return None

    logger.debug("Semantic: page %d wins with score %.3f (margin %.3f)", ranked[0] + 1, top_score, margin)
    return ranked[:TOP_K_PAGES]
