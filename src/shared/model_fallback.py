"""Shared Groq model-fallback helper: try a primary model, log and fall back on outright failure.

Used by every LLM call site in this project (field_extraction.py's
extract_fund_fields / extract_real_fund_fields, query_parser.py's
parse_query) so a rate-limit or other API failure on one model doesn't have
to stop the whole pipeline. Groq's rate limits are per-model, not
per-account - falling back to a different model gets a genuinely fresh
quota pool, not just a retry against the same exhausted one.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Verified by hand, live, against this project's actual forced tool-call
# schemas on real (not just synthetic-fixture) documents - both call the
# right tool with the right fields, not just "some model that responds".
# Order matters: the first is the strongest available; each later one is a
# smaller sibling on Groq with its own separate daily/per-minute quota.
FALLBACK_MODELS: tuple[str, ...] = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")


def models_to_try(primary: str, fallback_models: Sequence[str] = FALLBACK_MODELS) -> tuple[str, ...]:
    """Return (primary, *every other model in fallback_models not already primary).

    Used so a caller who explicitly asks for a specific model still gets
    that model tried first, with the standard fallback chain behind it -
    rather than either ignoring their choice or trying it twice.
    """
    rest = tuple(model for model in fallback_models if model != primary)
    return (primary, *rest)


def call_with_model_fallback(call_fn: Callable[[str], T], *, models: Sequence[str]) -> T:
    """Call call_fn(model) for each model in *models*, in order, until one succeeds.

    Falls back to the next model - logging a warning first - only on an
    outright failure of the call itself (rate limit, network error,
    malformed response, etc.). This never falls back because a field
    legitimately came back null - that is a correct answer from a model
    that worked fine, not a failure to retry past. `call_fn` is responsible
    for that distinction: it must return normally (with nulls where
    appropriate) for "field not found", and only raise for a genuine call
    failure.

    Args:
        call_fn: Called once per model with that model's name. Must raise
            on failure, not return a sentinel - this function can only
            distinguish "worked" from "didn't" via exceptions.
        models: Models to try, in order. Must be non-empty.

    Returns:
        Whatever the first successful `call_fn(model)` returns.

    Raises:
        ValueError: If `models` is empty.
        Exception: The last model's exception, if every model in `models` failed.
    """
    if not models:
        raise ValueError("models must be non-empty")

    last_exc: Exception | None = None
    for index, model in enumerate(models):
        try:
            return call_fn(model)
        except Exception as exc:  # noqa: BLE001 - any failure here is a fallback candidate, logged either way
            last_exc = exc
            if index + 1 < len(models):
                logger.warning(
                    "Model %r failed (%s) - falling back to %r", model, exc, models[index + 1]
                )
            else:
                logger.error("Model %r failed (%s) - no more fallback models to try", model, exc)

    assert last_exc is not None  # unreachable: models is non-empty, so the loop ran at least once
    raise last_exc
