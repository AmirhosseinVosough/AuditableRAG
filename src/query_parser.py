"""Phase 10: translate an arbitrary natural-language question into a structured QuerySpec.

This is the first of three places the agentic layer (Phase 10-14) is allowed
to use an LLM at all (per the core architectural mandate in
AGENTIC_LAYER_BUILD_PROMPT.md): translating a question into a QuerySpec via a
forced tool-call, never free-form reasoning about what the user meant. The
model never decides how to filter or calculate anything itself - it only
extracts the structured request; Phase 11/12 apply it using the existing
deterministic filter/extract/verify/calculate logic, unchanged.

Two tools, one required call (`tool_choice="required"`), model picks which:

    record_query_spec        - the question maps cleanly onto QuerySpec's fields
    reject_unsupported_query - it doesn't, and the model must say why

This is a genuinely different tool_choice shape than Phase 4/9c's forced
*named* tool call. Phase 4 has exactly one thing to do every time (extract
these fields), so forcing a single named tool is correct there. Phase 10 has
a real choice to make - can this question be answered at all - so
`tool_choice="required"` with two tools, letting the model pick, is the
right fit: never free text, but able to say "not supported" instead of
guessing a QuerySpec that doesn't actually represent the question.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from groq import Groq
from groq.types.chat import ChatCompletionToolChoiceOptionParam, ChatCompletionToolParam

from src.field_extraction import DEFAULT_MODEL
from src.model_fallback import call_with_model_fallback, models_to_try


@dataclass(frozen=True)
class QuerySpec:
    """A structured, supported request this pipeline can actually answer.

    Mirrors `fund_filter.FilterSpec`'s is_esg/status convention (None means
    "no constraint on this field") plus two fields FilterSpec doesn't have:

    closed_quarter_exclusions: specific quarters the question names as a
        reason to exclude a fund, e.g. "excluding funds closed in Q3"
        records ("Q3",). Kept separate from `status` because a question
        could in principle name a quarter without implying "active only"
        (though the canonical example question does - see the module
        docstring's discussion of that phrasing in the system prompt below).
        Empty tuple, not None, when no quarter is named - there's nothing
        ambiguous about "no exclusions", so there's no need for a third
        state.

    requested_metric: today, always "weighted_average_expense_ratio" - the
        only metric `calculator.py` computes. Explicit as its own field
        (rather than assumed) so a question asking for a different metric
        (total AUM, fund count, ...) has something concrete to fail to
        match, and `parse_query` can reject it instead of silently
        answering the one question it knows how to answer.
    """

    is_esg: bool | None
    status: Literal["active", "closed"] | None
    closed_quarter_exclusions: tuple[str, ...]
    requested_metric: Literal["weighted_average_expense_ratio"]


class UnsupportedQueryError(Exception):
    """Raised when a question cannot be mapped onto QuerySpec's supported fields.

    Carries the original question and the model's stated reason, so a human
    reading a log (or a caller deciding how to respond to the user) can see
    exactly what was asked and why it doesn't fit - not just that parsing
    "failed" in some generic sense.
    """

    def __init__(self, *, question: str, reason: str) -> None:
        self.question = question
        self.reason = reason
        super().__init__(
            f"Question could not be mapped to a supported QuerySpec: {reason!r} "
            f"(question: {question!r})"
        )


_QUERY_SPEC_TOOL_NAME = "record_query_spec"
_REJECT_TOOL_NAME = "reject_unsupported_query"

_QUERY_SPEC_TOOL_SCHEMA: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": _QUERY_SPEC_TOOL_NAME,
        "description": (
            "Record a structured QuerySpec for a fund-aggregation question this "
            "pipeline can actually answer: a weighted-average-expense-ratio "
            "calculation over funds filtered by ESG status, active/closed status, "
            "and/or specific closed-quarter exclusions. Only call this if the "
            "question maps cleanly onto exactly these fields - if it asks for a "
            "different metric, a filter this schema has no field for, or is not "
            "really a fund-aggregation question at all, call "
            "reject_unsupported_query instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "is_esg": {
                    "type": "boolean",
                    "description": (
                        "True if the question asks about ESG/sustainable funds, "
                        "false if it asks about non-ESG/conventional funds. Omit "
                        "if the question does not filter on ESG status at all."
                    ),
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "closed"],
                    "description": (
                        "'active' if the question implies only currently-operating "
                        "funds should count. This includes phrasing like 'excluding "
                        "closed funds' or 'excluding funds closed in Q3' - both signal "
                        "that closed funds in general should not count, not just the "
                        "one quarter named, so both should set status='active' (in "
                        "addition to recording the named quarter in "
                        "closed_quarter_exclusions below). 'closed' if the question "
                        "explicitly asks about closed funds. Omit if the question "
                        "does not filter on active/closed status at all."
                    ),
                },
                "closed_quarter_exclusions": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["Q1", "Q2", "Q3", "Q4"]},
                    "description": (
                        "Specific quarters the question names as an exclusion reason, "
                        "e.g. 'excluding funds closed in Q3' records [\"Q3\"]. Empty "
                        "list if the question does not name any specific quarter, "
                        "even if it otherwise filters on status."
                    ),
                },
                "requested_metric": {
                    "type": "string",
                    "enum": ["weighted_average_expense_ratio"],
                    "description": (
                        "The metric being requested. Only "
                        "'weighted_average_expense_ratio' is supported by this "
                        "pipeline today - if the question asks for anything else "
                        "(total AUM, fund count, a ranking, ...), do not set this "
                        "field to this value; call reject_unsupported_query instead."
                    ),
                },
            },
            "required": ["requested_metric"],
        },
    },
}

_REJECT_TOOL_SCHEMA: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": _REJECT_TOOL_NAME,
        "description": (
            "Call this instead of record_query_spec when the question cannot be "
            "mapped onto the supported QuerySpec fields - e.g. it asks for a metric "
            "other than weighted average expense ratio, a filter criterion this "
            "schema has no field for (geography, fund size, manager name, ...), or "
            "is not a fund-aggregation question at all."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "A short, specific explanation of why the question is out of scope.",
                },
            },
            "required": ["reason"],
        },
    },
}

_SYSTEM_PROMPT = (
    "You translate a natural-language fund-aggregation question into a structured "
    "query for a deterministic pipeline. Call record_query_spec if the question "
    "maps cleanly onto its fields, or reject_unsupported_query if it does not - "
    "never answer in free text, and never guess a QuerySpec for a question the "
    "schema cannot actually represent. Different wordings of the same underlying "
    "request must produce the same QuerySpec - focus on what is actually being "
    "asked, not the specific phrasing used to ask it."
)


def parse_query(
    question: str,
    *,
    client: Groq | None = None,
    model: str = DEFAULT_MODEL,
) -> QuerySpec:
    """Translate *question* into a QuerySpec via a forced tool-call, or reject it.

    Args:
        question: The natural-language question to parse.
        client: Groq client to use. Constructed fresh (`Groq()`) if omitted.
        model: Model to call.

    Returns:
        The parsed QuerySpec.

    Raises:
        ValueError: If `question` is empty/whitespace-only, if the model's
            tool-call arguments aren't valid JSON, or if the model's
            response doesn't include either expected tool call at all.
        UnsupportedQueryError: If the model determined the question cannot
            be mapped onto QuerySpec's supported fields.

    Automatic model fallback: if the call to *model* fails outright (rate
    limit, network error, malformed tool-call JSON, neither expected tool
    called), this falls back to the next model in
    `model_fallback.FALLBACK_MODELS`, logging a warning first. A question
    the model correctly rejects as out of scope is NOT a failure - it is a
    legitimate answer - so it is never retried against a different model;
    see the `_call` closure below for how that distinction is kept out of
    `call_with_model_fallback`'s generic exception-means-retry logic.
    """
    if not question.strip():
        raise ValueError("Cannot parse an empty question")

    active_client = client or Groq()

    # "required" (not a specific named tool): the model must call one of the
    # two tools below, but which one is a real decision it has to make -
    # see the module docstring for why this differs from Phase 4/9c's forced
    # single-named-tool pattern.
    tool_choice: ChatCompletionToolChoiceOptionParam = "required"

    def _call(model_name: str) -> QuerySpec | UnsupportedQueryError:
        """Returns a QuerySpec on success, or an *unraised* UnsupportedQueryError instance
        when the model correctly rejects the question. Returning it (rather than raising)
        is deliberate: call_with_model_fallback treats any exception as "this model failed,
        try the next one," and a correct rejection must never trigger that - it's the right
        answer, not a failure. parse_query raises it itself, once, after the fallback
        machinery has already decided this was the model's final word.
        """
        response = active_client.chat.completions.create(
            model=model_name,
            max_tokens=400,
            seed=42,
            tools=[_QUERY_SPEC_TOOL_SCHEMA, _REJECT_TOOL_SCHEMA],
            tool_choice=tool_choice,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )

        tool_calls = response.choices[0].message.tool_calls or []
        for call in tool_calls:
            if call.function.name == _REJECT_TOOL_NAME:
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Model returned malformed tool-call arguments: {exc}") from exc
                return UnsupportedQueryError(
                    question=question, reason=str(args.get("reason", "not specified"))
                )

            if call.function.name == _QUERY_SPEC_TOOL_NAME:
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Model returned malformed tool-call arguments: {exc}") from exc
                return QuerySpec(
                    is_esg=args.get("is_esg"),
                    status=args.get("status"),
                    closed_quarter_exclusions=tuple(args.get("closed_quarter_exclusions", ())),
                    requested_metric=args["requested_metric"],
                )

        raise ValueError(
            "Model response did not include either expected tool call "
            f"({_QUERY_SPEC_TOOL_NAME!r} or {_REJECT_TOOL_NAME!r})"
        )

    result = call_with_model_fallback(_call, models=models_to_try(model))
    if isinstance(result, UnsupportedQueryError):
        raise result
    return result


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a simple KEY=VALUE .env file, if present.

    Duplicated from the other phase modules rather than imported - see
    field_extraction.py's copy for the full reasoning. Existing environment
    variables are never overwritten.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# --- Phase 10 stopping-condition demo -----------------------------------------

# Five distinct wordings of the original example question
# ("weighted average expense ratio for ESG funds, excluding funds closed in
# Q3"), none identical to that original phrasing - the point is proving
# paraphrase-robustness, not that the model can repeat the exact sentence
# it was originally given.
_PARAPHRASED_QUESTIONS = (
    "What is the weighted average expense ratio for ESG funds, excluding any that closed in Q3?",
    "Across ESG-focused funds that are still active - leaving out the ones that shut down in "
    "the third quarter - what's the AUM-weighted average expense ratio?",
    "Calculate the weighted-average expense ratio of sustainable funds, not counting Q3 closures.",
    "Looking only at ESG funds that remain active today, and setting aside any that closed in "
    "Q3, what is their weighted average expense ratio?",
    "I want the weighted average expense ratio across ESG funds. Exclude anything that closed in Q3.",
)

# Deliberately out of scope: a different metric (total AUM, not weighted
# average expense ratio) that QuerySpec has no field to represent.
_OUT_OF_SCOPE_QUESTION = "What is the total AUM of all ESG funds?"


def _run_phase_10_demo() -> None:
    """Run the five paraphrases plus the out-of-scope question and print the results."""
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment (or in a "
            "project-root .env file), then re-run `python -m src.query_parser`."
        )

    client = Groq()

    print("=== Five paraphrases of the original example question ===\n")
    specs: list[QuerySpec] = []
    for question in _PARAPHRASED_QUESTIONS:
        spec = parse_query(question, client=client)
        specs.append(spec)
        print(f"Q: {question}")
        print(f"   -> {spec}\n")

    all_identical = all(spec == specs[0] for spec in specs)
    print(f"All 5 QuerySpecs identical: {all_identical}\n")

    print("=== Deliberately out-of-scope question ===\n")
    print(f"Q: {_OUT_OF_SCOPE_QUESTION}")
    try:
        spec = parse_query(_OUT_OF_SCOPE_QUESTION, client=client)
        print(f"   -> UNEXPECTEDLY ACCEPTED as {spec} (this should have been rejected)")
    except UnsupportedQueryError as exc:
        print(f"   -> Rejected, as expected: {exc.reason}")


if __name__ == "__main__":
    _run_phase_10_demo()
