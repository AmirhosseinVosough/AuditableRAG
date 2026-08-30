"""Phase 10: translate an arbitrary natural-language question into a structured QuerySpec.

This is the first of three places the agentic layer (Phase 10-14) is allowed
to use an LLM at all (per the core architectural mandate in
AGENTIC_LAYER_BUILD_PROMPT.md): translating a question into a QuerySpec via a
forced tool-call, never free-form reasoning about what the user meant. The
model never decides how to filter or calculate anything itself - it only
extracts the structured request; Phase 11/12 apply it using the existing
deterministic filter/extract/verify/calculate logic, unchanged.

Three tools, one required call (`tool_choice="required"`), model picks which:

    record_query_spec        - the question maps cleanly onto QuerySpec's fields
    reject_unsupported_query - it doesn't, and the model must say why
    ask_clarifying_question  - Phase 17: every field the question needs DOES
                                exist in QuerySpec, but the wording doesn't
                                clearly resolve to one value for at least one
                                of them (e.g. "expense ratio for ESG funds"
                                never says whether closed funds count) - the
                                honest response is a specific question back,
                                not a guessed QuerySpec or an outright reject

This is a genuinely different tool_choice shape than Phase 4/9c's forced
*named* tool call. Phase 4 has exactly one thing to do every time (extract
these fields), so forcing a single named tool is correct there. Phase 10 has
a real choice to make - can this question be answered at all, and if so, is
it actually clear - so `tool_choice="required"` with multiple tools, letting
the model pick, is the right fit: never free text, but able to say "not
supported" or "ambiguous, here's what I need to know" instead of guessing a
QuerySpec that doesn't actually represent the question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from groq import Groq
from groq.types.chat import ChatCompletionToolChoiceOptionParam, ChatCompletionToolParam

from src.extraction.field_extraction import DEFAULT_MODEL
from src.shared.env import require_groq_api_key
from src.shared.model_fallback import call_with_model_fallback, models_to_try


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


@dataclass(frozen=True)
class ClarificationNeeded:
    """Phase 17: the question is answerable in principle, but its wording leaves one field unresolved.

    Distinct from `UnsupportedQueryError`: unsupported means the schema has
    no field that could ever represent what's being asked (a different
    metric, a dimension this system doesn't track at all). This means every
    field the question needs *does* exist in `QuerySpec`, but this specific
    question's wording doesn't clearly resolve to one value for at least one
    of them - "expense ratio for ESG funds" never says whether closed funds
    should count, so `status` is genuinely ambiguous, not just "no
    constraint" (which would itself be a valid, resolved value).

    Returned, not raised, exactly like a `QuerySpec` - this is a valid,
    expected outcome of parsing, not a failure. Callers are expected to
    surface `question` to the end user and call `parse_query` again with a
    refined question; `parse_query` itself does not loop or retry on this
    outcome - see the module docstring.
    """

    question: str
    ambiguous_field: Literal["is_esg", "status", "closed_quarter_exclusions", "requested_metric"]


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

_CLARIFY_TOOL_NAME = "ask_clarifying_question"

_CLARIFY_TOOL_SCHEMA: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": _CLARIFY_TOOL_NAME,
        "description": (
            "Call this when the question is answerable by this pipeline in principle "
            "- every field it needs exists in QuerySpec - but the wording does not "
            "clearly resolve to one value for at least one of those fields. This is "
            "different from reject_unsupported_query: unsupported means no field "
            "could ever represent what is being asked (e.g. a different metric like "
            "total AUM). Ambiguous means the field exists, but this specific "
            "question's wording does not pin it down - e.g. 'expense ratio for ESG "
            "funds' never says whether closed funds should count, so status is "
            "genuinely ambiguous, not simply 'no constraint'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "A short, specific question to ask the user back, naming "
                        "exactly what is unclear - not a generic 'can you clarify?'."
                    ),
                },
                "ambiguous_field": {
                    "type": "string",
                    "enum": ["is_esg", "status", "closed_quarter_exclusions", "requested_metric"],
                    "description": "Which QuerySpec field the question's wording leaves unresolved.",
                },
            },
            "required": ["question", "ambiguous_field"],
        },
    },
}

_SYSTEM_PROMPT = (
    "You translate a natural-language fund-aggregation question into a structured "
    "query for a deterministic pipeline. Call record_query_spec if the question maps "
    "cleanly onto its fields. Call reject_unsupported_query if the schema has no "
    "field that could ever represent what is being asked - e.g. a different metric "
    "like total AUM, or a filter dimension this schema does not track at all, such "
    "as geography or fund manager. Call ask_clarifying_question if every field the "
    "question needs DOES exist in the schema, but the wording does not clearly "
    "resolve to one value for at least one of them. This includes two distinct "
    "shapes, both count: (1) partial - some filter is stated but another dimension "
    "is left out, e.g. 'expense ratio for ESG funds' never says whether closed "
    "funds should count, so status is ambiguous, not absent; and (2) bare - no "
    "filter is stated on either dimension at all, e.g. 'what is the weighted "
    "average expense ratio' with no mention of ESG or status. A bare question is "
    "NOT the same as a deliberate 'no constraint' request - the schema's None "
    "meaning 'no constraint' is a value the user must actually communicate, not a "
    "default to assume from silence, so a bare question is ambiguous on BOTH "
    "is_esg and status, not resolved to null on both. Never answer in free text, "
    "and never guess a value, including None/no-constraint, for a field the "
    "question does not actually specify. Different wordings of the same underlying "
    "request must produce the same result - focus on what is actually being asked, "
    "not the specific phrasing used to ask it."
)


def parse_query(
    question: str,
    *,
    client: Groq | None = None,
    model: str = DEFAULT_MODEL,
) -> QuerySpec | ClarificationNeeded:
    """Translate *question* into a QuerySpec via a forced tool-call, reject it, or ask for clarification.

    Args:
        question: The natural-language question to parse.
        client: Groq client to use. Constructed fresh (`Groq()`) if omitted.
        model: Model to call.

    Returns:
        The parsed QuerySpec if the question was clear. A `ClarificationNeeded`
        (Phase 17) if the question is answerable in principle but its wording
        leaves one field unresolved - this is a valid outcome, not an error;
        callers surface `.question` to the end user and call `parse_query`
        again with a refined question, exactly like any other REASON step.

    Raises:
        ValueError: If `question` is empty/whitespace-only, if the model's
            tool-call arguments aren't valid JSON, or if the model's
            response doesn't include any of the expected tool calls at all.
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

    def _call(model_name: str) -> QuerySpec | ClarificationNeeded | UnsupportedQueryError:
        """Returns a QuerySpec/ClarificationNeeded on success, or an *unraised* UnsupportedQueryError
        instance when the model correctly rejects the question. Returning it (rather than raising)
        is deliberate: call_with_model_fallback treats any exception as "this model failed,
        try the next one," and a correct rejection must never trigger that - it's the right
        answer, not a failure. parse_query raises it itself, once, after the fallback
        machinery has already decided this was the model's final word. ClarificationNeeded
        needs no such treatment - it's a plain dataclass, never an exception, so it can
        never be mistaken for a failure by call_with_model_fallback in the first place.
        """
        response = active_client.chat.completions.create(
            model=model_name,
            max_tokens=400,
            seed=42,
            tools=[_QUERY_SPEC_TOOL_SCHEMA, _REJECT_TOOL_SCHEMA, _CLARIFY_TOOL_SCHEMA],
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

            if call.function.name == _CLARIFY_TOOL_NAME:
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Model returned malformed tool-call arguments: {exc}") from exc
                return ClarificationNeeded(
                    question=str(args["question"]),
                    ambiguous_field=args["ambiguous_field"],
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
            "Model response did not include any of the expected tool calls "
            f"({_QUERY_SPEC_TOOL_NAME!r}, {_REJECT_TOOL_NAME!r}, or {_CLARIFY_TOOL_NAME!r})"
        )

    result = call_with_model_fallback(_call, models=models_to_try(model))
    if isinstance(result, UnsupportedQueryError):
        raise result
    return result


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
    require_groq_api_key("python -m src.agentic.query_parser")

    client = Groq()

    print("=== Five paraphrases of the original example question ===\n")
    specs: list[QuerySpec | ClarificationNeeded] = []
    for question in _PARAPHRASED_QUESTIONS:
        spec = parse_query(question, client=client)
        specs.append(spec)
        print(f"Q: {question}")
        print(f"   -> {spec}\n")

    all_identical = all(spec == specs[0] for spec in specs)
    all_query_specs = all(isinstance(spec, QuerySpec) for spec in specs)
    print(f"All 5 QuerySpecs identical: {all_identical} (all resolved directly, no clarification: {all_query_specs})\n")

    print("=== Deliberately out-of-scope question ===\n")
    print(f"Q: {_OUT_OF_SCOPE_QUESTION}")
    try:
        spec = parse_query(_OUT_OF_SCOPE_QUESTION, client=client)
        print(f"   -> UNEXPECTEDLY ACCEPTED as {spec} (this should have been rejected)")
    except UnsupportedQueryError as exc:
        print(f"   -> Rejected, as expected: {exc.reason}")


# --- Phase 17 stopping-condition demo ------------------------------------------

# Genuinely ambiguous on purpose: every field this needs (is_esg, status)
# exists in QuerySpec, but the wording never says whether closed funds
# should count - unlike _OUT_OF_SCOPE_QUESTION, where no field could ever
# represent what's being asked at all.
_AMBIGUOUS_QUESTION = "Expense ratio for ESG funds"


def _run_phase_17_demo() -> None:
    """Re-run Phase 10's regression set, then prove a genuinely ambiguous question asks back instead of guessing."""
    require_groq_api_key("python -m src.agentic.query_parser")

    print("=" * 70)
    print("PHASE 10 REGRESSION (must be unaffected by the new clarification tool)")
    print("=" * 70)
    _run_phase_10_demo()

    print("\n" + "=" * 70)
    print("PHASE 17: genuinely ambiguous question")
    print("=" * 70)
    client = Groq()
    print(f"Q: {_AMBIGUOUS_QUESTION}")
    result = parse_query(_AMBIGUOUS_QUESTION, client=client)
    if isinstance(result, ClarificationNeeded):
        print(f"   -> ClarificationNeeded(ambiguous_field={result.ambiguous_field!r}, question={result.question!r})")
        print(f"\nCorrectly asked for clarification instead of guessing: {result.ambiguous_field == 'status'}")
    else:
        print(f"   -> UNEXPECTEDLY resolved directly as {result!r} (expected ClarificationNeeded on 'status')")


if __name__ == "__main__":
    _run_phase_17_demo()
