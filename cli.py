"""An interactive command-line loop around the agentic pipeline - ask, clarify, re-ask, answer.

Run it:
    pip install -r requirements.txt
    cp .env.example .env                # fill in GROQ_API_KEY
    python -m cli                       # real-data mode (data/user_uploads/)
    python -m cli --source synthetic    # built-in fixture set instead
    python -m cli "weighted average expense ratio for ESG active funds"

Why this file exists: `query_parser.parse_query` can judge a question
answerable in principle but too vague to resolve to a single `QuerySpec` -
it returns a `ClarificationNeeded` (Phase 17) rather than guessing. Nothing
inside the pipeline consumes an answer to that clarifying question:
`parse_query` and `run_agentic_pipeline` are single-shot by design (see
AGENTIC_LAYER_BUILD_PROMPT.md Phase 17 - *"Bounding how many rounds of
back-and-forth a caller allows (a UI, a CLI script) is a caller-level policy
choice, outside this phase's scope"*). This file is that caller-level
policy: a bounded loop that surfaces the clarifying question, reads a
human's answer, folds it into the question text, and re-runs - up to
`--max-rounds` times before giving up.

Deliberately thin, same "no logic of its own" convention `api.py` follows:
no pipeline / filtering / arithmetic / question-interpretation logic here.
The clarifying answer is folded back in by plain string concatenation, never
a second LLM call to "merge" it - the model still only ever extracts
structure from text, per the core mandate. All the real work stays in
`src/agentic/agent_orchestrator.py`, unchanged; `_print_result` is imported
from there rather than reimplemented, so the terminal output format matches
that module's own __main__ demos exactly.

Note: `run_agentic_pipeline` writes a run log for every call, so a session
that takes N clarification rounds leaves N+1 files in `outputs/run_logs/`
(one per round, plus the final answer) - each a complete, self-explanatory
record of that step, exactly as every other pipeline outcome is.
"""

from __future__ import annotations

import argparse
import sys

from src.agentic.agent_orchestrator import _print_result, run_agentic_pipeline
from src.agentic.query_parser import UnsupportedQueryError
from src.shared.env import require_groq_api_key

# Caller-level round cap - the same "iteration caps are enforced in code, not
# left to the model" principle every retry loop in this project follows,
# applied to the clarification back-and-forth. Nothing here can loop forever:
# even if the model asked a brand-new clarifying question every single round,
# the loop still stops after this many.
DEFAULT_MAX_ROUNDS = 3


def _fold_in_clarification(question: str, ambiguous_field: str, answer: str) -> str:
    """Append the human's clarifying answer to the question text - no LLM call.

    Plain concatenation on purpose: the next `parse_query` re-reads the
    combined text and extracts structure from it exactly as it would from
    any other phrasing. Naming the `ambiguous_field` in the appended line
    just gives the model an unambiguous anchor for what the answer resolves
    ("status: only active" rather than a bare "only active" it would have to
    re-associate with the right field itself).
    """
    return f"{question}\n\n(Clarification - {ambiguous_field}: {answer})"


def _prompt(text: str) -> str:
    """input() that treats EOF (Ctrl-D) as a clean exit instead of a traceback."""
    try:
        return input(text)
    except EOFError:
        print()
        sys.exit(0)


def run_cli(question: str | None, *, source: str, max_rounds: int) -> int:
    """Ask -> (clarify -> re-ask)* -> answer. Returns a process exit code.

    Args:
        question: The first question. If None, it is read from stdin.
        source: "real" (reads `data/user_uploads/`) or "synthetic" (the
            built-in fixture set) - passed straight through to
            `run_agentic_pipeline`.
        max_rounds: Hard cap on how many times a `ClarificationNeeded`
            outcome is allowed to send the question back to the model.

    Returns:
        0 if the question resolved (to an answer, or to a clean, expected
        "nothing qualified"); 1 if it was rejected as unsupported, was
        empty, or stayed ambiguous through every allowed round.
    """
    if question is None:
        question = _prompt("Question: ").strip()

    # max_rounds clarifying answers -> up to max_rounds + 1 pipeline calls: the
    # first ask, then one re-run per answer given. The final re-run (after the
    # last allowed answer) still gets a chance to resolve; it just can't prompt
    # for another answer nothing would act on.
    for answered in range(max_rounds + 1):
        try:
            result = run_agentic_pipeline(question, source=source)
        except UnsupportedQueryError as exc:
            # A correct rejection, not a crash - the model's own stated
            # reason, passed straight through (same as api.py does).
            print(f"\nNot supported: {exc.reason}")
            return 1
        except ValueError as exc:
            print(f"\n{exc}")
            return 1

        if not result.get("needs_clarification"):
            print()
            _print_result(result)
            return 0

        if answered == max_rounds:
            break

        # ClarificationNeeded: show it, read a human's answer, fold it into
        # the question, loop back and re-run with the refined text.
        print("\n" + "-" * 68)
        print(result["clarifying_question"])
        answer = _prompt("> ").strip()
        if not answer:
            print("\nNo answer given - stopping.")
            return 1
        question = _fold_in_clarification(question, result["ambiguous_field"], answer)

    print(
        f"\nStill ambiguous after {max_rounds} clarification round(s). Try asking "
        "again from scratch with the ESG and active/closed scope stated explicitly."
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m cli",
        description=(
            "Ask the agentic fund-aggregation pipeline a question and answer its "
            "clarifying questions in a bounded loop."
        ),
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="The question to ask. Omit to be prompted for it interactively.",
    )
    parser.add_argument(
        "--source",
        choices=("real", "synthetic"),
        default="real",
        help="'real' (default) reads data/user_uploads/; 'synthetic' uses the built-in fixture set.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
        metavar="N",
        help=f"Clarification rounds allowed before giving up (default {DEFAULT_MAX_ROUNDS}).",
    )
    args = parser.parse_args()

    if args.max_rounds < 1:
        parser.error("--max-rounds must be at least 1")

    # Same startup step every __main__ demo in this project uses: load .env,
    # then exit early with a clear message if GROQ_API_KEY still isn't set -
    # this is an interactive tool for a human, so a friendly "copy .env.example"
    # message beats a raw Groq() traceback on the first call (api.py skips this
    # only because a server shouldn't refuse to start over a missing key).
    require_groq_api_key("python -m cli")

    try:
        sys.exit(run_cli(args.question, source=args.source, max_rounds=args.max_rounds))
    except KeyboardInterrupt:
        print()
        sys.exit(130)


if __name__ == "__main__":
    main()
