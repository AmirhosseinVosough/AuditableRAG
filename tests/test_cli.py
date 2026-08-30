"""Fast, GROQ_API_KEY-free tests for `cli.run_cli` - the interactive clarification loop.

Same split-of-concerns idea `test_query_clarification.py` uses: that file proves
`parse_query` raises a `ClarificationNeeded` and that `run_agentic_pipeline` stops
at REASON on one. This file proves the layer above that - the `cli.py` loop that a
human actually drives: given a run that keeps asking for clarification, does the
loop surface each question, fold each typed answer back into the question text,
re-run, and stop at the round cap rather than forever?

`cli.run_agentic_pipeline` and `cli._prompt` are both faked here, so no Groq client
is ever constructed and no stdin is read - the loop's control flow is the entire
subject. `_prompt` (not `builtins.input`) is the patch point on purpose: it's the
one seam `cli.py` funnels every interactive read through, exactly so a test can
replace it without touching `input` globally.
"""

from __future__ import annotations

from unittest.mock import patch

import cli

_RESOLVED_RESULT = {
    "question": "q",
    "query_spec": {
        "is_esg": True,
        "status": "active",
        "closed_quarter_exclusions": [],
        "requested_metric": "weighted_average_expense_ratio",
    },
    "final_answer": 0.1234,
    "included_funds": [],
    "excluded_funds": [],
    "needs_human_review": [],
    "hard_stopped": False,
    "hard_stop_reason": None,
    "decision_trace": [],
}


def _clarification(field: str = "is_esg", question: str = "ESG, non-ESG, or all funds?") -> dict:
    return {"needs_clarification": True, "clarifying_question": question, "ambiguous_field": field}


class _FakePipeline:
    """Returns a queued result per call and records the question string each call got."""

    def __init__(self, results: list[dict]) -> None:
        self._results = results
        self.questions: list[str] = []
        self.sources: list[str] = []

    def __call__(self, question: str, *, source: str) -> dict:
        self.questions.append(question)
        self.sources.append(source)
        return self._results[len(self.questions) - 1]


def test_clear_question_resolves_in_one_call() -> None:
    """No clarification -> one pipeline call, exit 0, no prompt for an answer."""
    pipeline = _FakePipeline([_RESOLVED_RESULT])

    with patch("cli.run_agentic_pipeline", pipeline), patch("cli._prompt") as prompt:
        rc = cli.run_cli("weighted average expense ratio for ESG active funds", source="real", max_rounds=3)

    assert rc == 0
    assert len(pipeline.questions) == 1
    prompt.assert_not_called()


def test_clarification_answer_is_folded_into_the_next_question() -> None:
    """Each typed answer is appended to the question text and the loop re-runs with it."""
    pipeline = _FakePipeline([_clarification("is_esg"), _clarification("status"), _RESOLVED_RESULT])

    with patch("cli.run_agentic_pipeline", pipeline), patch("cli._prompt", side_effect=["esg funds", "only active"]):
        rc = cli.run_cli("whats the average expense ratio", source="real", max_rounds=3)

    assert rc == 0
    assert len(pipeline.questions) == 3
    # First call: the raw question. Later calls: the accumulated text.
    assert pipeline.questions[0] == "whats the average expense ratio"
    assert "(Clarification - is_esg: esg funds)" in pipeline.questions[1]
    assert "(Clarification - is_esg: esg funds)" in pipeline.questions[2]
    assert "(Clarification - status: only active)" in pipeline.questions[2]
    # `source` is threaded through unchanged on every re-run.
    assert pipeline.sources == ["real", "real", "real"]


def test_clarifying_question_is_shown_to_the_user(capsys) -> None:
    pipeline = _FakePipeline([_clarification(question="Should closed funds count?"), _RESOLVED_RESULT])

    with patch("cli.run_agentic_pipeline", pipeline), patch("cli._prompt", side_effect=["active only"]):
        cli.run_cli("expense ratio for esg funds", source="synthetic", max_rounds=3)

    assert "Should closed funds count?" in capsys.readouterr().out


def test_loop_stops_at_the_round_cap_when_never_resolved() -> None:
    """max_rounds=3 -> at most 3 answers read and 4 pipeline calls, then give up with exit 1."""
    pipeline = _FakePipeline([_clarification()] * 10)

    with patch("cli.run_agentic_pipeline", pipeline), patch(
        "cli._prompt", side_effect=["a", "b", "c", "d", "e"]
    ) as prompt:
        rc = cli.run_cli("vague", source="real", max_rounds=3)

    assert rc == 1
    assert len(pipeline.questions) == 4  # first ask + one re-run per answer
    assert prompt.call_count == 3  # never prompts for an answer it can't act on


def test_empty_answer_stops_the_loop() -> None:
    pipeline = _FakePipeline([_clarification()])

    with patch("cli.run_agentic_pipeline", pipeline), patch("cli._prompt", side_effect=[""]):
        rc = cli.run_cli("vague", source="real", max_rounds=3)

    assert rc == 1
    assert len(pipeline.questions) == 1


def test_unsupported_query_is_reported_not_raised(capsys) -> None:
    err = cli.UnsupportedQueryError(question="q", reason="total AUM is not a supported metric")

    with patch("cli.run_agentic_pipeline", side_effect=err):
        rc = cli.run_cli("total AUM of all ESG funds", source="real", max_rounds=3)

    assert rc == 1
    assert "total AUM is not a supported metric" in capsys.readouterr().out


def test_first_question_is_read_from_prompt_when_not_given_as_arg() -> None:
    pipeline = _FakePipeline([_RESOLVED_RESULT])

    with patch("cli.run_agentic_pipeline", pipeline), patch("cli._prompt", side_effect=["typed at the prompt"]):
        rc = cli.run_cli(None, source="synthetic", max_rounds=3)

    assert rc == 0
    assert pipeline.questions == ["typed at the prompt"]
