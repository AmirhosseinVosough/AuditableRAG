"""Phase 17: regression tests for the query-clarification outcome, fast and GROQ_API_KEY-free.

Two layers, matching how test_agentic_orchestrator.py splits its own concerns:

1. `query_parser.parse_query` itself - does a tool call to `ask_clarifying_question`
   actually produce a `ClarificationNeeded` with the right fields? Here the Groq
   client's `chat.completions.create` is faked directly (returning a fabricated
   tool-call response), since this is the one place in the whole test suite that
   needs to prove parse_query's own tool-selection logic, not just how a caller
   reacts to its result.

2. `agent_orchestrator.run_agentic_pipeline` - given a `ClarificationNeeded` (faked
   at the `parse_query` call site, same style test_agentic_orchestrator.py already
   uses for `QuerySpec`), does the orchestrator stop at REASON, on both the
   synthetic and real dispatch paths, without ever reaching ACT-scope - and does it
   still get written to the run log like every other outcome?
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.agentic.agent_orchestrator import run_agentic_pipeline
from src.agentic.query_parser import ClarificationNeeded, QuerySpec, parse_query


def _fake_tool_call_response(tool_name: str, arguments: dict[str, object]) -> SimpleNamespace:
    """Build a minimal stand-in for a Groq chat-completion response with one tool call.

    Only shaped as deep as `query_parser._call` actually reads:
    `response.choices[0].message.tool_calls[i].function.{name,arguments}`.
    """
    call = SimpleNamespace(function=SimpleNamespace(name=tool_name, arguments=json.dumps(arguments)))
    message = SimpleNamespace(tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


# --- Layer 1: parse_query's own tool-selection logic ---------------------------


def test_ambiguous_question_returns_clarification_needed() -> None:
    """A tool call to ask_clarifying_question produces a ClarificationNeeded, not a QuerySpec."""
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_tool_call_response(
        "ask_clarifying_question",
        {"question": "Should closed funds be included?", "ambiguous_field": "status"},
    )

    result = parse_query("Expense ratio for ESG funds", client=client)

    assert isinstance(result, ClarificationNeeded)
    assert result.question == "Should closed funds be included?"
    assert result.ambiguous_field == "status"


def test_clear_question_still_returns_query_spec_unaffected_by_the_new_tool() -> None:
    """Adding a third tool doesn't change behavior for a question the model resolves directly."""
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_tool_call_response(
        "record_query_spec",
        {
            "is_esg": True,
            "status": "active",
            "closed_quarter_exclusions": ["Q3"],
            "requested_metric": "weighted_average_expense_ratio",
        },
    )

    result = parse_query(
        "weighted average expense ratio for ESG funds, excluding funds closed in Q3", client=client
    )

    assert isinstance(result, QuerySpec)
    assert result.is_esg is True
    assert result.status == "active"


# --- Layer 2: the orchestrator's REASON-step branch, both dispatch paths -------


def test_synthetic_orchestrator_stops_at_reason_on_clarification_needed(tmp_path: Path) -> None:
    clarification = ClarificationNeeded(question="Should closed funds be included?", ambiguous_field="status")

    with patch("src.agentic.agent_orchestrator.parse_query", return_value=clarification):
        result = run_agentic_pipeline(
            "Expense ratio for ESG funds", client=MagicMock(), run_log_dir=tmp_path
        )

    assert result["needs_clarification"] is True
    assert result["clarifying_question"] == "Should closed funds be included?"
    assert result["ambiguous_field"] == "status"
    # Nothing past REASON ran - this is the smaller shape, not the full one
    # padded with empty defaults.
    assert "included_funds" not in result
    assert "query_spec" not in result

    reason_entries = [e for e in result["decision_trace"] if e["step"] == "REASON"]
    assert len(reason_entries) == 1
    assert "status" in reason_entries[0]["detail"]

    # Still written to the run log, same as every other terminal outcome.
    logged_files = list(tmp_path.glob("run_agentic_*.json"))
    assert len(logged_files) == 1


def test_real_orchestrator_stops_at_reason_on_clarification_needed(tmp_path: Path) -> None:
    """Same branch, real-data dispatch - proves ACT-scope (load_real_pdfs et al.) never even runs."""
    clarification = ClarificationNeeded(question="Should closed funds be included?", ambiguous_field="status")

    with patch("src.agentic.agent_orchestrator.parse_query", return_value=clarification):
        # raw_pdf_dir points at an empty directory - if ACT-scope ran at all,
        # load_real_pdfs would just return zero candidates rather than error,
        # so an empty dir alone wouldn't prove REASON stopped the run early.
        # The real proof is `included_funds`/`excluded_funds` being absent
        # below - those keys only exist once ACT-scope has actually run.
        result = run_agentic_pipeline(
            "Expense ratio for ESG funds",
            source="real",
            raw_pdf_dir=tmp_path,
            client=MagicMock(),
            run_log_dir=tmp_path,
        )

    assert result["needs_clarification"] is True
    assert result["ambiguous_field"] == "status"
    assert "included_funds" not in result
    assert "excluded_funds" not in result

    logged_files = list(tmp_path.glob("run_agentic_real_*.json"))
    assert len(logged_files) == 1
