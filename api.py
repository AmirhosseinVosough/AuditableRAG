"""A minimal FastAPI wrapper around the agentic pipeline - ask a question, get a result, no Python required.

Run it:
    pip install -r requirements.txt
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs in a browser. That auto-generated page
(FastAPI's built-in Swagger UI) is the actual non-Python-friendly interface
this file exists for - it renders a form for the /ask endpoint below, lets
you type a question, click "Execute", and see the full JSON result, with no
curl command and no code written at all.

Deliberately thin, same "no logic of its own" convention the orchestrator
itself follows: this file contains no pipeline/filtering/arithmetic logic -
it only calls `run_agentic_pipeline` and translates its result and known
exceptions into HTTP responses. All the actual work still happens in
`src/agentic/agent_orchestrator.py`, unchanged.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agentic.agent_orchestrator import run_agentic_pipeline
from src.agentic.query_parser import UnsupportedQueryError
from src.shared.env import load_dotenv

# Populate GROQ_API_KEY (and HF_TOKEN, if set) from .env once, at server
# startup - not per-request. Same file/behavior every __main__ demo already
# relies on (see src/shared/env.py); a missing key still fails loudly, just
# on the first request rather than at import time, via Groq()'s own error.
load_dotenv()

app = FastAPI(
    title="Fund Aggregation Pipeline",
    description=(
        "Ask a fund-aggregation question (e.g. \"weighted average expense ratio for "
        "ESG active funds\") and get back the full agentic pipeline's result: which "
        "funds were included/excluded/flagged for human review, the final number, and "
        "the complete decision trace - the same JSON that would otherwise be printed "
        "to a terminal or written to outputs/run_logs/, as one HTTP response."
    ),
)


class QuestionRequest(BaseModel):
    question: str
    source: Literal["synthetic", "real"] = "synthetic"


@app.get("/")
def root() -> dict[str, str]:
    """Landing message - points a visitor at the interactive docs, the actual intended entry point."""
    return {
        "message": "Fund Aggregation Pipeline API is running.",
        "try_it": "Open /docs in your browser for an interactive form - no code required.",
    }


@app.post("/ask")
def ask(request: QuestionRequest) -> dict:
    """Run the full agentic pipeline for *request.question* and return its complete result.

    A synchronous endpoint on purpose ('def', not 'async def'): FastAPI runs
    plain 'def' path operations in a worker thread automatically, so this
    call - which can take anywhere from a few seconds to roughly a minute,
    since it's making several real, sequential LLM calls - doesn't block the
    whole server from handling other requests while it runs. No background-
    job/polling setup needed to get that for free.

    Args:
        request.question: Any natural-language fund-aggregation question the
            pipeline supports, e.g. "weighted average expense ratio for ESG
            funds, excluding funds closed in Q3".
        request.source: "synthetic" (default) - answers against the built-in,
            always-available synthetic fixture set with a known-correct
            answer; good for a first try with zero setup. "real" - answers
            against data/user_uploads/ (see README.md for adding your own
            PDFs there).

    Returns:
        The exact dict `run_agentic_pipeline` returns - included_funds,
        excluded_funds, needs_human_review, final_answer, decision_trace,
        llm_calls, total_tokens, estimated_cost_usd, etc. Nothing is
        filtered or reshaped here.

    Raises:
        HTTPException(400): If the question doesn't map to a supported
            QuerySpec (never silently guessed - the model's own stated
            reason is passed straight through), or if `source` is invalid.
    """
    try:
        return run_agentic_pipeline(request.question, source=request.source)
    except UnsupportedQueryError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"This question isn't supported: {exc.reason}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
