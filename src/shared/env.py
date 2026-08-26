"""Centralized environment/.env loading - every phase demo needs the same two things before a live call.

Every phase's `__main__` demo (`pipeline.py`, `verification.py`,
`field_extraction.py`, `extraction_cascade.py`, `query_parser.py`,
`retrieval.py`, `agent_orchestrator.py`) needs the same two steps before it
can make a real Groq call: load `.env` if present, then confirm
`GROQ_API_KEY` actually ended up set. That was duplicated seven times - a
private `_load_dotenv` re-defined fresh in each file, each computing its own
idea of the project root via `Path(__file__).resolve().parents[N]`, guessing
`N` from that file's own depth in the tree.

Centralizing it here also fixes a real, live bug, not just tidies up
duplication: `query_parser.py`'s copy still used `parents[1]` - one level too
shallow after the `src/` reorg - silently breaking `.env`-based
`GROQ_API_KEY` loading for Phase 10's demo specifically (every other file's
copy had already been corrected to `parents[2]` in an earlier pass, but this
one call site was missed). This module's own file path is fixed at exactly
`src/shared/env.py`, always two levels below the project root - so
`PROJECT_ROOT` is computed correctly once, here, and every caller inherits
that correctness instead of re-deriving (and risking re-breaking) it.

See `.env.example` at the project root for the actual variables this project
uses.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """Populate `os.environ` from a simple KEY=VALUE `.env` file, if present.

    Avoids adding a `python-dotenv` dependency for this one use - same
    minimal parser every per-file copy already had (skip blank lines and
    `#` comments, strip surrounding quotes from values). Existing
    environment variables are never overwritten, so a value already
    exported in the shell always wins over `.env`.

    Args:
        path: `.env` file to read. Defaults to `PROJECT_ROOT / ".env"` -
            pass explicitly only to override that (e.g. a test pointing at
            a temporary `.env` file).
    """
    target = path if path is not None else PROJECT_ROOT / ".env"
    if not target.is_file():
        return
    for line in target.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_groq_api_key(run_command: str) -> None:
    """Load `.env`, then raise `SystemExit` with a clear message if `GROQ_API_KEY` still isn't set.

    Every phase demo's boilerplate was: call `_load_dotenv`, then check
    `os.environ.get("GROQ_API_KEY")`, then raise the same shape of
    `SystemExit` with a slightly different suggested re-run command. This
    collapses that into one call.

    Args:
        run_command: The exact `python -m ...` invocation to suggest
            re-running, e.g. `"python -m src.agentic.query_parser"` - each
            phase demo names its own module here, since that part
            genuinely differs per caller.

    Raises:
        SystemExit: If `GROQ_API_KEY` is not set after loading `.env`.
    """
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Set it in your environment, or copy "
            ".env.example to .env and fill it in, then re-run "
            f"`{run_command}`."
        )
